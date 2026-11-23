"""Synthetic review tasks, version 2.

Each problem group holds one stated problem, one worked derivation and one short
program. The derivation and the program are corrupted independently, so the two
stages of a group have different answers:

    analysis stage  judges the derivation against the stated problem
    code stage      judges the program against the stated problem

Three design rules replace the version-1 generator, whose prompts named the
injected error in plain words:

1. A derivation shows numbers and steps only. Finding its error means checking
   those numbers against the stated problem.
2. Every error has several written forms, and every form has a harmless twin
   that looks similar (for example swapped limit reads plus a reversed step,
   which cancel out). A fixed substring therefore does not identify the label.
3. One derivation layout and one program layout per family are kept out of
   train/val/test and appear only in the test_template split.

Problems use small integers so that the correct value can be worked out by
hand; no stage needs more than integer arithmetic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import object_sha, write_json, write_jsonl
from .verify import NumericInterpreter, check_completion, hidden_rebindings, numeric_check

TASK_VERSION = 2

SEEN_STYLES = {
    'integral': {'derivation': ['antiderivative', 'termwise'], 'code': ['slices', 'loop']},
    'linear': {'derivation': ['cramer', 'adjugate'], 'code': ['rank_guard', 'det_guard']},
}
HELDOUT_STYLES = {
    'integral': {'derivation': ['limits_table'], 'code': ['weight_vector']},
    'linear': {'derivation': ['substitution'], 'code': ['default_none']},
}

# Balanced decks. Accept-everything scores about one third on either stage.
DERIVATION_DECKS = {
    'integral': ['ok', 'integration_sign', 'antiderivative', 'ok', 'arithmetic',
                 'integration_sign', 'ok', 'antiderivative', 'arithmetic'],
    'well_small': ['ok', 'rhs_scale', 'arithmetic'],
    'well_scaled': ['ok', 'rhs_scale'],
    'ill': ['ok', 'rhs_scale'],
    'singular': ['ok', 'singular_system'],
}
CODE_DECKS = {
    'integral': ['ok', 'integration_sign', 'simpson_weights'],
    'regular': ['ok', 'rhs_scale', 'singular_system'],
    'singular': ['ok', 'singular_system'],
}
LINEAR_SUBKINDS = ['well_small', 'well_scaled', 'ill', 'singular']

GLOSSARY = ('Issue enums: ok; integration_sign (limits or step direction reversed); '
            'antiderivative (a term integrated wrongly); arithmetic (set-up right, a number miscalculated); '
            'simpson_weights (wrong Simpson node weights or node ranges); '
            'rhs_scale (right-hand side or solution scaled); singular_system (singularity of A mishandled).')
ANALYSIS_INSTRUCTION = (
    'The derivation and the program were written separately; either one may contain an error. '
    'Review the DERIVATION against the stated problem and give a brief explanation. End with one line '
    'FINAL={"value":number_or_list_or_null,"condition":"one_enum","verdict":"accept_or_revise","issue":"one_issue"}. '
    'value is the correct answer to the stated problem, which may differ from the submitted one; use null when A is singular. '
    'verdict and issue describe the derivation only. '
    'Condition enums: not_applicable (integrals), well_conditioned, ill_conditioned, singular. ' + GLOSSARY)
CODE_INSTRUCTION = (
    'The derivation and the program were written separately; either one may contain an error. '
    'Review the PROGRAM against the stated problem. Return a short top-level Python script only: a corrected version '
    'that reads the supplied read-only dictionary p and defines result, verdict and issue. '
    'verdict and issue describe the submitted program only; use accept and ok when it was already correct. '
    'Do not overwrite p or its fields. The same script will be rerun on hidden parameter rebindings of the same algorithm family. '
    'For a singular system result=None. Use numpy as np if needed. '
    'No file/network access, user functions or while loops; bounded for/range is accepted. ' + GLOSSARY)


# ---------------------------------------------------------------- small helpers

def pick(rng, options):
    return options[int(rng.integers(len(options)))]


def fmt(value):
    """Integer-valued numbers print without a decimal point."""
    value = float(value)
    return str(int(value)) if value == int(value) else repr(round(value, 4))


def par(value):
    text = fmt(value)
    return f'({text})' if text.startswith('-') else text


def vec(values):
    return '[' + ', '.join(fmt(v) for v in values) + ']'


def times(left, right):
    return f'{par(left)}*{par(right)}'


def poly_text(terms, merge=True):
    """Render [(coefficient, power), ...] as text such as '3 - 4*x + 6*x**2'."""
    if merge:
        merged = {}
        for coefficient, power in terms:
            merged[power] = merged.get(power, 0) + coefficient
        terms = sorted(((c, pw) for pw, c in merged.items()), key=lambda item: item[1])
    parts = []
    for coefficient, power in terms:
        if coefficient == 0:
            continue
        size = fmt(abs(coefficient))
        if power == 0:
            body = size
        else:
            variable = 'x' if power == 1 else f'x**{power}'
            body = variable if size == '1' else f'{size}*{variable}'
        parts.append(('-' if coefficient < 0 else '+', body))
    if not parts:
        return '0'
    text = ('-' if parts[0][0] == '-' else '') + parts[0][1]
    for sign, body in parts[1:]:
        text += f' {sign} {body}'
    return text


def split_for(index, count):
    if count % 6:
        raise ValueError('group counts must be divisible by six')
    return 'train' if index < 2 * count // 3 else 'val' if index < 5 * count // 6 else 'test'


class Deck:
    """Shuffled cycling through a balanced list, so small pools still see every card."""

    def __init__(self, cards, rng):
        self.cards = list(cards)
        self.rng = rng
        self.queue = []

    def draw(self):
        if not self.queue:
            order = self.rng.permutation(len(self.cards))
            self.queue = [self.cards[int(i)] for i in order]
        return self.queue.pop()


# ---------------------------------------------------------------- references

def integral_value(p):
    a, b = p['a'], p['b']
    return float(sum(c * (b ** (j + 1) - a ** (j + 1)) / (j + 1) for j, c in enumerate(p['coefficients'])))


def reference(p):
    """Private reference for a public problem; also used for hidden rebindings."""
    if p['kind'] == 'integral':
        return {'value': integral_value(p), 'condition': 'not_applicable', 'atol': 2e-5, 'rtol': 1e-5}
    A = np.asarray(p['A'], dtype=float)
    b = np.asarray(p['b'], dtype=float)
    singular = int(np.linalg.matrix_rank(A)) < len(b)
    cond = float(np.linalg.cond(A))
    loose = cond > 1e3
    return {'value': None if singular else np.linalg.solve(A, b).tolist(),
            'condition': 'singular' if singular else 'ill_conditioned' if loose else 'well_conditioned',
            'condition_number': None if singular else cond,
            'atol': 1e-5 if loose else 1e-7, 'rtol': 1e-4 if loose else 1e-6, 'residual_tolerance': 1e-8}


def mathematical_hash(p):
    """Identity of the underlying mathematics; discretization and submitted candidates are excluded."""
    keys = ['kind', 'coefficients', 'a', 'b'] if p['kind'] == 'integral' else ['kind', 'A', 'b']
    return object_sha({k: p[k] for k in keys})


def problem_text(p):
    if p['kind'] == 'integral':
        integrand = poly_text([(c, j) for j, c in enumerate(p['coefficients'])])
        return (f"Integrate f(x) = {integrand} from a={fmt(p['a'])} to b={fmt(p['b'])}. "
                f"Use composite Simpson with n={p['n']} even subintervals.")
    A = '[' + ', '.join(vec(row) for row in p['A']) + ']'
    return (f"For A={A} and b={vec(p['b'])}, determine the unique solution to A x=b. Return no vector when A is singular. "
            'Classify condition as well_conditioned if cond2(A)<=1000, ill_conditioned if cond2(A)>1000, or singular when rank deficient.')


# ---------------------------------------------------------------- integral family

def make_integral(rng):
    """Polynomial with c_j = (j+1)*k_j, so the antiderivative and its endpoint values are integers."""
    degree = int(rng.integers(1, 4))
    k = [int(rng.integers(-5, 6)), 0, 0, 0]
    for j in range(1, degree + 1):
        k[j] = int(rng.integers(-3, 4))
    if k[degree] == 0:
        k[degree] = pick(rng, [-2, -1, 1, 2])
    low, high = sorted(int(v) for v in rng.choice(np.arange(-2, 4), size=2, replace=False))
    a, b = (high, low) if rng.random() < 0.25 else (low, high)
    return {'kind': 'integral', 'integrand': 'cubic', 'coefficients': [float(k[j] * (j + 1)) for j in range(4)],
            'a': float(a), 'b': float(b), 'n': pick(rng, [8, 12, 16, 24]), 'method': 'composite_simpson'}


def antiderivative_terms(p):
    return [(int(c) // (j + 1), j + 1) for j, c in enumerate(p['coefficients']) if c != 0]


def evaluate_terms(terms, x):
    return sum(c * int(x) ** power for c, power in terms)


def integral_check_text(p):
    a, b = int(p['a']), int(p['b'])
    terms = antiderivative_terms(p)
    upper, lower = evaluate_terms(terms, b), evaluate_terms(terms, a)
    return (f'Check: F(x) = {poly_text(terms)}; F({fmt(b)}) = {upper}; F({fmt(a)}) = {lower}; '
            f'F({fmt(b)}) - F({fmt(a)}) = {upper - lower}.')


def integral_derivation(p, issue, style, rng):
    """Worked analytic derivation. Returns None when the requested error would not change the result."""
    a, b = int(p['a']), int(p['b'])
    true_terms = antiderivative_terms(p)
    terms = list(true_terms)
    diagnosis = 'The submitted derivation agrees with this check.'
    if issue == 'antiderivative':
        index = pick(rng, [i for i, (_, power) in enumerate(true_terms) if power >= 2])
        coefficient, power = true_terms[index]
        # Either the division by the new power is skipped, or the power is not raised.
        terms[index] = (coefficient * power, power) if rng.random() < 0.5 else (coefficient, power - 1)
        diagnosis = (f'The submitted antiderivative mishandles the term {poly_text([(coefficient * power, power - 1)])}; '
                     f'it integrates to {poly_text([(coefficient, power)])}.')
    upper, lower = (a, b) if issue == 'integration_sign' else (b, a)
    if issue == 'integration_sign':
        diagnosis = f'The submitted derivation subtracts in the reverse order for an integral from {fmt(a)} to {fmt(b)}.'
    value_upper, value_lower = evaluate_terms(terms, upper), evaluate_terms(terms, lower)
    result = value_upper - value_lower
    slip = pick(rng, [-10, -2, -1, 1, 2, 10]) if issue == 'arithmetic' else 0
    if issue == 'arithmetic':
        diagnosis = f'The submitted set-up is right but a number is miscalculated; the result should be {result}.'

    if style == 'termwise':
        pieces = [c * (upper ** power - lower ** power) for c, power in terms]
        where = pick(rng, ['term', 'final']) if slip else None
        if where == 'term':
            pieces[int(rng.integers(len(pieces)))] += slip
        claim = sum(pieces) + (slip if where == 'final' else 0)
        integrand_terms = [(c, j) for j, c in enumerate(p['coefficients']) if c != 0]
        lines = []
        for (source, (c, power), piece) in zip(integrand_terms, terms, pieces):
            span = (f'({par(upper)} - {par(lower)})' if power == 1 else
                    f'({par(upper)}**{power} - {par(lower)}**{power})')
            lines.append(f'{poly_text([source])}: {par(c)}*{span} = {piece}.')
        text = '\n'.join(lines + [f'Sum of terms = {claim}.'])
    else:
        where = pick(rng, ['upper', 'lower', 'final']) if slip else None
        shown_upper = value_upper + (slip if where == 'upper' else 0)
        shown_lower = value_lower + (slip if where == 'lower' else 0)
        claim = shown_upper - shown_lower + (slip if where == 'final' else 0)
        formula = poly_text(terms)
        if style == 'antiderivative':
            endpoints = [f'F({fmt(upper)}) = {shown_upper}.', f'F({fmt(lower)}) = {shown_lower}.']
            if rng.random() < 0.5:
                endpoints.reverse()
            last = (f'Integral = F({fmt(upper)}) - F({fmt(lower)}) = {claim}.' if rng.random() < 0.6 else
                    f'Integral = -F({fmt(lower)}) + F({fmt(upper)}) = {claim}.')
            text = ' '.join([f'F(x) = {formula}.'] + endpoints + [last])
        else:  # limits_table, held out
            text = (f'Take G(x) = {formula}. At the upper limit x = {fmt(upper)}, G = {shown_upper}. '
                    f'At the lower limit x = {fmt(lower)}, G = {shown_lower}. The integral equals {claim}.')
    true_value = evaluate_terms(true_terms, b) - evaluate_terms(true_terms, a)
    if (claim == true_value) != (issue == 'ok'):
        return None
    return {'text': text, 'claim': float(claim), 'diagnosis': diagnosis}


def integral_code(issue, style, rng):
    """Returns (submitted, corrected). Sign is right exactly when the limit reads and the step agree."""
    h = pick(rng, ['h', 'step', 'dx'])
    y = pick(rng, ['y', 'fx', 'vals'])
    x = pick(rng, ['x', 't', 'nodes'])
    total = pick(rng, ['total', 'acc', 's'])
    w = pick(rng, ['w', 'wt', 'weights'])
    swapped_reads = rng.random() < 0.5
    sign_site = pick(rng, ['reads', 'step'])
    step_form = int(rng.integers(2))
    nodes_form = int(rng.integers(2))
    horner = rng.random() < 0.4
    factor_form = int(rng.integers(2))

    if style == 'slices':
        ok_weights = [f'4*np.sum({y}[1:-1:2])+2*np.sum({y}[2:-1:2])',
                      f'2*np.sum({y}[2:-1:2])+4*np.sum({y}[1:-1:2])',
                      f'4*np.sum({y}[1::2])+2*np.sum({y}[2:-1:2])']
        bad_weights = [f'3*np.sum({y}[1:-1:2])+2*np.sum({y}[2:-1:2])',
                       f'2*np.sum({y}[1:-1:2])+4*np.sum({y}[2:-1:2])',
                       f'4*np.sum({y}[1:-1:2])+2*np.sum({y}[2::2])',
                       f'4*np.sum({y}[2:-1:2])+2*np.sum({y}[1:-1:2])']
    elif style == 'loop':
        def loop(header, test, first, second):
            return [f'{total}={y}[0]+{y}[n]', header, f'    if {test}:', f'        {total}={total}+{first}*{y}[i]',
                    '    else:', f'        {total}={total}+{second}*{y}[i]']
        inner = 'for i in range(1,n):'
        ok_weights = [loop(inner, 'i%2==1', 4, 2), loop(inner, 'i%2==0', 2, 4)]
        bad_weights = [loop(inner, 'i%2==0', 4, 2), loop(inner, 'i%2==1', 2, 4),
                       loop(inner, 'i%2==1', 3, 2), loop('for i in range(1,n+1):', 'i%2==1', 4, 2)]
    else:  # weight_vector, held out
        def weights(*assignments):
            return [f'{w}=np.ones(n+1)'] + [f'{w}[{span}]={value}' for span, value in assignments]
        ok_weights = [weights(('1:-1:2', 4), ('2:-1:2', 2)), weights(('2:-1:2', 2), ('1::2', 4))]
        bad_weights = [weights(('1:-1:2', 2), ('2:-1:2', 4)), weights(('1:-1:2', 4), ('2::2', 2)),
                       weights(('1:-1:2', 3), ('2:-1:2', 2))]
    ok_choice = pick(rng, ok_weights)
    bad_choice = pick(rng, bad_weights)

    def render(sign_bug, weights_bug):
        reads_swapped = swapped_reads != (sign_bug and sign_site == 'reads')
        step_reversed = swapped_reads != (sign_bug and sign_site == 'step')
        lines = ['import numpy as np',
                 "a=p['b']" if reads_swapped else "a=p['a']",
                 "b=p['a']" if reads_swapped else "b=p['b']",
                 "n=p['n']", "c=p['coefficients']",
                 f'{x}=np.linspace(a,b,n+1)' if nodes_form == 0 else f'{x}=a+(b-a)*np.arange(n+1)/n',
                 f'{y}=c[0]+{x}*(c[1]+{x}*(c[2]+{x}*c[3]))' if horner else f'{y}=c[0]+c[1]*{x}+c[2]*{x}**2+c[3]*{x}**3']
        if step_reversed:
            lines.append(f'{h}=(a-b)/n' if step_form == 0 else f'{h}=-(b-a)/n')
        else:
            lines.append(f'{h}=(b-a)/n' if step_form == 0 else f'{h}=-(a-b)/n')
        chosen = bad_choice if weights_bug else ok_choice
        if style == 'slices':
            inside = f'{y}[0]+{y}[-1]+{chosen}'
            lines.append(f'result=({h}/3)*({inside})' if factor_form == 0 else f'result={h}*({inside})/3')
        elif style == 'loop':
            lines += chosen
            lines.append(f'result=({h}/3)*{total}' if factor_form == 0 else f'result={h}*{total}/3')
        else:
            lines += chosen
            lines.append(f'result=({h}/3)*np.sum({w}*{y})' if factor_form == 0 else f'result={h}*np.dot({w},{y})/3')
        return '\n'.join(lines)

    return render(issue == 'integration_sign', issue == 'simpson_weights'), render(False, False)


# ---------------------------------------------------------------- linear family

def make_linear(rng, subkind, consistent=True):
    """2x2 integer systems. Ill-conditioned ones have determinant +-1, so their solutions stay integers."""
    if subkind in ('well_small', 'well_scaled'):
        while True:
            M = rng.integers(-4, 5, (2, 2))
            det = int(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])
            if det != 0 and np.count_nonzero(M) >= 3:
                break
        A = M * (1 if subkind == 'well_small' else pick(rng, [10, 20]))
    elif subkind == 'ill':
        k = int(rng.integers(28, 46))
        A = np.array([[k, k + 1], [k - 1, k]])
        change = pick(rng, ['none', 'transpose', 'swap_rows', 'negate_row', 'negate_column'])
        if change == 'transpose':
            A = A.T.copy()
        elif change == 'swap_rows':
            A = A[::-1].copy()
        elif change == 'negate_row':
            A[0] *= -1
        elif change == 'negate_column':
            A[:, 1] *= -1
    else:
        u = rng.integers(1, 5, 2) * rng.choice([-1, 1], 2)
        v = rng.integers(1, 5, 2) * rng.choice([-1, 1], 2)
        A = np.outer(u, v) * pick(rng, [1, 2, 3])
    while True:
        x = rng.integers(-3, 4, 2)
        if np.any(x):
            break
    b = A @ x
    if subkind == 'singular' and not consistent:
        b = b + np.array([0, pick(rng, [-3, -2, -1, 1, 2, 3])])
    return {'kind': 'linear', 'A': A.astype(float).tolist(), 'b': b.astype(float).tolist(),
            'requested': 'unique_solution_or_null'}, [int(v) for v in x]


def linear_numbers(p):
    (a11, a12), (a21, a22) = [[int(v) for v in row] for row in p['A']]
    b1, b2 = [int(v) for v in p['b']]
    return a11, a12, a21, a22, b1, b2


def conditioning_ratio(p):
    a11, a12, a21, a22, _, _ = linear_numbers(p)
    return a11 ** 2 + a12 ** 2 + a21 ** 2 + a22 ** 2, abs(a11 * a22 - a12 * a21)


def linear_check_text(p, r):
    a11, a12, a21, a22, b1, b2 = linear_numbers(p)
    det = a11 * a22 - a12 * a21
    head = f'Check: det(A) = {times(a11, a22)} - {times(a12, a21)} = {det}'
    if det == 0:
        return head + ', so A is singular and A x = b has no unique solution.'
    n1, n2 = b1 * a22 - a12 * b2, a11 * b2 - b1 * a21
    squares, size = conditioning_ratio(p)
    side = 'above' if squares / size > 1000 else 'below'
    return (f'{head}; x1 = {n1}/{par(det)} = {fmt(n1 / det)}; x2 = {n2}/{par(det)} = {fmt(n2 / det)}. '
            f'Conditioning: sum of squared entries / |det| = {squares}/{size} = about {squares / size:.0f}, '
            f"{side} 1000, so {r['condition']}.")


def linear_derivation(p, issue, style, rng, particular):
    """Worked 2x2 derivation. `particular` is the vector used to build b (needed by the substitution layout)."""
    a11, a12, a21, a22, b1, b2 = linear_numbers(p)
    det = a11 * a22 - a12 * a21
    singular = det == 0
    diagnosis = 'The submitted derivation agrees with this check.'

    if singular and issue == 'ok':
        text = {'cramer': f'det(A) = {times(a11, a22)} - {times(a12, a21)} = 0. A is singular, so A x = b has no unique solution.',
                'adjugate': 'det(A) = 0, so A has no inverse and A x = b has no unique solution.',
                'substitution': f'det(A) = {times(a11, a22)} - {times(a12, a21)} = 0, so A is singular and A x = b has no unique solution.'}[style]
        return {'text': text, 'claim': None, 'diagnosis': diagnosis}

    if singular:  # singular_system: the derivation treats A as invertible
        diagnosis = 'The submitted derivation treats a singular A as invertible.'
        wrong_det = a11 * a22 + a12 * a21 if rng.random() < 0.5 else pick(rng, [-4, -2, -1, 1, 2, 4])
        if style == 'substitution':
            c1, c2 = particular
            if a11 * c1 + a12 * c2 != b1 or a21 * c1 + a22 * c2 != b2:
                return None
            ending = (f'Both rows reproduce b, and det(A) = {wrong_det} is nonzero, so this is the unique solution.'
                      if rng.random() < 0.5 else 'Both rows reproduce b, so this is the unique solution.')
            text = (f'Try x = {vec([c1, c2])}. Row 1: {times(a11, c1)} + {times(a12, c2)} = {b1}. '
                    f'Row 2: {times(a21, c1)} + {times(a22, c2)} = {b2}. ' + ending)
            return {'text': text, 'claim': [float(c1), float(c2)], 'diagnosis': diagnosis}
        n1, n2 = b1 * a22 - a12 * b2, a11 * b2 - b1 * a21
        if n1 == 0 and n2 == 0:
            return None
        x1, x2 = round(n1 / wrong_det, 4), round(n2 / wrong_det, 4)
        if style == 'cramer':
            sign = '+' if wrong_det == a11 * a22 + a12 * a21 else '-'
            text = (f'det(A) = {times(a11, a22)} {sign} {times(a12, a21)} = {wrong_det}. '
                    f'x1 = ({times(b1, a22)} - {times(a12, b2)})/{par(wrong_det)} = {n1}/{par(wrong_det)} = {fmt(x1)}. '
                    f'x2 = ({times(a11, b2)} - {times(b1, a21)})/{par(wrong_det)} = {n2}/{par(wrong_det)} = {fmt(x2)}. '
                    f'So x = {vec([x1, x2])}.')
        else:
            text = (f'det(A) = {wrong_det}. adj(A) = [[{a22}, {-a12}], [{-a21}, {a11}]]. With b = {vec([b1, b2])}, '
                    f'adj(A) b = {vec([n1, n2])}. x = adj(A) b / det(A) = {vec([x1, x2])}.')
        return {'text': text, 'claim': [float(x1), float(x2)], 'diagnosis': diagnosis}

    scale = pick(rng, [2, -1, 3]) if issue == 'rhs_scale' else 1
    u1, u2 = scale * b1, scale * b2  # the right-hand side the derivation actually works with
    n1, n2 = u1 * a22 - a12 * u2, a11 * u2 - u1 * a21
    x1, x2 = n1 // det, n2 // det
    if n1 % det or n2 % det:
        return None
    true = [(b1 * a22 - a12 * b2) // det, (a11 * b2 - b1 * a21) // det]
    if issue == 'rhs_scale':
        diagnosis = (f'The submitted derivation solves for right-hand side {vec([u1, u2])}, '
                     f'but the stated b is {vec([b1, b2])}.')
    slip = pick(rng, [-2, -1, 1, 2]) if issue == 'arithmetic' else 0
    component = int(rng.integers(2))
    if issue == 'arithmetic':
        diagnosis = f'The submitted set-up is right but a number is miscalculated; the solution should be {vec(true)}.'

    if style == 'substitution':
        c = [x1, x2]
        c[component] += slip
        rows = [u1, u2]  # with a slip the stated row values are still the right-hand side, which is the false step
        ending = (f'Both rows reproduce b, and det(A) = {det} is nonzero, so this is the unique solution.'
                  if rng.random() < 0.5 else 'Both rows reproduce b, so this is the unique solution.')
        text = (f'Try x = {vec(c)}. Row 1: {times(a11, c[0])} + {times(a12, c[1])} = {rows[0]}. '
                f'Row 2: {times(a21, c[0])} + {times(a22, c[1])} = {rows[1]}. ' + ending)
        claim = c
    else:
        shown_n, shown_x = [n1, n2], [x1, x2]
        if slip:
            shown_x[component] += slip
            if rng.random() < 0.5:  # the numerator carries the slip, the quotient follows it
                shown_n[component] += slip * det
        if style == 'cramer':
            text = (f'det(A) = {times(a11, a22)} - {times(a12, a21)} = {det}. '
                    f'x1 = ({times(u1, a22)} - {times(a12, u2)})/{par(det)} = {shown_n[0]}/{par(det)} = {shown_x[0]}. '
                    f'x2 = ({times(a11, u2)} - {times(u1, a21)})/{par(det)} = {shown_n[1]}/{par(det)} = {shown_x[1]}. '
                    f'So x = {vec(shown_x)}.')
        else:
            text = (f'det(A) = {det}. adj(A) = [[{a22}, {-a12}], [{-a21}, {a11}]]. With b = {vec([u1, u2])}, '
                    f'adj(A) b = {vec(shown_n)}. x = adj(A) b / det(A) = {vec(shown_x)}.')
        claim = shown_x
    if (list(claim) == true) != (issue == 'ok'):
        return None
    return {'text': text, 'claim': [float(v) for v in claim], 'diagnosis': diagnosis}


def linear_code(issue, style, singular, rng):
    """Returns (submitted, corrected). With A'=sA*A, b'=sb*b, result=sr*solve(A',b') the answer is right iff sr*sb==sA."""
    A = pick(rng, ['A', 'M'])
    b = pick(rng, ['b', 'rhs'])
    ok_scales = (1, 1, 1) if rng.random() < 0.4 else pick(rng, [(2, 2, 1), (1, 2, 0.5), (2, 1, 2)])
    bad_scales = pick(rng, [(1, 2, 1), (1, 2, 1), (1, -1, 1), (1, 3, 1), (1, 1, 2), (2, 1, 1)])
    inline_b = rng.random() < 0.5
    doubled_by_adding = rng.random() < 0.5  # b+b in place of 2*b, for faulty and harmless programs alike

    if style == 'rank_guard':
        rank, size = f'np.linalg.matrix_rank({A})', f'len({b})'
        ok_guards = [(f'{rank}<{size}', 'none_first'), (f'{rank}=={size}', 'solve_first'), (f'{rank}!={size}', 'none_first')]
        never, always, inverted = (f'{rank}>{size}', 'none_first'), (f'{rank}<={size}', 'none_first'), (f'{rank}=={size}', 'none_first')
    elif style == 'det_guard':
        ok_guards = [('abs(d)<1e-9', 'none_first'), ('abs(d)>1e-9', 'solve_first')]
        never, always, inverted = ('abs(d)<-1e-9', 'none_first'), ('abs(d)<1e9', 'none_first'), ('abs(d)>1e-9', 'none_first')
    else:  # default_none, held out
        ok_guards = [('r==n', 'solve_only'), ('r>=n', 'solve_only'), ('not r<n', 'solve_only')]
        never, always, inverted = (pick(rng, ['r>0', 'r<=n']), 'solve_only'), ('r>n', 'solve_only'), ('r<n', 'solve_only')
    ok_guard = pick(rng, ok_guards)
    # A guard that never fires only shows on a singular A; one that always fires only shows on a regular A.
    bad_guard = pick(rng, [never, inverted, None]) if singular else pick(rng, [always, inverted])

    def render(scales, guard):
        scale_a, scale_b, scale_r = scales
        head = {1: '', 2: '2*', 3: '3*', -1: '-'}
        loaded = "np.array(p['b'])"
        if scale_b == 2 and doubled_by_adding:
            b_line, argument = (f'{b}={loaded}', f'{b}+{b}') if inline_b else (f'{b}={loaded}+{loaded}', b)
        else:
            b_line, argument = (f'{b}={loaded}', f'{head[scale_b]}{b}') if inline_b else (f'{b}={head[scale_b]}{loaded}', b)
        lines = ['import numpy as np', f"{A}={head[scale_a]}np.array(p['A'])", b_line]
        if style == 'default_none':
            solve = f'np.dot(np.linalg.solve({A},np.eye(n)),{argument})'
        else:
            solve = f'np.linalg.solve({A},{argument})'
        solve = {1: solve, 2: f'2*{solve}', 0.5: f'{solve}/2'}[scale_r]
        if style == 'det_guard':
            lines.append(f'd=np.linalg.det({A})')
        if style == 'default_none':
            lines += [f'n=len({b})', f'r=np.linalg.matrix_rank({A})']
        if guard is None:
            lines.append(f'result={solve}')
        elif guard[1] == 'none_first':
            lines += [f'if {guard[0]}:', '    result=None', 'else:', f'    result={solve}']
        elif guard[1] == 'solve_first':
            lines += [f'if {guard[0]}:', f'    result={solve}', 'else:', '    result=None']
        else:
            lines += ['result=None', f'if {guard[0]}:', f'    result={solve}']
        return '\n'.join(lines)

    submitted = render(bad_scales if issue == 'rhs_scale' else ok_scales,
                       bad_guard if issue == 'singular_system' else ok_guard)
    corrected = render((1, 1, 1) if issue == 'rhs_scale' else ok_scales, ok_guard)
    return submitted, corrected


# ---------------------------------------------------------------- behaviour checks used while generating

def run_on(code, problem, family):
    try:
        value = NumericInterpreter().run(code, problem)['result']
    except Exception:
        return False
    return numeric_check(value, {'family': family, 'public_problem': problem, 'private_reference': reference(problem)})[0]


def correct_everywhere(code, problem, family):
    probes = [problem] + hidden_rebindings({'public_problem': problem})
    return all(run_on(code, q, family) for q in probes)


# ---------------------------------------------------------------- groups and rows

def make_group(family, index, split, pool, heldout, seed, decks, seen, audit):
    block = 'test_template' if heldout else pool if split == 'train' else split
    rng = np.random.default_rng([seed, 0 if family == 'integral' else 1, index])
    styles = (HELDOUT_STYLES if heldout else SEEN_STYLES)[family]

    def deck(kind, key, cards):
        name = (family, block, kind, key)
        if name not in decks:
            decks[name] = Deck(cards, np.random.default_rng([seed, 7, len(decks)]))
        return decks[name]

    subkind = 'integral' if family == 'integral' else deck('subkind', '', LINEAR_SUBKINDS).draw()
    singular = subkind == 'singular'
    derivation_issue = deck('derivation', subkind, DERIVATION_DECKS[subkind]).draw()
    code_key = 'integral' if family == 'integral' else 'singular' if singular else 'regular'
    code_issue = deck('code', code_key, CODE_DECKS[code_key]).draw()

    for _ in range(400):
        derivation_style, code_style = pick(rng, styles['derivation']), pick(rng, styles['code'])
        if family == 'integral':
            p, particular = make_integral(rng), None
        else:
            # Cramer and adjugate need a nonzero numerator to show a wrong "solution" for a singular A.
            consistent = not (singular and derivation_issue == 'singular_system' and derivation_style != 'substitution')
            p, particular = make_linear(rng, subkind, consistent)
        key = mathematical_hash(p)
        if key in seen:
            audit['duplicate_proposals_rejected'] += 1
            continue
        r = reference(p)
        if family == 'integral':
            if abs(r['value']) < 1 or abs(r['value']) > 400 or sum(c != 0 for c in p['coefficients']) < 2:
                continue
            derivation = integral_derivation(p, derivation_issue, derivation_style, rng)
            submitted, corrected = integral_code(code_issue, code_style, rng)
            check_text = integral_check_text(p)
        else:
            squares, size = conditioning_ratio(p)
            expected = 'singular' if singular else 'ill_conditioned' if subkind == 'ill' else 'well_conditioned'
            # The hand rule (squares/|det| against 1000) must agree with cond2, with a wide margin.
            if r['condition'] != expected or (not singular and ((squares / size > 1000) != (subkind == 'ill')
                                                                or 300 < r['condition_number'] < 3000)):
                continue
            if not singular:
                # b was built from an integer vector, so the exact solution replaces the floating-point one.
                r['value'] = [float(v) for v in particular]
            probes = [reference(q) for q in hidden_rebindings({'public_problem': p})]
            if any(q['condition_number'] is not None and q['condition_number'] > 1e6 for q in probes):
                continue
            derivation = linear_derivation(p, derivation_issue, derivation_style, rng, particular)
            submitted, corrected = linear_code(code_issue, code_style, singular, rng)
            check_text = linear_check_text(p, r)
        if derivation is None:
            continue
        # The label of a program must match what it does: correct ones pass everywhere, faulty ones fail on the stated problem.
        if not correct_everywhere(corrected, p, family) or (code_issue == 'ok') != run_on(submitted, p, family):
            audit['behaviour_resamples'] += 1
            continue
        break
    else:
        raise RuntimeError(f'could not build {family} group {index}: {derivation_issue}/{code_issue}')
    seen.add(key)

    p['submitted_derivation'] = derivation['text']
    p['submitted_code'] = submitted
    group = f'{family}-{index:05d}'
    shared = problem_text(p) + '\nSUBMITTED_DERIVATION:\n' + derivation['text'] + '\nSUBMITTED_PYTHON:\n' + submitted + '\n'
    design = {'subkind': subkind, 'derivation_style': derivation_style, 'code_style': code_style,
              'derivation_issue': derivation_issue, 'code_issue': code_issue,
              'submitted_claim': derivation['claim'], 'template': 'heldout' if heldout else 'seen'}
    rows = []
    for stage, issue in [('analysis', derivation_issue), ('code', code_issue)]:
        verdict = 'accept' if issue == 'ok' else 'revise'
        if stage == 'analysis':
            final = {'value': r['value'], 'condition': r['condition'], 'verdict': verdict, 'issue': issue}
            target = check_text + ' ' + derivation['diagnosis'] + '\nFINAL=' + json.dumps(final, separators=(',', ':'))
            prompt = shared + ANALYSIS_INSTRUCTION
        else:
            target = corrected + f'\nverdict={verdict!r}\nissue={issue!r}'
            prompt = shared + CODE_INSTRUCTION
        row = {'problem_id': f'{group}-{stage}', 'group_id': group, 'family': family, 'stage': stage, 'domain': stage,
               'split': split, 'pool': pool, 'prompt': prompt, 'public_problem': p, 'semantic_hash': key, 'target': target,
               'private_reference': {**r, 'expected_verdict': verdict, 'expected_issue': issue}, 'design': design,
               'task_version': TASK_VERSION, 'provenance': 'programmatically generated synthetic scientific-computing task'}
        if not check_completion(row, target)['auto_pass']:
            raise RuntimeError('reference target fails its own verifier: ' + row['problem_id'])
        rows.append(row)
    return rows


def make_anchor(index, count, seed):
    rng = np.random.default_rng(seed + 900001 + index)
    tag = f'unit_{int(rng.integers(10, 999))}'
    number = int(rng.integers(0, 100))
    expected = {'label': tag, 'count': number}
    split = split_for(index, count)
    return {'problem_id': f'instruction-{index:05d}-anchor', 'group_id': f'instruction-{index:05d}', 'family': 'instruction',
            'stage': 'anchor', 'domain': 'anchor', 'split': split, 'pool': 'anchor' if split == 'train' else 'evaluation',
            'prompt': (f'Return exactly one JSON object with label equal to the literal string {tag!r} and count equal to '
                       f'the integer {number}. Use only keys label and count. No commentary.'),
            'public_problem': {'label_literal': tag, 'count_literal': number},
            'target': json.dumps(expected, separators=(',', ':')), 'private_reference': {'expected_json': expected},
            'task_version': TASK_VERSION,
            'provenance': 'programmatically generated instruction-format task from a separate pool'}


def count_by(rows, key):
    table = {}
    for row in rows:
        name = key(row)
        table[name] = table.get(name, 0) + 1
    return dict(sorted(table.items()))


def generate(out, groups_per_family=24, anchor_groups=24, seed=17, template_holdout_groups=None):
    holdout = groups_per_family // 6 if template_holdout_groups is None else template_holdout_groups
    rows, seen, decks = [], set(), {}
    audit = {'duplicate_proposals_rejected': 0, 'behaviour_resamples': 0}
    for family in ['integral', 'linear']:
        for index in range(groups_per_family + holdout):
            heldout = index >= groups_per_family
            split = 'test_template' if heldout else split_for(index, groups_per_family)
            pool = ('teacher_sft' if index % 2 == 0 else 'distill') if split == 'train' else 'evaluation'
            rows.extend(make_group(family, index, split, pool, heldout, seed, decks, seen, audit))
    for index in range(anchor_groups):
        rows.append(make_anchor(index, anchor_groups, seed))

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / 'all.jsonl', rows)
    splits = ['train', 'val', 'test', 'test_template']
    for split in splits:
        write_jsonl(out / f'{split}.jsonl', [r for r in rows if r['split'] == split])
    primary = [r for r in rows if r['stage'] != 'anchor']
    groups = {r['group_id']: r['design'] for r in primary}
    manifest = {
        'task_version': TASK_VERSION, 'seed': seed, 'groups_per_primary_family': groups_per_family,
        'template_holdout_groups_per_family': holdout, 'anchor_groups': anchor_groups, 'rows': len(rows),
        'split_counts': {s: sum(r['split'] == s for r in rows) for s in splits},
        'pool_counts': {p: sum(r['pool'] == p for r in rows) for p in ['teacher_sft', 'distill', 'anchor', 'evaluation']},
        'split_rule': 'whole problem groups; first 2/3 train, next 1/6 validation, last 1/6 test; extra groups form test_template',
        'pool_rule': 'training primary groups alternate teacher_sft/distill by group parity; independent anchor instruction pool',
        'group_isolation': True, 'data_kind': 'synthetic',
        'styles': {'seen': SEEN_STYLES, 'heldout_in_test_template_only': HELDOUT_STYLES},
        'claim_scope': ('Two algorithm families: quadrature and linear systems. Train, validation and test use distinct problems '
                        'with shared layouts. The template test uses layouts held out from training.'),
        'semantic_duplicate_audit': audit, 'unique_primary_math_problems': len(seen),
        'issue_by_split_and_stage': {s: {stage: count_by([r for r in primary if r['split'] == s and r['stage'] == stage],
                                                         lambda r: r['private_reference']['expected_issue'])
                                         for stage in ['analysis', 'code']} for s in splits},
        'issue_by_training_pool': {pool: {stage: count_by([r for r in primary if r['pool'] == pool and r['stage'] == stage],
                                                          lambda r: r['private_reference']['expected_issue'])
                                          for stage in ['analysis', 'code']} for pool in ['teacher_sft', 'distill']},
        'derivation_by_code_issue_groups': count_by(groups.values(), lambda d: d['derivation_issue'] + ' | ' + d['code_issue']),
        'fingerprint': object_sha(rows)}
    write_json(out / 'manifest.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data')
    parser.add_argument('--groups-per-family', type=int, default=24)
    parser.add_argument('--anchor-groups', type=int, default=24)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--template-holdout-groups', type=int, default=None,
                        help='extra groups per family that use held-out layouts; default is one sixth of --groups-per-family')
    args = parser.parse_args()
    generate(args.out, args.groups_per_family, args.anchor_groups, args.seed, args.template_holdout_groups)
