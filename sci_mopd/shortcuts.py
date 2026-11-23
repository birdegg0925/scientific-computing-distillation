"""Model-free baselines for detecting shortcuts in the scientific-computing tasks.

Version 1 of the task set could be scored 0.914 by a rule that only matched
substrings in the prompt. These two baselines stay in the project so that the
same check runs on every regenerated data set:

    accept_all  says every candidate is fine and copies the submitted answer
    keyword     the version-1 style rule: fixed substrings decide the label,
                no arithmetic, no comparison against the stated problem

Neither baseline is a model. Their scores are a floor that a trained model has
to clear, and `python -m sci_mopd.shortcuts` fails when the keyword rule gets
above the allowed macro score.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

from .common import read_jsonl, write_json
from .evaluate import evaluate_rows, macro

CANONICAL_LINEAR = ("import numpy as np\nA=np.array(p['A'])\nb=np.array(p['b'])\n"
                    "if np.linalg.matrix_rank(A)<len(b):\n    result=None\nelse:\n    result=np.linalg.solve(A,b)")


def prompt_parts(prompt):
    problem, rest = prompt.split('\nSUBMITTED_DERIVATION:\n', 1)
    derivation, rest = rest.split('\nSUBMITTED_PYTHON:\n', 1)
    code = rest.split('\nThe derivation and the program were written separately', 1)[0]
    return problem, derivation, code


def submitted_claim(derivation):
    """The last number or vector the derivation states, read as text."""
    if 'no unique solution' in derivation:
        return None
    vectors = re.findall(r'\[[^\[\]]*\]', derivation)
    if vectors:
        return json.loads(vectors[-1])
    return float(re.findall(r'-?\d+(?:\.\d+)?', derivation)[-1])


def final_line(value, condition, verdict, issue):
    return 'Shortcut baseline.\nFINAL=' + json.dumps({'value': value, 'condition': condition, 'verdict': verdict, 'issue': issue})


def labelled(code, issue):
    return code + f"\nverdict={'accept' if issue == 'ok' else 'revise'!r}\nissue={issue!r}"


def anchor(row):
    label = re.search(r"literal string '([^']*)'", row['prompt']).group(1)
    count = int(re.search(r'the integer (-?\d+)', row['prompt']).group(1))
    return json.dumps({'label': label, 'count': count})


def accept_all(row):
    if row['stage'] == 'anchor':
        return anchor(row)
    problem, derivation, code = prompt_parts(row['prompt'])
    if row['stage'] == 'code':
        return labelled(code, 'ok')
    claim = submitted_claim(derivation)
    condition = 'not_applicable' if problem.startswith('Integrate') else 'singular' if claim is None else 'well_conditioned'
    return final_line(claim, condition, 'accept', 'ok')


def keyword(row):
    if row['stage'] == 'anchor':
        return anchor(row)
    problem, derivation, code = prompt_parts(row['prompt'])
    integral = problem.startswith('Integrate')
    if row['stage'] == 'code':
        if integral:
            if re.search(r'=\(a-b\)/n|=-\(b-a\)/n', code):
                fixed = re.sub(r'=\(a-b\)/n|=-\(b-a\)/n', '=(b-a)/n', code)
                return labelled(fixed, 'integration_sign')
            if '3*' in code:
                return labelled(code.replace('3*', '4*'), 'simpson_weights')
            return labelled(code, 'ok')
        if 'if ' not in code:
            return labelled(CANONICAL_LINEAR, 'singular_system')
        if re.search(r'2\*|3\*|/2|=-np|,-', code):
            return labelled(CANONICAL_LINEAR, 'rhs_scale')
        return labelled(code, 'ok')
    # The derivation shows numbers only, so no substring names its error; the size of A is the one cue left.
    claim = submitted_claim(derivation)
    if integral:
        condition = 'not_applicable'
    elif claim is None:
        condition = 'singular'
    else:
        matrix = problem.split('For A=', 1)[1].split(' and b=', 1)[0]
        largest = max(abs(float(v)) for v in re.findall(r'-?\d+(?:\.\d+)?', matrix))
        condition = 'ill_conditioned' if largest >= 27 else 'well_conditioned'
    return final_line(claim, condition, 'accept', 'ok')


BASELINES = {'accept_all': accept_all, 'keyword': keyword}


def score(data, name):
    """Run one baseline through the project verifier; anchor rows are left out of the macro as usual."""
    predictions = [{'problem_id': r['problem_id'], 'completion': BASELINES[name](r), 'engine': name} for r in data]
    rows = evaluate_rows(data, predictions)
    expected = {r['problem_id']: r['private_reference'].get('expected_issue') for r in data}
    by_issue = {}
    for r in rows:
        if r['stage'] == 'anchor':
            continue
        cell = by_issue.setdefault(r['stage'] + ' | ' + expected[r['problem_id']], [0, 0])
        cell[0] += bool(r['auto_pass'])
        cell[1] += 1
    stages = {s: [sum(r['auto_pass'] for r in rows if r['stage'] == s), sum(r['stage'] == s for r in rows)]
              for s in ['analysis', 'code']}
    return {'baseline': name, 'macro': macro(rows), 'by_stage': {s: f'{p}/{n}' for s, (p, n) in stages.items()},
            'by_stage_and_issue': {k: f'{p}/{n}' for k, (p, n) in sorted(by_issue.items())}}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Score the model-free shortcut baselines on one or more data files.')
    parser.add_argument('--data', nargs='+', required=True)
    parser.add_argument('--max-keyword-macro', type=float, default=0.40,
                        help='fail when the keyword rule scores above this two-stage macro on any file')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    report, failed = {}, False
    for path in args.data:
        data = read_jsonl(path)
        report[path] = [score(data, name) for name in BASELINES]
        for item in report[path]:
            print(f"{path}  {item['baseline']:<10} macro={item['macro']:.3f}  {item['by_stage']}")
            if item['baseline'] == 'keyword':
                for cell, value in item['by_stage_and_issue'].items():
                    print(f'    {cell:<34} {value}')
                failed = failed or item['macro'] > args.max_keyword_macro
    if args.out:
        write_json(args.out, {'max_keyword_macro': args.max_keyword_macro, 'gate_passed': not failed, 'files': report})
    print('shortcut gate:', 'FAILED' if failed else 'passed', f'(keyword macro limit {args.max_keyword_macro})')
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
