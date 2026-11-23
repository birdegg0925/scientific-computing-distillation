import copy

import pytest

from sci_mopd.common import DERIVATION_ISSUES, PROGRAM_ISSUES, model_messages, write_jsonl
from sci_mopd.data import HELDOUT_STYLES, SEEN_STYLES, correct_everywhere, generate, mathematical_hash, run_on
from sci_mopd.shortcuts import prompt_parts, score
from sci_mopd.verify import NumericInterpreter, RejectedCode, check_completion, final_record, numeric_check


@pytest.fixture(scope='module')
def rows(tmp_path_factory):
    return generate(tmp_path_factory.mktemp('data'), 24, 24, 17)


@pytest.fixture(scope='module')
def larger(tmp_path_factory):
    """Big enough for every pool to hold every issue; a different seed from the shipped data."""
    return generate(tmp_path_factory.mktemp('larger'), 96, 24, 23)


def primary(rows, stage=None):
    return [r for r in rows if r['stage'] != 'anchor' and (stage is None or r['stage'] == stage)]


# ---------------------------------------------------------------- splits and pools

def test_group_and_semantic_isolation(rows):
    groups, semantics = {}, {}
    for r in rows:
        groups.setdefault(r['group_id'], set()).add((r['split'], r['pool']))
        if r['family'] != 'instruction':
            semantics.setdefault(r['semantic_hash'], set()).add((r['group_id'], r['split'], r['pool']))
    assert all(len(x) == 1 for x in groups.values())
    assert all(len(x) == 1 for x in semantics.values())
    assert {r['split'] for r in rows} == {'train', 'val', 'test', 'test_template'}


def test_each_training_pool_holds_every_issue(larger):
    for pool in ['teacher_sft', 'distill']:
        for stage, expected in [('analysis', DERIVATION_ISSUES), ('code', PROGRAM_ISSUES)]:
            found = {r['private_reference']['expected_issue'] for r in primary(larger, stage) if r['pool'] == pool}
            assert found == set(expected)


def test_heldout_layouts_never_reach_training(larger):
    for r in primary(larger):
        allowed = (HELDOUT_STYLES if r['split'] == 'test_template' else SEEN_STYLES)[r['family']]
        assert r['design']['derivation_style'] in allowed['derivation']
        assert r['design']['code_style'] in allowed['code']
        assert (r['design']['template'] == 'heldout') == (r['split'] == 'test_template')


# ---------------------------------------------------------------- what the labels mean

def test_derivation_and_program_errors_are_independent(larger):
    combos = {(r['design']['derivation_issue'] == 'ok', r['design']['code_issue'] == 'ok') for r in primary(larger)}
    assert combos == {(True, True), (True, False), (False, True), (False, False)}
    for r in primary(larger):
        key = 'derivation_issue' if r['stage'] == 'analysis' else 'code_issue'
        assert r['private_reference']['expected_issue'] == r['design'][key]


def test_program_label_matches_program_behaviour(larger):
    for r in primary(larger, 'code'):
        submitted = r['public_problem']['submitted_code']
        clean = r['private_reference']['expected_issue'] == 'ok'
        assert run_on(submitted, r['public_problem'], r['family']) == clean
        if clean:
            assert correct_everywhere(submitted, r['public_problem'], r['family'])


def test_derivation_label_matches_its_claim(larger):
    for r in primary(larger, 'analysis'):
        claim = r['design']['submitted_claim']
        assert numeric_check(claim, r)[0] == (r['private_reference']['expected_issue'] == 'ok')


def test_all_oracle_targets_and_hidden_rebindings(rows):
    assert not [r['problem_id'] for r in rows if not check_completion(r, r['target'])['auto_pass']]


# ---------------------------------------------------------------- nothing private in the prompt

def test_gold_never_appended_to_prompt(rows):
    r = copy.deepcopy(rows[0])
    before = model_messages(r)
    r['target'] = 'PRIVATE_TARGET_CANARY'
    r['private_reference'] = {'canary': 'PRIVATE_REFERENCE'}
    r['design'] = {'canary': 'PRIVATE_DESIGN'}
    assert model_messages(r) == before


def test_prompt_states_no_diagnosis(larger):
    instructions = {'analysis': set(), 'code': set()}
    for r in primary(larger):
        problem, derivation, code = prompt_parts(r['prompt'])
        instructions[r['stage']].add(r['prompt'].split(code + '\n', 1)[1])
        shown = derivation + code
        for word in set(DERIVATION_ISSUES + PROGRAM_ISSUES) - {'ok'}:
            assert word not in shown
        for phrase in ['Check:', 'submitted derivation', 'miscalculated', 'reverse order', 'mishandles', 'in place of']:
            assert phrase not in shown
    # The text after the candidates is the same for every row of a stage, so no label can hide there.
    assert all(len(texts) == 1 for texts in instructions.values())


def test_shortcut_baselines_stay_near_the_accept_everything_floor(larger):
    for split in ['val', 'test', 'test_template']:
        data = [r for r in larger if r['split'] == split]
        assert score(data, 'accept_all')['macro'] <= 0.40
        assert score(data, 'keyword')['macro'] <= 0.40


# ---------------------------------------------------------------- verifier behaviour

def test_constant_answer_cannot_pass_program_generalization(rows):
    r = next(r for r in rows if r['stage'] == 'code' and r['family'] == 'integral')
    v = r['private_reference']
    code = f"result={v['value']!r}\nverdict={v['expected_verdict']!r}\nissue={v['expected_issue']!r}"
    checked = check_completion(r, code)
    assert checked['numeric_ok'] and not checked['auto_pass']
    assert not all(checked['hidden_probe_passes'])


def test_singular_constant_null_does_not_pass_hidden_full_rank_probe(rows):
    r = next(r for r in rows if r['stage'] == 'code' and r['private_reference'].get('condition') == 'singular')
    v = r['private_reference']
    code = f"result=None\nverdict={v['expected_verdict']!r}\nissue={v['expected_issue']!r}"
    assert not check_completion(r, code)['auto_pass']


@pytest.mark.parametrize('source', [
    "result=[0]*1000000000",
    "result='X'*1000000000000",
    "import numpy as np\nresult=np.eye(2,1000000000)",
    "import numpy as np\nresult=np.ones((4096,1))+np.ones((1,4096))",
    "import numpy as np\nresult=np.dot(np.ones((4096,1)),np.ones((1,4096)))",
    "import numpy as np\nresult=np.zeros((1000000000000,1000000000000))",
    "import numpy as np\nresult=np.ones((-1,2))"])
def test_preallocation_limits(source):
    with pytest.raises((RejectedCode, ValueError)):
        NumericInterpreter().run(source)


@pytest.mark.parametrize('source', [
    "import os\nresult=1",
    "result=open('anything')",
    "while True:\n result=1",
    "result=().__class__",
    "p={'a':1}\nresult=1",
    "p['a']=0\nresult=1",
    "def solve():\n return 1\nresult=solve()"])
def test_forbidden_grammar_and_input_mutation(source):
    with pytest.raises((RejectedCode, ValueError)):
        NumericInterpreter().run(source, {'a': 2})


def test_final_requires_explicit_null_and_labels():
    with pytest.raises(ValueError):
        final_record('Reason. FINAL={"condition":"singular","verdict":"accept","issue":"ok"}')
    assert final_record('Reason. FINAL={"value":null,"condition":"singular","verdict":"accept","issue":"ok"}')[0]['value'] is None


def test_bounded_loops_work():
    assert NumericInterpreter().run('result=0\nfor i in range(4):\n result+=i')['result'] == 6
    with pytest.raises(RejectedCode):
        NumericInterpreter().run('result=0\nfor i in range(1000000):\n result+=i')


def test_boolean_is_not_a_numeric_answer_or_anchor_integer(rows):
    r = copy.deepcopy(next(r for r in rows if r['family'] == 'integral'))
    r['private_reference']['value'] = 1.
    assert numeric_check(True, r)[0] is False
    anchor = copy.deepcopy(next(r for r in rows if r['stage'] == 'anchor'))
    anchor['private_reference']['expected_json'] = {'label': 'x', 'count': 1}
    assert not check_completion(anchor, '{"label":"x","count":true}')['auto_pass']


def test_evaluation_rejects_duplicate_or_extra_ids(rows):
    from sci_mopd.evaluate import evaluate_rows
    r = rows[0]
    p = {'problem_id': r['problem_id'], 'completion': r['target'], 'engine': 'oracle'}
    with pytest.raises(ValueError):
        evaluate_rows([r, r], [p])
    with pytest.raises(ValueError):
        evaluate_rows([r], [p, {'problem_id': 'extra', 'completion': 'x'}])


def test_boolean_short_circuit_returns_operands():
    assert NumericInterpreter().run('result=0 or 2')['result'] == 2
    assert NumericInterpreter().run('result=False and (1/0)')['result'] is False
    assert NumericInterpreter().run('result=3 or (1/0)')['result'] == 3


def test_readonly_dictionary_cannot_be_aliased():
    with pytest.raises(RejectedCode):
        NumericInterpreter().run("q=p\nq['a']=0\nresult=1", {'a': 2})
    env = NumericInterpreter().run("a=p['A']\na[0]=9\nresult=1", {'A': [1, 2]})
    assert env['p']['A'] == [1, 2]


def test_hf_stage_peak_mixture_is_rejected(tmp_path, rows):
    from sci_mopd.evaluate import main
    data = [rows[0], rows[1]]
    write_jsonl(tmp_path / 'data.jsonl', data)
    write_jsonl(tmp_path / 'pred.jsonl', [{'problem_id': r['problem_id'], 'completion': r['target'], 'engine': 'hf',
                                           'checkpoint_id': str(i)} for i, r in enumerate(data)])
    with pytest.raises(ValueError):
        main(['--data', str(tmp_path / 'data.jsonl'), '--predictions', str(tmp_path / 'pred.jsonl'), '--out', str(tmp_path / 'eval')])


def test_math_semantic_hash_ignores_discretization_and_candidates():
    p = {'kind': 'integral', 'integrand': 'cubic', 'coefficients': [1., 4., 0., 0.], 'a': 0., 'b': 1., 'n': 8}
    q = copy.deepcopy(p)
    q['n'] = 24
    q['submitted_code'] = 'anything'
    q['submitted_derivation'] = 'anything'
    assert mathematical_hash(p) == mathematical_hash(q)
    q['b'] = 2.
    assert mathematical_hash(p) != mathematical_hash(q)
