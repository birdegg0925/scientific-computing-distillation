"""Evidence-boundary checks for the companion workflow, with synthetic references."""
from types import SimpleNamespace as Args
import json
import pytest
import workflow_tools as w


@pytest.fixture
def corpus(tmp_path):
    from sci_mopd.data import generate
    generate(tmp_path / "data", 24, 24, 17)
    return tmp_path


def oracle_predictions(data, path, broken=False):
    rows = [{"problem_id": r["problem_id"], "completion": "invalid" if broken else r["target"], "engine": "oracle"} for r in data]
    w.write_rows(path, rows)
    return str(path)


def test_missing_prediction_cannot_disappear_from_denominator(corpus):
    data = w.read_rows(corpus / "data/val.jsonl")
    path = corpus / "short.jsonl"
    oracle_predictions(data[:-1], path)
    with pytest.raises(ValueError, match="coverage"):
        w.predictions_for(data, path)


def test_joint_group_success_is_not_macro_stage_accuracy():
    rows = [
        {"group_id": "g1", "stage": "analysis", "auto_pass": False},
        {"group_id": "g1", "stage": "code", "auto_pass": True},
        {"group_id": "g2", "stage": "analysis", "auto_pass": True},
        {"group_id": "g2", "stage": "code", "auto_pass": True},
    ]
    result = w.rates(rows)
    assert result["primary_macro"] == .75
    assert result["paired_primary"]["rate"] == .5


def test_private_labels_not_accepted_as_predictions(corpus):
    data = w.read_rows(corpus / "data/val.jsonl")
    path = corpus / "leaky.jsonl"
    w.write_rows(path, [{"problem_id": r["problem_id"], "completion": r["target"], "target": r["target"]} for r in data])
    with pytest.raises(ValueError, match="private"):
        w.predictions_for(data, path)


def test_oracle_cannot_be_reported_as_teacher_targets(corpus):
    data = [r for r in w.read_rows(corpus / "data/train.jsonl") if r["pool"] == "distill"]
    source = corpus / "train_view.jsonl"; w.write_rows(source, data)
    pred = oracle_predictions(data, corpus / "oracle.jsonl")
    args = Args(data=str(source), predictions=pred, out=str(corpus / "targets.jsonl"), report=str(corpus / "filter.json"), accepted_only=True, allow_reference_demo=False)
    with pytest.raises(ValueError, match="only recorded hf"):
        w.teacher_targets(args)
    args.allow_reference_demo = True
    w.teacher_targets(args)
    report = json.loads((corpus / "filter.json").read_text())
    assert report["reference_demo_only"] is True
    assert report["retained"] == len(data)


def test_admission_retains_manual_review_gate(corpus):
    path = corpus / "data/val.jsonl"; data = w.read_rows(path)
    before = oracle_predictions(data, corpus / "bad.jsonl", broken=True)
    after = oracle_predictions(data, corpus / "good.jsonl")
    out = corpus / "admission.json"
    w.admit(Args(data=str(path), baseline=before, teacher=after, stage="analysis", bootstrap=100, seed=29, require_positive_ci=True, out=str(out), allow_reference_demo=True))
    result = json.loads(out.read_text())
    assert result["automatic_gain_gate"] is True
    assert result["manual_review_required"] is True
    assert result["manual_review_status"] == "not_performed"
    assert result["status"] == "reference_demo_only"


def test_test_set_cannot_select_checkpoints(corpus):
    path = corpus / "data/test.jsonl"
    with pytest.raises(ValueError, match="validation"):
        w.choose(Args(data=str(path), baseline="unused", predictions=[], out="unused"))


def test_checkpoint_choice_requires_no_anchor_regression(corpus):
    path = corpus / "data/val.jsonl"; data = w.read_rows(path)
    baseline = oracle_predictions(data, corpus / "baseline.jsonl")
    good = oracle_predictions(data, corpus / "good.jsonl")
    bad = corpus / "bad_anchor.jsonl"
    w.write_rows(bad, [{"problem_id": r["problem_id"], "completion": "invalid" if r["stage"] == "anchor" else r["target"], "engine": "oracle"} for r in data])
    out = corpus / "choice.json"
    w.choose(Args(data=str(path), baseline=baseline, predictions=[str(bad), good], out=str(out), allow_reference_demo=True))
    result = json.loads(out.read_text())
    assert result["selected"]["predictions"] == good
    assert result["candidates"][0]["eligible"] is False
    assert result["test_used"] is False
    assert result["release_ready"] is False


def test_real_selection_rejects_oracle_engine(corpus):
    path = corpus / "data/val.jsonl"; data = w.read_rows(path)
    pred = oracle_predictions(data, corpus / "reference.jsonl")
    with pytest.raises(ValueError, match="only recorded hf"):
        w.choose(Args(data=str(path), baseline=pred, predictions=[pred], out=str(corpus / "choice.json")))


def test_checkpoint_receipt_rejects_stage_splicing(corpus):
    path = corpus / "data/val.jsonl"; data = w.read_rows(path)
    pred = corpus / "spliced.jsonl"
    w.write_rows(pred, [{"problem_id": r["problem_id"], "completion": r["target"], "engine": "hf", "checkpoint_id": "math-best" if r["stage"] == "analysis" else "code-best"} for r in data])
    w.write_json(str(pred) + ".manifest.json", {"source_data_sha256": w.digest(path), "predictions_sha256": w.digest(pred), "checkpoint_id": "math-best", "non_model_reference": False})
    with pytest.raises(ValueError, match="mixed or unbound"):
        w.model_receipt(path, pred)


def test_checkpoint_receipt_detects_prediction_edit(corpus):
    path = corpus / "data/val.jsonl"; data = w.read_rows(path)
    pred = corpus / "edited.jsonl"
    w.write_rows(pred, [{"problem_id": r["problem_id"], "completion": r["target"], "engine": "hf", "checkpoint_id": "one-fixture"} for r in data])
    w.write_json(str(pred) + ".manifest.json", {"source_data_sha256": w.digest(path), "predictions_sha256": "old-file-hash", "checkpoint_id": "one-fixture", "non_model_reference": False})
    with pytest.raises(ValueError, match="does not match"):
        w.model_receipt(path, pred)


def test_primary_regression_is_not_an_eligible_candidate(corpus):
    path = corpus / "data/val.jsonl"; data = w.read_rows(path)
    baseline = oracle_predictions(data, corpus / "baseline.jsonl")
    bad = corpus / "primary_regression.jsonl"
    w.write_rows(bad, [{"problem_id": r["problem_id"], "completion": r["target"] if r["stage"] == "anchor" else "invalid", "engine": "oracle"} for r in data])
    out = corpus / "choice.json"
    w.choose(Args(data=str(path), baseline=baseline, predictions=[str(bad)], out=str(out), allow_reference_demo=True))
    result = json.loads(out.read_text())
    assert result["selected"]["predictions"] == str(bad)
    assert result["release_candidate"] is None
    assert result["release_ready"] is False
    assert result["selected_role"].startswith("research evaluation only")


def test_router_is_explicitly_not_a_single_student(corpus):
    path = corpus / "data/val.jsonl"; data = w.read_rows(path)
    ref = oracle_predictions(data, corpus / "reference.jsonl")
    out = corpus / "route.jsonl"
    w.route(Args(data=str(path), math=ref, code=ref, anchor=ref, out=str(out), allow_reference_demo=True))
    rows = w.read_rows(out)
    assert all(r["engine"] == "expert_router_reference_demo" and r["checkpoint_id"] is None for r in rows)
    receipt = json.loads((corpus / "route.jsonl.manifest.json").read_text())
    assert receipt["reference_demo_only"] is True
    assert receipt["checkpoint_id"] is None
