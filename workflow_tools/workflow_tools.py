"""Companion tools for data views, offline teacher targets, and validation decisions.

The tools do not train models and never turn an automatic score into a reviewed proof.
Run --help for the complete command contract.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))


def read_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [r["problem_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate problem_id in " + str(path))
    if not rows:
        raise ValueError("empty data: " + str(path))
    return rows


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def predictions_for(data, path):
    pred = read_rows(path)
    if any("target" in p or "private_reference" in p for p in pred):
        raise ValueError("prediction file must not contain private labels")
    mapping = {p["problem_id"]: p for p in pred}
    expected = {r["problem_id"] for r in data}
    if set(mapping) != expected:
        raise ValueError("prediction coverage differs from data; do not silently drop failed cases")
    if any(not isinstance(p.get("completion"), str) for p in pred):
        raise ValueError("every prediction requires a completion string")
    return mapping


def scored_rows(data, path):
    from sci_mopd.verify import check_completion
    mapping = predictions_for(data, path)
    return [check_completion(r, mapping[r["problem_id"]]["completion"]) for r in data]


def model_receipt(data_path, predictions_path, allow_reference_demo=False):
    data = read_rows(data_path)
    mapping = predictions_for(data, predictions_path)
    if allow_reference_demo:
        return {"reference_demo_only": True, "checkpoint_id": None}
    if any(p.get("engine") != "hf" for p in mapping.values()):
        raise ValueError("only recorded hf inference can be a model candidate; references require --allow-reference-demo")
    receipt_path = Path(str(predictions_path) + ".manifest.json")
    if not receipt_path.exists():
        raise ValueError("model inference receipt is missing: " + str(receipt_path))
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("predictions_sha256") != digest(predictions_path):
        raise ValueError("prediction file does not match its inference receipt")
    if receipt.get("source_data_sha256") != digest(data_path):
        raise ValueError("inference receipt belongs to different input data")
    checkpoint = receipt.get("checkpoint_id")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError("model inference receipt requires a checkpoint_id")
    if {p.get("checkpoint_id") for p in mapping.values()} != {checkpoint}:
        raise ValueError("mixed or unbound checkpoint identities in prediction file")
    if receipt.get("non_model_reference") is not False:
        raise ValueError("reference inference cannot be admitted as a trained model")
    return {"reference_demo_only": False, "checkpoint_id": checkpoint, "receipt_sha256": digest(receipt_path)}


def rates(rows):
    result = {}
    for stage in ("analysis", "code", "anchor"):
        selected = [r for r in rows if r["stage"] == stage]
        n = len(selected)
        passed = sum(bool(r["auto_pass"]) for r in selected)
        result[stage] = {"n": n, "auto_pass": passed, "rate": passed / n if n else None}
    if not result["analysis"]["n"] or not result["code"]["n"]:
        macro = None
    else:
        macro = (result["analysis"]["rate"] + result["code"]["rate"]) / 2
    grouped = {}
    for row in rows:
        if row["stage"] in {"analysis", "code"}:
            stages = grouped.setdefault(row["group_id"], {})
            if row["stage"] in stages:
                raise ValueError("duplicate stage inside a problem group")
            stages[row["stage"]] = bool(row["auto_pass"])
    complete = [stages for stages in grouped.values() if set(stages) == {"analysis", "code"}]
    paired_pass = sum(all(stages.values()) for stages in complete)
    paired = {
        "complete_groups": len(complete), "both_pass": paired_pass,
        "rate": paired_pass / len(complete) if complete else None,
        "incomplete_groups": len(grouped) - len(complete),
        "scope": "same-input independent-stage joint pass; not sequential interaction",
    }
    return {"by_stage": result, "primary_macro": macro, "anchor_rate": result["anchor"]["rate"], "paired_primary": paired}


def paired_interval(data, before, after, stage, samples=2000, seed=29):
    if samples < 100:
        raise ValueError("at least 100 bootstrap draws required")
    groups = {}
    for row, left, right in zip(data, before, after):
        if row["stage"] == stage:
            groups.setdefault(row["group_id"], []).append(int(right["auto_pass"]) - int(left["auto_pass"]))
    if len(groups) < 2:
        raise ValueError("at least two independent groups required")
    group_values = [statistics.mean(v) for v in groups.values()]
    rng = random.Random(seed)
    draws = sorted(statistics.mean(rng.choices(group_values, k=len(group_values))) for _ in range(samples))
    return {
        "groups": len(group_values), "difference": statistics.mean(group_values),
        "bootstrap_95_interval": [draws[int(.025 * (samples - 1))], draws[int(.975 * (samples - 1))]],
        "draws": samples, "seed": seed,
        "interpretation": "paired group resampling; validation selection is not an unbiased final test",
    }


def select(args):
    rows = read_rows(args.data)
    if args.stage:
        rows = [r for r in rows if r["stage"] in args.stage]
    if args.pool:
        rows = [r for r in rows if r["pool"] in args.pool]
    if not rows:
        raise ValueError("selection is empty")
    write_rows(args.out, rows)
    print(json.dumps({"selected_rows": len(rows), "groups": len({r["group_id"] for r in rows}), "out": args.out}))


def report(args):
    data = read_rows(args.data)
    mapping = predictions_for(data, args.predictions)
    receipt = model_receipt(args.data, args.predictions) if {r.get("engine") for r in mapping.values()} == {"hf"} else None
    output = {
        **rates(scored_rows(data, args.predictions)),
        "prediction_engines": dict(Counter(r.get("engine", "unspecified") for r in mapping.values())),
        "data_sha256": digest(args.data), "predictions_sha256": digest(args.predictions),
        "split_counts": dict(Counter(r["split"] for r in data)),
        "manual_explanation_review": "not_performed", "selection_performed": False,
        "single_model_receipt": receipt,
    }
    write_json(args.out, output)
    print(json.dumps(output))


def route(args):
    data = read_rows(args.data)
    files = {"analysis": args.math, "code": args.code, "anchor": args.anchor}
    demo = getattr(args, "allow_reference_demo", False)
    receipts = {stage: model_receipt(args.data, path, demo) for stage, path in files.items()}
    maps = {stage: predictions_for(data, path) for stage, path in files.items()}
    result = []
    for row in data:
        item = dict(maps[row["stage"]][row["problem_id"]])
        item["source_checkpoint_id"] = item.get("checkpoint_id")
        item["checkpoint_id"] = None
        item["engine"] = "expert_router_reference_demo" if demo else "expert_router"
        result.append(item)
    write_rows(args.out, result)
    write_json(str(args.out) + ".manifest.json", {
        "kind": "explicit-stage multi-model routing baseline, not a single student",
        "component_receipts": receipts, "non_model_reference": demo,
        "source_data_sha256": digest(args.data), "predictions_sha256": digest(args.out),
        "reference_demo_only": demo, "checkpoint_id": None,
    })
    print(json.dumps({"routed_rows": len(result), "single_student": False, "reference_demo_only": demo, "out": args.out}))


def teacher_targets(args):
    data = read_rows(args.data)
    if any(r.get("split") != "train" or r.get("pool") != "distill" for r in data):
        raise ValueError("offline targets may only be prepared from train/distill rows")
    mapping = predictions_for(data, args.predictions)
    receipt = model_receipt(args.data, args.predictions, args.allow_reference_demo)
    from sci_mopd.verify import check_completion
    output, counts = [], Counter()
    for row in data:
        prediction = mapping[row["problem_id"]]
        result = check_completion(row, prediction["completion"])
        counts["generated"] += 1
        counts["auto_pass"] += int(bool(result["auto_pass"]))
        if args.accepted_only and not result["auto_pass"]:
            continue
        copied = dict(row)
        copied["target"] = prediction["completion"]
        copied["target_provenance"] = {
            "kind": "reference_pipeline_demo" if args.allow_reference_demo else "teacher_generation",
            "predictions_sha256": digest(args.predictions), "auto_pass": result["auto_pass"],
            "explanation_human_reviewed": False, **receipt,
        }
        output.append(copied)
    if not output:
        raise ValueError("no teacher targets retained; teacher/data require review")
    write_rows(args.out, output)
    report = {
        **counts, "retained": len(output), "accepted_only": args.accepted_only,
        "per_stage_retained": dict(Counter(r["stage"] for r in output)),
        "data_sha256": digest(args.data), "predictions_sha256": digest(args.predictions),
        "manual_explanation_review": "not_performed",
        "reference_demo_only": args.allow_reference_demo,
    }
    write_json(args.report, report)
    print(json.dumps(report))


def admit(args):
    data = read_rows(args.data)
    if any(r.get("split") != "val" for r in data):
        raise ValueError("teacher admission uses validation data only")
    demo = getattr(args, "allow_reference_demo", False)
    base_receipt = model_receipt(args.data, args.baseline, demo)
    teacher_receipt = model_receipt(args.data, args.teacher, demo)
    baseline = scored_rows(data, args.baseline)
    teacher = scored_rows(data, args.teacher)
    interval = paired_interval(data, baseline, teacher, args.stage, args.bootstrap, args.seed)
    positive = interval["difference"] > 0
    ci_positive = interval["bootstrap_95_interval"][0] > 0
    automatic = positive and (ci_positive or not args.require_positive_ci)
    result = {
        "stage": args.stage, "baseline": rates(baseline), "teacher": rates(teacher),
        "paired_validation": interval, "automatic_gain_gate": automatic,
        "require_positive_ci": args.require_positive_ci,
        "manual_review_required": True, "manual_review_status": "not_performed",
        "status": "reference_demo_only" if demo else "ready_for_behavior_review" if automatic else "not_ready",
        "reference_demo_only": demo, "baseline_receipt": base_receipt, "teacher_receipt": teacher_receipt,
        "data_sha256": digest(args.data), "baseline_sha256": digest(args.baseline),
        "teacher_sha256": digest(args.teacher),
        "scope": "automatic task fields only; does not certify arbitrary derivations or prefixes",
    }
    write_json(args.out, result)
    print(json.dumps({"automatic_gain_gate": automatic, "manual_review_required": True, "out": args.out}))


def choose(args):
    data = read_rows(args.data)
    if any(r.get("split") != "val" for r in data):
        raise ValueError("checkpoint selection uses validation data only")
    demo = getattr(args, "allow_reference_demo", False)
    base_receipt = model_receipt(args.data, args.baseline, demo)
    baseline = rates(scored_rows(data, args.baseline))
    if baseline["anchor_rate"] is None:
        raise ValueError("anchor validation rows required")
    candidates = []
    for path in args.predictions:
        receipt = model_receipt(args.data, path, demo)
        score = rates(scored_rows(data, path))
        if score["primary_macro"] is None or score["anchor_rate"] is None:
            raise ValueError("all three validation stages are required")
        eligible = score["anchor_rate"] >= baseline["anchor_rate"] and all(
            score["by_stage"][s]["rate"] >= baseline["by_stage"][s]["rate"] for s in ("analysis", "code")
        )
        candidates.append({"predictions": str(path), "predictions_sha256": digest(path), **score, "eligible": eligible, "receipt": receipt})
    eligible = [c for c in candidates if c["eligible"]]
    key = lambda c: (c["primary_macro"], c["anchor_rate"])
    winner = max(candidates, key=key) if candidates else None
    release_candidate = max(eligible, key=key) if eligible else None
    result = {
        "selection_rule": "research: maximize primary macro, then anchor; ties keep first listed; release gate separately requires no point-estimate regression in all three stages",
        "baseline": baseline, "candidates": candidates, "selected": winner,
        "selected_role": "research evaluation only, including unfavorable results",
        "release_candidate": release_candidate,
        "baseline_receipt": base_receipt, "reference_demo_only": demo,
        "automatic_primary_gain": bool(winner and winner["primary_macro"] > baseline["primary_macro"]),
        "manual_review_status": "not_performed", "release_ready": False,
        "data_sha256": digest(args.data), "test_used": False,
        "note": "point-estimate validation gate, not a population guarantee; no eligible candidate means no release",
    }
    write_json(args.out, result)
    print(json.dumps({"candidates": len(candidates), "selected": winner["predictions"] if winner else None, "out": args.out}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("select")
    p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    p.add_argument("--stage", nargs="+", choices=["analysis", "code", "anchor"])
    p.add_argument("--pool", nargs="+", choices=["teacher_sft", "distill", "anchor", "evaluation"])
    p.set_defaults(func=select)
    p = commands.add_parser("report")
    p.add_argument("--data", required=True); p.add_argument("--predictions", required=True)
    p.add_argument("--out", required=True); p.set_defaults(func=report)
    p = commands.add_parser("route")
    p.add_argument("--data", required=True); p.add_argument("--math", required=True)
    p.add_argument("--code", required=True); p.add_argument("--anchor", required=True)
    p.add_argument("--out", required=True); p.add_argument("--allow-reference-demo", action="store_true")
    p.set_defaults(func=route)
    p = commands.add_parser("teacher-targets")
    p.add_argument("--data", required=True); p.add_argument("--predictions", required=True)
    p.add_argument("--out", required=True); p.add_argument("--report", required=True)
    p.add_argument("--accepted-only", action="store_true")
    p.add_argument("--allow-reference-demo", action="store_true", help="pipeline test only; never report as teacher training")
    p.set_defaults(func=teacher_targets)
    p = commands.add_parser("admit")
    p.add_argument("--data", required=True); p.add_argument("--baseline", required=True)
    p.add_argument("--teacher", required=True); p.add_argument("--stage", choices=["analysis", "code"], required=True)
    p.add_argument("--out", required=True); p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=29); p.add_argument("--require-positive-ci", action="store_true")
    p.add_argument("--allow-reference-demo", action="store_true", help="reference-only workflow test; not teacher admission")
    p.set_defaults(func=admit)
    p = commands.add_parser("choose")
    p.add_argument("--data", required=True); p.add_argument("--baseline", required=True)
    p.add_argument("--predictions", nargs="+", required=True); p.add_argument("--out", required=True)
    p.add_argument("--allow-reference-demo", action="store_true", help="reference-only workflow test; not release selection")
    p.set_defaults(func=choose)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
