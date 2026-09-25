"""Eval harness: run a model over the eval set, score, aggregate, compare, gate."""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Sequence

from metrics import contains_match, exact_match, rubric_grade, token_f1

METRIC_KEYS = ["exact_match", "token_f1", "contains", "rubric_score", "rubric_pass"]
# Blocking metrics can fail the gate on regression; the rest are tracked (informational) only.
BLOCKING_KEYS = ["rubric_pass", "contains"]


def run_eval(model: Callable[[str], str], eval_set: Sequence[Dict[str, object]], rubric_threshold: float = 0.7) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    t0 = time.perf_counter()
    for ex in eval_set:
        pred = model(str(ex["question"]))
        golds = [str(ex["gold"])] + [str(a) for a in ex.get("aliases", [])]  # type: ignore[union-attr]
        rg = rubric_grade(pred, ex["rubric"], pass_threshold=rubric_threshold)  # type: ignore[arg-type]
        rows.append({
            "id": ex["id"], "category": ex["category"], "prediction": pred,
            "exact_match": exact_match(pred, golds), "token_f1": round(token_f1(pred, golds), 4),
            "contains": contains_match(pred, golds), "rubric_score": rg["score"], "rubric_pass": float(rg["passed"]),
        })
    elapsed = time.perf_counter() - t0
    return {"rows": rows, "overall": aggregate(rows), "per_category": per_category(rows), "runtime_s": round(elapsed, 4)}


def aggregate(rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    n = max(len(rows), 1)
    return {k: round(sum(float(r[k]) for r in rows) / n, 4) for k in METRIC_KEYS} | {"n": len(rows)}


def per_category(rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    cats = sorted({str(r["category"]) for r in rows})
    return {c: aggregate([r for r in rows if r["category"] == c]) for c in cats}


def compare(reference: Dict[str, object], candidate: Dict[str, object], max_drop: float = 0.05,
            blocking: Sequence[str] = BLOCKING_KEYS) -> Dict[str, object]:
    """Regression comparison: overall + per-category deltas, flag drops > max_drop."""
    ro, co = reference["overall"], candidate["overall"]  # type: ignore[index]
    deltas = {k: round(co[k] - ro[k], 4) for k in METRIC_KEYS}  # type: ignore[index]
    regressions = [f"overall.{k}" for k, d in deltas.items() if d < -max_drop and k in blocking]
    cat_deltas: Dict[str, Dict[str, float]] = {}
    for c, cm in candidate["per_category"].items():  # type: ignore[union-attr]
        rm = reference["per_category"][c]  # type: ignore[index]
        cat_deltas[c] = {k: round(cm[k] - rm[k], 4) for k in METRIC_KEYS}
        regressions += [f"{c}.{k}" for k, d in cat_deltas[c].items() if d < -max_drop and k in blocking]
    # per-item flips (pass -> fail on rubric)
    ref_rows = {r["id"]: r for r in reference["rows"]}  # type: ignore[union-attr]
    flips = [r["id"] for r in candidate["rows"] if ref_rows[r["id"]]["rubric_pass"] == 1.0 and r["rubric_pass"] == 0.0]  # type: ignore[union-attr]
    fixes = [r["id"] for r in candidate["rows"] if ref_rows[r["id"]]["rubric_pass"] == 0.0 and r["rubric_pass"] == 1.0]  # type: ignore[union-attr]
    return {"overall_delta": deltas, "per_category_delta": cat_deltas, "regressions": regressions,
            "pass_to_fail": flips, "fail_to_pass": fixes, "max_drop": max_drop, "blocking_metrics": list(blocking)}


def gate(candidate: Dict[str, object], thresholds: Dict[str, float], comparison: Dict[str, object] | None = None) -> Dict[str, object]:
    """Pass/fail release gate: absolute thresholds AND no flagged regressions."""
    o = candidate["overall"]  # type: ignore[index]
    checks = {f"{k}>={v}": bool(o[k] >= v) for k, v in thresholds.items()}  # type: ignore[index]
    if comparison is not None:
        checks["no_regressions"] = not comparison["regressions"]
    return {"checks": checks, "passed": all(checks.values())}
