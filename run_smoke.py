#!/usr/bin/env python3
"""Eval-harness smoke: run 4 toy model variants -> score -> compare -> gate -> results/."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from dataset import DOCUMENTS, EVAL_SET
from harness import BLOCKING_KEYS, METRIC_KEYS, compare, gate, run_eval
from models import make_models
from smoke_plots import make_plots, write_results_md

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEED = 42
RUBRIC_THRESHOLD = 0.7
MAX_DROP = 0.05
GATE = {"rubric_pass": 0.75, "contains": 0.6}
# (candidate, reference) pairs for the regression report
PAIRS = [("rag_top1", "baseline_random"), ("rag_top1_span", "rag_top1"), ("rag_top3_concat", "rag_top1")]


def _compact(js: str) -> str:
    """Put short scalar lists on one line to keep metrics.json readable and small."""
    return re.sub(r"\[\s+([^\[\]{}]*?)\s+\]", lambda m: "[" + re.sub(r"\s+", " ", m.group(1)) + "]", js)


def main() -> None:
    t0 = time.perf_counter()
    models = make_models(seed=SEED)
    runs = {name: run_eval(fn, EVAL_SET, rubric_threshold=RUBRIC_THRESHOLD) for name, fn in models.items()}
    comparisons = {}
    for cand, ref in PAIRS:
        c = compare(runs[ref], runs[cand], max_drop=MAX_DROP)
        c["gate"] = gate(runs[cand], GATE, c)
        comparisons[f"{cand} vs {ref}"] = c
    gated = [k.split(" vs ")[0] for k, c in comparisons.items() if c["gate"]["passed"]]
    best = max(gated, key=lambda n: runs[n]["overall"]["rubric_pass"]) if gated else None
    runtime = time.perf_counter() - t0

    metrics = {
        "project": "ai-learn-10-eval-harness", "seed": SEED, "n_docs": len(DOCUMENTS), "n_items": len(EVAL_SET),
        "categories": sorted({str(e["category"]) for e in EVAL_SET}), "metric_keys": METRIC_KEYS,
        "config": {"rubric_threshold": RUBRIC_THRESHOLD, "max_drop": MAX_DROP, "gate_thresholds": GATE, "blocking_metrics": BLOCKING_KEYS, "pairs": PAIRS},
        "models": runs, "comparisons": comparisons, "best_gated_model": best, "runtime_s": round(runtime, 4),
    }
    RESULTS.mkdir(exist_ok=True)
    plots = make_plots(RESULTS, runs, comparisons, METRIC_KEYS)
    metrics["plots"] = plots
    # compact per-item rows (full predictions are sampled in RESULTS.md)
    slim = json.loads(json.dumps(metrics))
    for r in slim["models"].values():
        r["rows"] = {row["id"]: [row[k] for k in METRIC_KEYS] for row in r["rows"]}
    slim["row_format"] = ["id -> " + ", ".join(METRIC_KEYS)]
    (RESULTS / "metrics.json").write_text(_compact(json.dumps(slim, indent=1)), encoding="utf-8")
    shot = {
        "project": metrics["project"], "seed": SEED, "n_items": len(EVAL_SET), "config": metrics["config"],
        "overall": {n: r["overall"] for n, r in runs.items()},
        "gates": {k: c["gate"]["passed"] for k, c in comparisons.items()},
        "best_gated_model": best, "runtime_s": metrics["runtime_s"],
        "pass": bool(best is not None and runs["rag_top1"]["overall"]["rubric_pass"] > runs["baseline_random"]["overall"]["rubric_pass"]),
    }
    (RESULTS / "JSON.shot").write_text(json.dumps(shot, indent=2), encoding="utf-8")
    write_results_md(RESULTS, metrics, plots)

    print(f"items={len(EVAL_SET)} categories={metrics['categories']}")
    for n, r in runs.items():
        print(f"  {n:16s} " + " ".join(f"{k}={r['overall'][k]:.3f}" for k in METRIC_KEYS))
    for k, c in comparisons.items():
        print(f"  [{'PASS' if c['gate']['passed'] else 'FAIL'}] {k}: d_rubric={c['overall_delta']['rubric_pass']:+.3f} regressions={c['regressions']}")
    print(f"best gated model: {best} | runtime {runtime:.3f}s | wrote {RESULTS}")


if __name__ == "__main__":
    main()
