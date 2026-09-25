"""Matplotlib SVG plots + RESULTS.md writer for the eval-harness smoke run."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from svg_utils import minify_svg  # noqa: E402

plt.rcParams["svg.hashsalt"] = "ai-learn-10"
plt.rcParams["svg.fonttype"] = "none"  # keep text as <text>, small SVGs
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
_META = {"Date": None}
_COLORS = ["#adb5bd", "#2a9d8f", "#e9c46a", "#e76f51", "#264653"]


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata=_META)
    plt.close(fig)
    path.write_text(minify_svg(buf.getvalue()), encoding="utf-8")
    return path.name


def make_plots(out: Path, runs: Dict[str, Dict[str, Any]], comparisons: Dict[str, Dict[str, Any]], metric_keys: List[str]) -> List[str]:
    out.mkdir(exist_ok=True)
    names = []
    models = list(runs)
    # 1) overall metrics per model
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.8 / len(models)
    x = np.arange(len(metric_keys))
    for i, m in enumerate(models):
        vals = [runs[m]["overall"][k] for k in metric_keys]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=m, color=_COLORS[i % len(_COLORS)], edgecolor="#222")
    ax.set_xticks(x, metric_keys)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Overall metrics by model variant")
    ax.legend(fontsize=8)
    names.append(_save(fig, out / "overall_metrics.svg"))
    # 2) per-category rubric pass rate
    cats = list(next(iter(runs.values()))["per_category"])
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(cats))
    for i, m in enumerate(models):
        vals = [runs[m]["per_category"][c]["rubric_pass"] for c in cats]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=m, color=_COLORS[i % len(_COLORS)], edgecolor="#222")
    ax.set_xticks(x, cats)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rubric pass rate")
    ax.set_title("Per-category rubric pass rate")
    ax.legend(fontsize=8)
    names.append(_save(fig, out / "per_category_rubric.svg"))
    # 3) regression deltas vs reference
    fig, axes = plt.subplots(1, len(comparisons), figsize=(4 * len(comparisons), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, cmp) in zip(axes, comparisons.items()):
        d = [cmp["overall_delta"][k] for k in metric_keys]
        ax.bar(metric_keys, d, color=["#2a9d8f" if v >= 0 else "#e76f51" for v in d], edgecolor="#222")
        ax.axhline(0, color="#333", lw=0.8)
        ax.axhline(-cmp["max_drop"], color="#e76f51", ls="--", lw=0.8)
        ax.set_title(name, fontsize=9)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    axes[0].set_ylabel("candidate - reference")
    names.append(_save(fig, out / "regression_deltas.svg"))
    return names


def write_results_md(out: Path, m: Dict[str, Any], plots: List[str]) -> None:
    keys = m["metric_keys"]
    L = ["# Results -- ai-learn-10-eval-harness", "", f"**Seed:** `{m['seed']}` | eval items={m['n_items']} | categories={len(m['categories'])} | "
         f"rubric pass threshold={m['config']['rubric_threshold']} | max allowed drop={m['config']['max_drop']}", "",
         "## Overall metrics (real smoke run)", "", "| Model | " + " | ".join(keys) + " |", "|---|" + "---:|" * len(keys)]
    for name, r in m["models"].items():
        L.append(f"| `{name}` | " + " | ".join(f"{r['overall'][k]:.4f}" for k in keys) + " |")
    L += ["", "## Per-category rubric pass rate", "", "| Model | " + " | ".join(m["categories"]) + " |", "|---|" + "---:|" * len(m["categories"])]
    for name, r in m["models"].items():
        L.append(f"| `{name}` | " + " | ".join(f"{r['per_category'][c]['rubric_pass']:.3f}" for c in m["categories"]) + " |")
    L += ["", "## Regression comparisons + release gate", "", f"Gate thresholds: `{m['config']['gate_thresholds']}` and no drop larger than {m['config']['max_drop']} on blocking metrics `{m['config']['blocking_metrics']}` (EM / token F1 / rubric_score are tracked, non-blocking).", "",
          "| Candidate vs reference | Δ rubric_pass | Δ token_f1 | Δ EM | regressions | fail→pass | pass→fail | Gate |", "|---|---:|---:|---:|---|---|---|:---:|"]
    for name, c in m["comparisons"].items():
        d = c["overall_delta"]
        L.append(f"| {name} | {d['rubric_pass']:+.4f} | {d['token_f1']:+.4f} | {d['exact_match']:+.4f} | {', '.join(c['regressions']) or '-'} | "
                 f"{', '.join(c['fail_to_pass']) or '-'} | {', '.join(c['pass_to_fail']) or '-'} | {'PASS' if c['gate']['passed'] else 'FAIL'} |")
    L += ["", f"**Best gated candidate:** `{m['best_gated_model']}`", "", "## Plots", ""] + [f"![{p}]({p})" for p in plots]
    L += ["", "## Sample predictions (`rag_top3_concat`)", "", "| id | cat | prediction | F1 | rubric |", "|---|---|---|---:|:---:|"]
    for row in m["models"]["rag_top3_concat"]["rows"][:10]:
        L.append(f"| {row['id']} | {row['category']} | {row['prediction'][:70]} | {row['token_f1']:.2f} | {'Y' if row['rubric_pass'] else 'N'} |")
    L += ["", f"Wall time: {m['runtime_s']:.3f}s on CPU.", ""]
    (out / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")
