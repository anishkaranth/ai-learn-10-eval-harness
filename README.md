# AI Learn 10 — LLM-style Eval Harness from Scratch

Build a small **evaluation harness** like the ones used to gate LLM releases: scoring functions, per-category breakdowns, a regression comparison between model variants, and a pass/fail release gate. No LLM API is used. The "models" are deterministic toy answerers, including the ai-learn-08-style TF-IDF + extractive RAG answerer.

Phase B (AI components) follows the tool-calling agent (`ai-learn-09`). The repo is self-contained and does not import earlier repos.

## Learning goals

- **Exact match (EM)** with SQuAD-style normalization (lowercase, strip punctuation and articles) plus alias answers
- **Token F1**: bag-of-token precision/recall against the gold answer (max over aliases)
- **Contains**: whether the gold span appears inside a longer generated answer
- **Rubric / keyword grader**: required, bonus, and forbidden keywords plus a length penalty. It is a cheap, transparent stand-in for an LLM judge.
- **Per-category breakdown**: python / ml / nlp / science
- **Regression comparison**: candidate vs reference deltas (overall + per category), with pass→fail / fail→pass item flips
- **Release gate**: absolute thresholds AND no drop larger than `max_drop` on *blocking* metrics. The other metrics are tracked but don't block.

## Brief architecture

```
EVAL_SET (question, gold, aliases, category, rubric)
      │
      ▼
model_fn(question) ──► prediction           (baseline_random | rag_top1 | rag_top1_span | rag_top3_concat)
      │
      ▼
metrics: EM · token F1 · contains · rubric_grade
      │
      ▼
aggregate ─► overall + per_category
      │
      ▼
compare(reference, candidate) ─► deltas, regressions, flips
      │
      ▼
gate(thresholds, no blocking regressions) ─► PASS / FAIL
```

| Variant | Idea |
|---------|------|
| `baseline_random` | First sentence of a random doc (no retrieval) |
| `rag_top1` | TF-IDF top-1 doc → best query-overlapping sentence (reference) |
| `rag_top1_span` | `rag_top1` + span compression (drop question words/stopwords) → short answers |
| `rag_top3_concat` | TF-IDF top-3 → 2 best sentences (more recall, longer answers) |

The smoke run shows the classic metric trade-off. Short spans raise EM and F1 but lose `contains`. Longer answers raise recall-style metrics but lower token F1. That's why the gate has to say which metrics are blocking.

## Layout

```
dataset.py             # toy corpus + 24 labeled eval items with categories + rubrics
metrics.py             # normalize, exact_match, token_f1, contains_match, rubric_grade
models.py              # TF-IDF index + 4 toy model variants
harness.py             # run_eval, aggregate, per_category, compare, gate
smoke_plots.py         # matplotlib SVG plots + RESULTS.md
run_smoke.py           # end-to-end smoke -> results/
notebooks/eval_harness.ipynb
results/               # committed RESULTS.md, metrics.json, JSON.shot, *.svg
```

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_smoke.py
```

Runs on CPU in about a second with seed 42. See `results/RESULTS.md` for the latest smoke metrics.

## What you'll learn next

LoRA from scratch (`ai-learn-11`), a toy CLIP (`ai-learn-12`), guardrails and structured output (`ai-learn-13`), and then the Phase C end-to-end assistant milestone (`ai-learn-14`).
