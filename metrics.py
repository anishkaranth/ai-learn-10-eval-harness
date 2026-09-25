"""Scoring functions for an LLM-style eval harness (pure Python).

- normalize_answer: SQuAD-style normalization (lowercase, strip punctuation/articles)
- exact_match: EM against gold + aliases
- token_f1: bag-of-token overlap F1 (max over gold + aliases)
- contains_match: normalized gold span appears inside the prediction
- rubric_grade: keyword rubric (required / bonus / forbidden / length) -> score in [0, 1]
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, List, Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation) - {"-"}


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch if ch not in _PUNCT else " " for ch in s)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def _tokens(s: str) -> List[str]:
    return normalize_answer(s).split()


def exact_match(pred: str, golds: Sequence[str]) -> float:
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in golds))


def _f1_single(pred: str, gold: str) -> float:
    pt, gt = _tokens(pred), _tokens(gold)
    if not pt or not gt:
        return float(pt == gt)
    common = Counter(pt) & Counter(gt)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pt)
    recall = same / len(gt)
    return 2 * precision * recall / (precision + recall)


def token_f1(pred: str, golds: Sequence[str]) -> float:
    return max(_f1_single(pred, g) for g in golds)


def contains_match(pred: str, golds: Sequence[str]) -> float:
    p = f" {normalize_answer(pred)} "
    return float(any(f" {normalize_answer(g)} " in p for g in golds if normalize_answer(g)))


def rubric_grade(pred: str, rubric: Dict[str, object], pass_threshold: float = 0.7) -> Dict[str, object]:
    """Keyword rubric grader (a cheap stand-in for an LLM judge).

    score = 0.8 * frac(required present) + 0.2 * frac(bonus present)
    Any forbidden keyword or exceeding max_words -> score multiplied by 0.5.
    """
    text = normalize_answer(pred).replace("-", " ")  # "self-attention" also matches "attention"
    toks = set(text.split())
    req: List[str] = list(rubric.get("required", []))  # type: ignore[arg-type]
    bonus: List[str] = list(rubric.get("bonus", []))  # type: ignore[arg-type]
    forbidden: List[str] = list(rubric.get("forbidden", []))  # type: ignore[arg-type]
    max_words = int(rubric.get("max_words", 50))  # type: ignore[arg-type]

    def has(k: str) -> bool:
        k = normalize_answer(k).replace("-", " ")
        return k in toks or (" " in k and k in text)

    req_hit = [k for k in req if has(k)]
    bonus_hit = [k for k in bonus if has(k)]
    forb_hit = [k for k in forbidden if has(k)]
    req_frac = len(req_hit) / len(req) if req else 1.0
    bonus_frac = len(bonus_hit) / len(bonus) if bonus else 1.0
    score = 0.8 * req_frac + 0.2 * bonus_frac
    too_long = len(text.split()) > max_words
    if forb_hit or too_long:
        score *= 0.5
    return {
        "score": round(score, 4),
        "passed": bool(score >= pass_threshold and req_frac == 1.0),
        "required_hit": req_hit,
        "bonus_hit": bonus_hit,
        "forbidden_hit": forb_hit,
        "too_long": too_long,
    }
