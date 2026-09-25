"""Toy 'model' variants under test (no LLM API; deterministic with seed 42).

- baseline_random:  answers with the first sentence of a random document (no retrieval)
- rag_top1:         TF-IDF retrieve top-1 doc, return the single best query-overlapping sentence
                    (the ai-learn-08 extractive RAG answerer, trimmed to one sentence)
- rag_top1_span:    rag_top1 + span compression: drop question words / stopwords from the chosen
                    sentence and keep the remaining words (short answers -> exact match becomes possible)
- rag_top3_concat:  TF-IDF retrieve top-3 docs, stitch the 2 best sentences (more recall, longer answers)
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from dataset import DOCUMENTS

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "in", "on", "for", "and", "or",
    "with", "by", "from", "what", "who", "how", "why", "when", "where", "which", "do", "does", "did",
    "you", "it", "its", "this", "that", "can", "at", "many", "times", "each", "used",
}


_SPAN_DROP = {"because", "they", "their", "often", "mainly", "occurs", "you", "one", "create", "called",
              "happens", "when", "use", "uses", "such", "as", "about", "into", "than", "has", "have"}


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class TfidfIndex:
    def __init__(self, docs: Sequence[Dict[str, str]]):
        self.docs = list(docs)
        vocab = sorted({t for d in self.docs for t in tokenize(d["text"])})
        self.w2i = {w: i for i, w in enumerate(vocab)}
        df = np.zeros(len(vocab))
        for d in self.docs:
            for t in set(tokenize(d["text"])):
                df[self.w2i[t]] += 1
        n = len(self.docs)
        self.idf = np.log((n + 1) / (df + 1)) + 1.0
        self.X = np.vstack([self._vec(d["text"]) for d in self.docs])

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(len(self.w2i))
        for t in tokenize(text):
            if t in self.w2i:
                v[self.w2i[t]] += 1
        m = v > 0
        v[m] = 1 + np.log(v[m])
        v *= self.idf
        nrm = np.linalg.norm(v)
        return v / nrm if nrm > 0 else v

    def search(self, query: str, k: int = 3) -> List[Tuple[int, float]]:
        s = self.X @ self._vec(query)
        order = np.argsort(-s, kind="stable")[:k]
        return [(int(i), float(s[i])) for i in order]


def _best_sentences(question: str, texts: Sequence[Tuple[int, str]], n: int) -> List[str]:
    q = {t for t in tokenize(question) if t not in _STOP}
    scored = []
    for rank, text in texts:
        for s in _SENT_RE.split(text.strip()):
            st = set(tokenize(s))
            ov = len(q & st) / max(len(q), 1)
            scored.append((ov + 0.05 * max(0, 3 - rank), s.strip()))
    scored.sort(key=lambda x: -x[0])
    out: List[str] = []
    for _, s in scored:
        if s not in out:
            out.append(s)
        if len(out) >= n:
            break
    return out


def make_models(seed: int = 42) -> Dict[str, Callable[[str], str]]:
    index = TfidfIndex(DOCUMENTS)
    rng = np.random.default_rng(seed)

    def baseline_random(question: str) -> str:
        d = DOCUMENTS[int(rng.integers(len(DOCUMENTS)))]
        return _SENT_RE.split(d["text"])[0]

    def rag_top1(question: str) -> str:
        (i, _), = index.search(question, k=1)
        return " ".join(_best_sentences(question, [(0, DOCUMENTS[i]["text"])], n=1))

    def rag_top1_span(question: str) -> str:
        sent = rag_top1(question)
        q = set(tokenize(question))
        keep = [w for w in re.findall(r"[A-Za-z0-9-]+", sent)
                if w.lower() not in q and w.lower() not in _STOP and w.lower() not in _SPAN_DROP]
        return " ".join(keep) if keep else sent

    def rag_top3_concat(question: str) -> str:
        hits = index.search(question, k=3)
        return " ".join(_best_sentences(question, [(r, DOCUMENTS[i]["text"]) for r, (i, _) in enumerate(hits)], n=2))

    return {"baseline_random": baseline_random, "rag_top1": rag_top1,
            "rag_top1_span": rag_top1_span, "rag_top3_concat": rag_top3_concat}
