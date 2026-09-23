"""BM25 稀疏检索（rank-bm25 生产实现 + 纯 Python 兜底）。作者：晨星"""
from __future__ import annotations

import math
from collections import Counter
from typing import Optional

from .embed import tokenize
from .types import Chunk


class BM25Index:
    """进程内 BM25。优先 rank_bm25，缺失时用内置纯 Python 实现。"""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._corpus_tokens: list[list[str]] = []
        self._bm25 = None
        self._df: Counter[str] = Counter()
        self._avgdl = 0.0
        self._use_lib = False
        try:
            import rank_bm25  # noqa: F401

            self._lib_available = True
        except ImportError:
            self._lib_available = False

    def rebuild(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        self._corpus_tokens = [tokenize(c.text) for c in self._chunks]
        if self._lib_available and self._corpus_tokens:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(self._corpus_tokens)
            self._use_lib = True
        else:
            self._use_lib = False
            self._df = Counter()
            for tokens in self._corpus_tokens:
                for tok in set(tokens):
                    self._df[tok] += 1
            total = sum(len(t) for t in self._corpus_tokens)
            self._avgdl = total / max(len(self._corpus_tokens), 1)

    def _score_builtin(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        k1, b = 1.5, 0.75
        n_docs = max(len(self._corpus_tokens), 1)
        tf = Counter(doc_tokens)
        score = 0.0
        for tok in query_tokens:
            df = self._df.get(tok, 0)
            if df == 0:
                continue
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            f = tf.get(tok, 0)
            denom = f + k1 * (1 - b + b * len(doc_tokens) / max(self._avgdl, 1e-9))
            score += idf * f * (k1 + 1) / max(denom, 1e-9)
        return score

    def search(self, query: str, top_n: int) -> list[tuple[Chunk, float]]:
        if not self._chunks:
            return []
        q_tokens = tokenize(query)
        if self._use_lib and self._bm25 is not None:
            scores = self._bm25.get_scores(q_tokens)
        else:
            scores = [
                self._score_builtin(q_tokens, doc) for doc in self._corpus_tokens
            ]
        ranked = sorted(
            zip(self._chunks, scores), key=lambda x: x[1], reverse=True
        )[:top_n]
        return [(c, float(s)) for c, s in ranked if s > 0]

    def count(self) -> int:
        return len(self._chunks)
