"""BM25 inverted index for full-text search over string fields."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._postings: Dict[str, Dict[str, int]] = defaultdict(dict)  # term -> {doc_id: tf}
        self._doc_len: Dict[str, int] = {}
        self._total_len = 0

    def add(self, doc_id: str, text: str) -> None:
        if doc_id in self._doc_len:
            self.remove(doc_id)
        tokens = tokenize(text)
        for term, tf in Counter(tokens).items():
            self._postings[term][doc_id] = tf
        self._doc_len[doc_id] = len(tokens)
        self._total_len += len(tokens)

    def remove(self, doc_id: str) -> None:
        length = self._doc_len.pop(doc_id, None)
        if length is None:
            return
        self._total_len -= length
        for term in list(self._postings):
            self._postings[term].pop(doc_id, None)
            if not self._postings[term]:
                del self._postings[term]

    def search(self, query: str, topk: int,
               allowed: Optional[Set[str]] = None) -> List[Tuple[str, float]]:
        n_docs = len(self._doc_len)
        if n_docs == 0:
            return []
        avg_len = self._total_len / n_docs
        scores: Dict[str, float] = defaultdict(float)
        for term in tokenize(query):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = math.log(1 + (n_docs - len(postings) + 0.5) / (len(postings) + 0.5))
            for doc_id, tf in postings.items():
                if allowed is not None and doc_id not in allowed:
                    continue
                norm = self.k1 * (1 - self.b + self.b * self._doc_len[doc_id] / avg_len)
                scores[doc_id] += idf * tf * (self.k1 + 1) / (tf + norm)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:topk]
