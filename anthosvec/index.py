"""Flat (brute-force) vector index backed by a contiguous numpy matrix.

Rows are append-only; deletes are tombstoned and compacted once they exceed
a quarter of the matrix. Exact search — no recall trade-off — which is the
right default for the sub-million-vector collections an embedded store sees.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .schema import Metric

_INITIAL_CAPACITY = 1024


class FlatIndex:
    def __init__(self, dim: int, metric: Metric) -> None:
        self.dim = dim
        self.metric = metric
        self._matrix = np.zeros((_INITIAL_CAPACITY, dim), dtype=np.float32)
        self._ids: List[Optional[str]] = []
        self._row_of: Dict[str, int] = {}
        self._tombstones = 0

    def __len__(self) -> int:
        return len(self._row_of)

    def add(self, doc_id: str, vector: Sequence[float]) -> None:
        vec = np.asarray(vector, dtype=np.float32)
        if vec.shape != (self.dim,):
            raise ValueError(
                f"vector for '{doc_id}' has shape {vec.shape}, expected ({self.dim},)")
        if doc_id in self._row_of:
            self._matrix[self._row_of[doc_id]] = vec
            return
        row = len(self._ids)
        if row >= self._matrix.shape[0]:
            grown = np.zeros((self._matrix.shape[0] * 2, self.dim), dtype=np.float32)
            grown[:row] = self._matrix[:row]
            self._matrix = grown
        self._matrix[row] = vec
        self._ids.append(doc_id)
        self._row_of[doc_id] = row

    def remove(self, doc_id: str) -> None:
        row = self._row_of.pop(doc_id, None)
        if row is None:
            return
        self._ids[row] = None
        self._matrix[row] = 0.0
        self._tombstones += 1
        if self._tombstones > max(64, len(self._ids) // 4):
            self._compact()

    def get(self, doc_id: str) -> Optional[np.ndarray]:
        row = self._row_of.get(doc_id)
        return None if row is None else self._matrix[row].copy()

    def search(self, vector: Sequence[float], topk: int,
               allowed: Optional[Set[str]] = None) -> List[Tuple[str, float]]:
        """Return up to topk (doc_id, score) pairs, best first."""
        if not self._row_of:
            return []
        query = np.asarray(vector, dtype=np.float32)
        if query.shape != (self.dim,):
            raise ValueError(f"query vector has shape {query.shape}, expected ({self.dim},)")
        n = len(self._ids)
        matrix = self._matrix[:n]

        if self.metric == Metric.L2:
            # lower distance = better; negate so that higher score = better
            scores = -np.linalg.norm(matrix - query, axis=1)
        elif self.metric == Metric.DOT:
            scores = matrix @ query
        else:  # cosine
            norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(query) or 1.0)
            norms[norms == 0] = 1.0
            scores = (matrix @ query) / norms

        order = np.argsort(scores)[::-1]
        results: List[Tuple[str, float]] = []
        for row in order:
            doc_id = self._ids[row]
            if doc_id is None:
                continue
            if allowed is not None and doc_id not in allowed:
                continue
            results.append((doc_id, float(scores[row])))
            if len(results) == topk:
                break
        return results

    def _compact(self) -> None:
        live = [(doc_id, self._matrix[row])
                for doc_id, row in sorted(self._row_of.items(), key=lambda kv: kv[1])]
        self._matrix = np.zeros((max(_INITIAL_CAPACITY, len(live) * 2), self.dim),
                                dtype=np.float32)
        self._ids = []
        self._row_of = {}
        self._tombstones = 0
        for doc_id, vec in live:
            self._matrix[len(self._ids)] = vec
            self._row_of[doc_id] = len(self._ids)
            self._ids.append(doc_id)
