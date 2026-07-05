"""Collection: the core AnthosVec object.

A collection lives in a directory on disk:

    path/
      schema.json     collection schema
      docs.jsonl      last compacted snapshot of all documents
      wal.jsonl       append-only write-ahead log since the snapshot

Every mutation is appended to the WAL before it is applied in memory, so a
crash never loses acknowledged writes. ``flush()`` folds the WAL into a new
snapshot; it also runs automatically once the WAL outgrows the snapshot.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Union

from .filters import compile_filter
from .index import FlatIndex
from .schema import CollectionSchema, DataType, Doc
from .text import BM25Index

_RRF_K = 60  # standard reciprocal-rank-fusion constant


@dataclass
class VectorQuery:
    field: str
    vector: Sequence[float]


@dataclass
class TextQuery:
    field: str
    text: str


Query = Union[VectorQuery, TextQuery]


class Collection:
    def __init__(self, path: str, schema: CollectionSchema) -> None:
        self.path = path
        self.schema = schema
        self._docs: Dict[str, Dict[str, Any]] = {}
        self._indexes = {v.name: FlatIndex(v.dim, v.metric) for v in schema.vectors}
        self._text = {f.name: BM25Index() for f in schema.fields if f.full_text}
        self._wal = open(os.path.join(path, "wal.jsonl"), "a", encoding="utf-8")
        self._load()

    # ------------------------------------------------------------------ writes

    def insert(self, docs: Union[Doc, Iterable[Doc]]) -> int:
        """Insert new documents. Raises if an id already exists."""
        return self._write(docs, overwrite=False)

    def upsert(self, docs: Union[Doc, Iterable[Doc]]) -> int:
        """Insert documents, replacing any existing ones with the same id."""
        return self._write(docs, overwrite=True)

    def delete(self, ids: Union[str, Iterable[str]]) -> int:
        if isinstance(ids, str):
            ids = [ids]
        deleted = 0
        for doc_id in ids:
            if doc_id not in self._docs:
                continue
            self._log({"op": "delete", "id": doc_id})
            self._apply_delete(doc_id)
            deleted += 1
        self._maybe_autoflush()
        return deleted

    def _write(self, docs: Union[Doc, Iterable[Doc]], overwrite: bool) -> int:
        if isinstance(docs, Doc):
            docs = [docs]
        written = 0
        for doc in docs:
            record = self._validate(doc)
            if not overwrite and doc.id in self._docs:
                raise ValueError(f"doc id '{doc.id}' already exists (use upsert)")
            self._log({"op": "upsert", **record})
            self._apply_upsert(record)
            written += 1
        self._maybe_autoflush()
        return written

    def _validate(self, doc: Doc) -> Dict[str, Any]:
        if not isinstance(doc.id, str) or not doc.id:
            raise ValueError("doc.id must be a non-empty string")
        for vs in self.schema.vectors:
            if vs.name not in doc.vectors:
                raise ValueError(f"doc '{doc.id}' is missing vector '{vs.name}'")
        for name in doc.vectors:
            self.schema.vector(name)  # raises on unknown vector field
        known_fields = {f.name for f in self.schema.fields}
        unknown = set(doc.fields) - known_fields
        if unknown:
            raise ValueError(f"doc '{doc.id}' has unknown fields: {sorted(unknown)}")
        return {
            "id": doc.id,
            "vectors": {k: [float(x) for x in v] for k, v in doc.vectors.items()},
            "fields": dict(doc.fields),
        }

    # ------------------------------------------------------------------- reads

    def get(self, ids: Union[str, Iterable[str]],
            include_vectors: bool = True) -> List[Optional[Doc]]:
        if isinstance(ids, str):
            ids = [ids]
        return [self._to_doc(i, include_vectors) if i in self._docs else None
                for i in ids]

    def query(self, queries: Union[Query, Sequence[Query]], topk: int = 10,
              filter: Optional[str] = None, include_vectors: bool = False) -> List[Doc]:
        """Run one query, or fuse several (hybrid) with reciprocal-rank fusion."""
        if isinstance(queries, (VectorQuery, TextQuery)):
            queries = [queries]
        if not queries:
            raise ValueError("query() needs at least one VectorQuery or TextQuery")

        allowed = self._filtered_ids(filter)
        rankings = [self._run_one(q, topk if len(queries) == 1 else topk * 4, allowed)
                    for q in queries]

        if len(rankings) == 1:
            scored = rankings[0]
        else:
            fused: Dict[str, float] = {}
            for ranking in rankings:
                for rank, (doc_id, _) in enumerate(ranking):
                    fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
            scored = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:topk]

        results = []
        for doc_id, score in scored:
            doc = self._to_doc(doc_id, include_vectors)
            doc.score = score
            results.append(doc)
        return results

    def _run_one(self, q: Query, topk: int, allowed: Optional[Set[str]]):
        if isinstance(q, VectorQuery):
            return self._index_of(q.field).search(q.vector, topk, allowed)
        if isinstance(q, TextQuery):
            if q.field not in self._text:
                raise KeyError(
                    f"'{q.field}' has no full-text index "
                    f"(declare FieldSchema(..., full_text=True))")
            return self._text[q.field].search(q.text, topk, allowed)
        raise TypeError(f"unsupported query type: {type(q).__name__}")

    def _index_of(self, name: str) -> FlatIndex:
        if name not in self._indexes:
            raise KeyError(f"no vector field named '{name}'")
        return self._indexes[name]

    def _filtered_ids(self, filter: Optional[str]) -> Optional[Set[str]]:
        if filter is None:
            return None
        predicate = compile_filter(filter)
        return {i for i, rec in self._docs.items() if predicate(rec["fields"])}

    def _to_doc(self, doc_id: str, include_vectors: bool) -> Doc:
        rec = self._docs[doc_id]
        vectors = {}
        if include_vectors:
            for name, index in self._indexes.items():
                vec = index.get(doc_id)
                if vec is not None:
                    vectors[name] = vec.tolist()
        return Doc(id=doc_id, vectors=vectors, fields=dict(rec["fields"]))

    def __len__(self) -> int:
        return len(self._docs)

    def stats(self) -> Dict[str, Any]:
        return {
            "name": self.schema.name,
            "docs": len(self._docs),
            "vector_fields": {v.name: {"dim": v.dim, "metric": v.metric.value}
                              for v in self.schema.vectors},
            "full_text_fields": sorted(self._text),
            "path": self.path,
        }

    # ------------------------------------------------------------- persistence

    def flush(self) -> None:
        """Fold the WAL into a fresh snapshot (atomic replace)."""
        fd, tmp = tempfile.mkstemp(dir=self.path, suffix=".jsonl")
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            for doc_id, rec in self._docs.items():
                vectors = {name: index.get(doc_id).tolist()
                           for name, index in self._indexes.items()
                           if index.get(doc_id) is not None}
                out.write(json.dumps({"id": doc_id, "vectors": vectors,
                                      "fields": rec["fields"]}) + "\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, os.path.join(self.path, "docs.jsonl"))
        self._wal.close()
        self._wal = open(os.path.join(self.path, "wal.jsonl"), "w", encoding="utf-8")

    def close(self) -> None:
        self.flush()
        self._wal.close()

    def __enter__(self) -> "Collection":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _log(self, entry: Dict[str, Any]) -> None:
        self._wal.write(json.dumps(entry) + "\n")
        self._wal.flush()

    def _maybe_autoflush(self) -> None:
        try:
            wal_size = os.path.getsize(os.path.join(self.path, "wal.jsonl"))
        except OSError:
            return
        if wal_size > 8 * 1024 * 1024:
            self.flush()

    def _apply_upsert(self, record: Dict[str, Any]) -> None:
        doc_id = record["id"]
        if doc_id in self._docs:
            self._apply_delete(doc_id)
        self._docs[doc_id] = {"fields": record["fields"]}
        for name, vec in record["vectors"].items():
            self._indexes[name].add(doc_id, vec)
        for name, text_index in self._text.items():
            value = record["fields"].get(name)
            if isinstance(value, str):
                text_index.add(doc_id, value)

    def _apply_delete(self, doc_id: str) -> None:
        self._docs.pop(doc_id, None)
        for index in self._indexes.values():
            index.remove(doc_id)
        for text_index in self._text.values():
            text_index.remove(doc_id)

    def _load(self) -> None:
        snapshot = os.path.join(self.path, "docs.jsonl")
        if os.path.exists(snapshot):
            with open(snapshot, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self._apply_upsert(json.loads(line))
        wal_path = os.path.join(self.path, "wal.jsonl")
        if os.path.exists(wal_path):
            with open(wal_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        break  # torn final write after a crash — ignore the tail
                    if entry["op"] == "delete":
                        self._apply_delete(entry["id"])
                    else:
                        self._apply_upsert(entry)
