"""AnthosVec — an embedded, in-process vector database by Anthos Intelligence.

    import anthosvec

    schema = anthosvec.CollectionSchema(
        name="example",
        vectors=anthosvec.VectorSchema("embedding", anthosvec.DataType.VECTOR_FP32, 4),
    )
    collection = anthosvec.create_and_open("./data", schema)
    collection.insert([anthosvec.Doc(id="doc_1", vectors={"embedding": [0.1, 0.2, 0.3, 0.4]})])
    results = collection.query(
        anthosvec.VectorQuery("embedding", vector=[0.4, 0.3, 0.3, 0.1]), topk=10)
"""

from __future__ import annotations

import json
import os

from .collection import Collection, TextQuery, VectorQuery
from .schema import (CollectionSchema, DataType, Doc, FieldSchema, Metric,
                     VectorSchema)

__version__ = "0.1.0"

__all__ = [
    "Collection", "CollectionSchema", "DataType", "Doc", "FieldSchema",
    "Metric", "TextQuery", "VectorQuery", "VectorSchema",
    "create_and_open", "open_collection",
]


def create_and_open(path: str, schema: CollectionSchema) -> Collection:
    """Create a new collection at ``path`` (or open it if one already exists
    with an identical schema) and return it."""
    schema_path = os.path.join(path, "schema.json")
    if os.path.exists(schema_path):
        existing = _read_schema(schema_path)
        if existing.to_dict() != schema.to_dict():
            raise ValueError(
                f"collection at '{path}' exists with a different schema")
        return Collection(path, existing)
    os.makedirs(path, exist_ok=True)
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema.to_dict(), f, indent=2)
    return Collection(path, schema)


def open_collection(path: str) -> Collection:
    """Open an existing collection at ``path``."""
    schema_path = os.path.join(path, "schema.json")
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"no collection at '{path}'")
    return Collection(path, _read_schema(schema_path))


def _read_schema(schema_path: str) -> CollectionSchema:
    with open(schema_path, encoding="utf-8") as f:
        return CollectionSchema.from_dict(json.load(f))
