"""Schema and document types for AnthosVec collections."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union


class DataType(str, Enum):
    VECTOR_FP32 = "vector_fp32"
    STRING = "string"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"


class Metric(str, Enum):
    COSINE = "cosine"
    DOT = "dot"
    L2 = "l2"


@dataclass
class VectorSchema:
    name: str
    data_type: DataType = DataType.VECTOR_FP32
    dim: int = 0
    metric: Metric = Metric.COSINE

    def __post_init__(self) -> None:
        # allow VectorSchema("embedding", DataType.VECTOR_FP32, 4) positional style
        if self.dim <= 0:
            raise ValueError(f"vector field '{self.name}' needs dim > 0")
        self.data_type = DataType(self.data_type)
        self.metric = Metric(self.metric)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "data_type": self.data_type.value,
            "dim": self.dim,
            "metric": self.metric.value,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VectorSchema":
        return cls(name=d["name"], data_type=DataType(d["data_type"]),
                   dim=d["dim"], metric=Metric(d["metric"]))


@dataclass
class FieldSchema:
    name: str
    data_type: DataType = DataType.STRING
    full_text: bool = False  # build a BM25 index over this string field

    def __post_init__(self) -> None:
        self.data_type = DataType(self.data_type)
        if self.full_text and self.data_type != DataType.STRING:
            raise ValueError(f"full_text is only valid on STRING fields ('{self.name}')")

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "data_type": self.data_type.value,
                "full_text": self.full_text}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FieldSchema":
        return cls(name=d["name"], data_type=DataType(d["data_type"]),
                   full_text=d.get("full_text", False))


@dataclass
class CollectionSchema:
    name: str
    vectors: Union[VectorSchema, Sequence[VectorSchema]] = ()
    fields: Sequence[FieldSchema] = ()

    def __post_init__(self) -> None:
        if isinstance(self.vectors, VectorSchema):
            self.vectors = [self.vectors]
        else:
            self.vectors = list(self.vectors)
        self.fields = list(self.fields)
        if not self.vectors:
            raise ValueError("a collection needs at least one vector field")
        names = [v.name for v in self.vectors] + [f.name for f in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("duplicate field names in schema")

    def vector(self, name: str) -> VectorSchema:
        for v in self.vectors:
            if v.name == name:
                return v
        raise KeyError(f"no vector field named '{name}'")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "vectors": [v.to_dict() for v in self.vectors],
            "fields": [f.to_dict() for f in self.fields],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CollectionSchema":
        return cls(
            name=d["name"],
            vectors=[VectorSchema.from_dict(v) for v in d["vectors"]],
            fields=[FieldSchema.from_dict(f) for f in d["fields"]],
        )


@dataclass
class Doc:
    id: str
    vectors: Dict[str, List[float]] = field(default_factory=dict)
    fields: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None

    def __repr__(self) -> str:
        score = f", score={self.score:.4f}" if self.score is not None else ""
        return f"Doc(id={self.id!r}, fields={self.fields!r}{score})"
