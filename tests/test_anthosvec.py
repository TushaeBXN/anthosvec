import os
import shutil
import tempfile

import pytest

import anthosvec as av


@pytest.fixture()
def path():
    d = tempfile.mkdtemp()
    yield os.path.join(d, "col")
    shutil.rmtree(d, ignore_errors=True)


def make_schema():
    return av.CollectionSchema(
        name="t",
        vectors=av.VectorSchema("emb", av.DataType.VECTOR_FP32, 3),
        fields=[av.FieldSchema("title", av.DataType.STRING, full_text=True),
                av.FieldSchema("price", av.DataType.FLOAT)],
    )


def seed(col):
    col.insert([
        av.Doc(id="a", vectors={"emb": [1, 0, 0]}, fields={"title": "red apples", "price": 5.0}),
        av.Doc(id="b", vectors={"emb": [0, 1, 0]}, fields={"title": "green pears", "price": 50.0}),
        av.Doc(id="c", vectors={"emb": [0.9, 0.1, 0]}, fields={"title": "apple cider", "price": 8.0}),
    ])


def test_vector_search_orders_by_similarity(path):
    col = av.create_and_open(path, make_schema())
    seed(col)
    res = col.query(av.VectorQuery("emb", [1, 0, 0]), topk=3)
    assert [d.id for d in res] == ["a", "c", "b"]
    assert res[0].score == pytest.approx(1.0)


def test_full_text_search(path):
    col = av.create_and_open(path, make_schema())
    seed(col)
    res = col.query(av.TextQuery("title", "apple"))
    assert {d.id for d in res} == {"c"}  # exact-token match only
    res = col.query(av.TextQuery("title", "apples cider"))
    assert {d.id for d in res} == {"a", "c"}


def test_filter(path):
    col = av.create_and_open(path, make_schema())
    seed(col)
    res = col.query(av.VectorQuery("emb", [1, 0, 0]), topk=3, filter="price < 10")
    assert [d.id for d in res] == ["a", "c"]


def test_filter_rejects_unsafe_expressions(path):
    col = av.create_and_open(path, make_schema())
    seed(col)
    with pytest.raises(ValueError):
        col.query(av.VectorQuery("emb", [1, 0, 0]), filter="__import__('os')")


def test_hybrid_rrf(path):
    col = av.create_and_open(path, make_schema())
    seed(col)
    res = col.query([av.VectorQuery("emb", [1, 0, 0]),
                     av.TextQuery("title", "cider")], topk=3)
    assert res[0].id == "c"  # ranked well by both signals


def test_insert_duplicate_raises_and_upsert_replaces(path):
    col = av.create_and_open(path, make_schema())
    seed(col)
    with pytest.raises(ValueError):
        col.insert(av.Doc(id="a", vectors={"emb": [0, 0, 1]}))
    col.upsert(av.Doc(id="a", vectors={"emb": [0, 0, 1]},
                      fields={"title": "rewritten", "price": 1.0}))
    doc = col.get("a")[0]
    assert doc.fields["title"] == "rewritten"
    assert doc.vectors["emb"] == pytest.approx([0, 0, 1])
    assert len(col) == 3


def test_delete(path):
    col = av.create_and_open(path, make_schema())
    seed(col)
    assert col.delete(["a", "missing"]) == 1
    assert len(col) == 2
    assert col.get("a") == [None]
    assert "a" not in {d.id for d in col.query(av.VectorQuery("emb", [1, 0, 0]), topk=3)}


def test_persistence_via_wal_without_flush(path):
    col = av.create_and_open(path, make_schema())
    seed(col)
    col.delete("b")
    # no flush/close: reopen must replay the WAL
    col2 = av.open_collection(path)
    assert len(col2) == 2
    assert [d.id for d in col2.query(av.VectorQuery("emb", [1, 0, 0]), topk=5)] == ["a", "c"]


def test_persistence_after_close(path):
    with av.create_and_open(path, make_schema()) as col:
        seed(col)
    with av.open_collection(path) as col:
        assert len(col) == 3
        assert col.query(av.TextQuery("title", "cider"))[0].id == "c"


def test_schema_mismatch_rejected(path):
    av.create_and_open(path, make_schema()).close()
    other = av.CollectionSchema(name="t", vectors=av.VectorSchema("emb", dim=8))
    with pytest.raises(ValueError):
        av.create_and_open(path, other)


def test_validation_errors(path):
    col = av.create_and_open(path, make_schema())
    with pytest.raises(ValueError):
        col.insert(av.Doc(id="x", vectors={}))  # missing vector
    with pytest.raises(ValueError):
        col.insert(av.Doc(id="x", vectors={"emb": [1, 2]}))  # wrong dim
    with pytest.raises(ValueError):
        col.insert(av.Doc(id="x", vectors={"emb": [1, 2, 3]}, fields={"nope": 1}))
