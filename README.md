# AnthosVec

**An open-source, in-process vector database — lightweight, dependency-free (numpy only), and designed to embed directly into your application.**

By [Anthos Intelligence](https://github.com/TushaeBXN/anthos). Think SQLite, but for vectors: no server to run, no service to configure, no network hop. `pip install`, open a directory, search.

```
┌─────────────────────────────┐
│        Your application     │
│  ┌───────────────────────┐  │
│  │       AnthosVec       │  │   no server · no config · no network
│  │  vectors + text + WAL │  │
│  └───────────┬───────────┘  │
└──────────────┼──────────────┘
               ▼
         ./your_data/
```

## Features

- **Embedded** — runs inside your process. Perfect for CLIs, notebooks, edge devices, and desktop apps where deploying a database server is overkill.
- **Exact vector search** — cosine, dot-product, and L2 over float32 vectors, brute-force numpy. No recall trade-offs at embedded scale.
- **Full-text search** — native BM25 keyword search over string fields, no external engine.
- **Hybrid retrieval** — combine vector queries and text queries in one call; results fused with reciprocal-rank fusion.
- **Structured filtering** — `"price < 100 and category == 'books'"` style filters, evaluated safely (no `eval` of arbitrary code).
- **Durable** — write-ahead log; every acknowledged write survives a crash. Snapshots compact automatically.
- **Multiple vector fields per doc** — index a title embedding and a body embedding side by side.

## Install

```bash
pip install anthosvec        # (or, from source:)
pip install -e .
```

Requires Python 3.9+ and numpy.

## Quickstart

```python
import anthosvec

schema = anthosvec.CollectionSchema(
    name="example",
    vectors=anthosvec.VectorSchema("embedding", anthosvec.DataType.VECTOR_FP32, 4),
    fields=[
        anthosvec.FieldSchema("title", anthosvec.DataType.STRING, full_text=True),
        anthosvec.FieldSchema("price", anthosvec.DataType.FLOAT),
    ],
)

collection = anthosvec.create_and_open(path="./anthosvec_example", schema=schema)

collection.insert([
    anthosvec.Doc(id="doc_1", vectors={"embedding": [0.1, 0.2, 0.3, 0.4]},
                  fields={"title": "the little vector database", "price": 9.99}),
    anthosvec.Doc(id="doc_2", vectors={"embedding": [0.2, 0.3, 0.4, 0.1]},
                  fields={"title": "embedded search engines", "price": 149.0}),
])

# vector search
results = collection.query(
    anthosvec.VectorQuery("embedding", vector=[0.4, 0.3, 0.3, 0.1]), topk=10)

# full-text search
results = collection.query(anthosvec.TextQuery("title", "vector database"))

# hybrid: vector + keyword, fused with RRF, with a structured filter
results = collection.query(
    [anthosvec.VectorQuery("embedding", vector=[0.4, 0.3, 0.3, 0.1]),
     anthosvec.TextQuery("title", "vector database")],
    topk=5,
    filter="price < 100",
)

collection.close()  # flushes the WAL into a snapshot
```

Reopen later with `anthosvec.open_collection("./anthosvec_example")` — the schema and data come back exactly as you left them.

## How it's stored

```
your_collection/
  schema.json    # collection schema
  docs.jsonl     # last compacted snapshot
  wal.jsonl      # write-ahead log since the snapshot
```

Every write is appended to the WAL before it's applied, and `flush()` (called automatically when the WAL grows past 8 MB, and on `close()`) atomically replaces the snapshot. A torn final write after a crash is detected and discarded on the next open.

## Roadmap

- [ ] ANN indexes (IVF-Flat, HNSW) for collections past ~1M vectors
- [ ] Sparse vector support
- [ ] Multi-process readers
- [ ] Binary snapshot format (memory-mapped)
- [ ] SDKs beyond Python

## License

Apache-2.0. © Anthos Intelligence (Brian Tushae Thomas).
