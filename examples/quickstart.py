"""AnthosVec quickstart: vector, full-text, hybrid, and filtered search."""

import shutil

import anthosvec

PATH = "./anthosvec_quickstart"
shutil.rmtree(PATH, ignore_errors=True)

schema = anthosvec.CollectionSchema(
    name="books",
    vectors=anthosvec.VectorSchema("embedding", anthosvec.DataType.VECTOR_FP32, 4),
    fields=[
        anthosvec.FieldSchema("title", anthosvec.DataType.STRING, full_text=True),
        anthosvec.FieldSchema("price", anthosvec.DataType.FLOAT),
    ],
)

with anthosvec.create_and_open(PATH, schema) as col:
    col.insert([
        anthosvec.Doc(id="doc_1", vectors={"embedding": [0.1, 0.2, 0.3, 0.4]},
                      fields={"title": "the little vector database", "price": 9.99}),
        anthosvec.Doc(id="doc_2", vectors={"embedding": [0.2, 0.3, 0.4, 0.1]},
                      fields={"title": "embedded search engines", "price": 149.0}),
        anthosvec.Doc(id="doc_3", vectors={"embedding": [0.9, 0.1, 0.0, 0.0]},
                      fields={"title": "cooking with cast iron", "price": 24.0}),
    ])

    print("-- vector search")
    for doc in col.query(anthosvec.VectorQuery("embedding", [0.4, 0.3, 0.3, 0.1]), topk=3):
        print("  ", doc)

    print("-- full-text search: 'vector database'")
    for doc in col.query(anthosvec.TextQuery("title", "vector database")):
        print("  ", doc)

    print("-- hybrid + filter: price < 100")
    hybrid = [anthosvec.VectorQuery("embedding", [0.4, 0.3, 0.3, 0.1]),
              anthosvec.TextQuery("title", "vector database")]
    for doc in col.query(hybrid, topk=3, filter="price < 100"):
        print("  ", doc)

print("-- reopened from disk")
with anthosvec.open_collection(PATH) as col:
    print("  ", col.stats())
