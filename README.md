# CJ's Utilities (`cjutil`)

Reusable modules for LLM access, text embeddings, key-value stores, and vector stores.

## Getting Started

### 1) Create and activate virtual environment

```bash
cd cjutil
python3.11 -m venv venv
source venv/bin/activate
```

### 2) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Set environment variables

```bash
export OPENAI_API_KEY="your-openai-key"
# Optional examples:
# export GOOGLE_API_KEY="your-gemini-key"
```

### 4) Configure import path

When running scripts directly, make sure Python can resolve `src/main`:

```bash
export PYTHONPATH="$(pwd)/src/main:$PYTHONPATH"
```

### 5) Quick smoke test

```bash
python - <<'PY'
from embeddings import of as embedding_of
from kv_stores import of as kv_of

embedder = embedding_of("bge-m3")
vec = embedder.embed_query("hello world")
print("Embedding length:", len(vec))

kv = kv_of("dict")
kv.set("k1", {"value": 123})
print("KV value:", kv.get("k1"))
PY
```

## Project Layout

- `src/main/llms`: LLM abstraction and provider implementations.
- `src/main/embeddings`: Text embedding abstraction and provider implementations.
- `src/main/kv_stores`: Key-value store abstraction with in-memory and PostgreSQL backends.
- `src/main/sql_stores`: SQL execution abstraction with PostgreSQL backend.
- `src/main/vector_stores`: Vector store abstraction with PostgreSQL/pgvector backend.
- `src/main/ml`: Traditional ML wrappers (classification, regression, clustering, and data prep).
- `src/test`: Integration-oriented tests for the modules above.

## llms Module

### Overview

`llms` provides a unified interface to multiple LLM providers through a factory method so callers can switch models without changing application logic.

### High-Level Objects

- `Llm` (base class): common invoke/response contract.
- Provider implementations:
  - `GptLlm` (OpenAI)
  - `GeminiLlm` (Google)
  - `DeepInfraLlm`
  - `HuggingFaceLlm`
  - `MockLlm` (for tests/dev)
- `llms.of(model_name, **kwargs)`: returns the correct provider instance.
- `llms.invoke(...)`: convenience helper with instance caching.

### Usage Example

```python
from llms import of, invoke

# Explicit instance creation
bot = of("gpt-4o-mini", model_key="YOUR_API_KEY")
response = bot.invoke("Summarize this text in 3 bullet points.")
print(response.text)

# Convenience one-shot call (with internal cache)
text = invoke("gpt-4o-mini", "Write a short release note.")
print(text)
```

## embeddings Module

### Overview

`embeddings` provides a provider-agnostic way to embed documents and queries, with support for local sentence-transformer models and hosted LangChain/OpenAI models.

### High-Level Objects

- `TextEmbedding` (base class):
  - `embed_documents(texts)` for passages/documents
  - `embed_query(text)` for user queries
- Provider implementations:
  - `SentenceTransformerEmbedding` (local Hugging Face models)
  - `LangChainEmbedding` (currently OpenAI embeddings via LangChain)
- `embeddings.of(model_name, **kwargs)`: provider/model factory.
- Convenience helpers with caching:
  - `embeddings.embed_documents(...)`
  - `embeddings.embed_query(...)`

### Usage Example

```python
from embeddings import of, embed_documents, embed_query

# Local model
local_embedder = of("bge-m3")
doc_vectors = local_embedder.embed_documents([
    "The quick brown fox jumps over the lazy dog.",
    "Embedding models convert text to numeric vectors."
])
query_vector = local_embedder.embed_query("What is an embedding model?")

# Convenience helper (internally caches by model_name)
batch = embed_documents("multilingual-e5-large", ["Hello world", "Bonjour le monde"])
qvec = embed_query("multilingual-e5-large", "search query text")
```

## kv_stores Module

### Overview

`kv_stores` defines a consistent key-value API with both in-memory and PostgreSQL implementations. It supports full upsert (`set`), partial update (`patch`), existence checks, deletes, and batch helpers.

### High-Level Objects

- `KeyValueStore` (base class):
  - `set`, `patch`, `get`, `exists`, `delete`
  - `set_many`, `get_many`
- `KeyNotFoundError`: raised when key is missing and no default is provided.
- `DictKeyValueStore`: lightweight in-memory backend.
- `PostgresKeyValueStore`: typed PostgreSQL backend with schema support.
- `KeyValueSchemaSpec`: schema definition for PostgreSQL table creation.
- `kv_stores.of(store_type, **kwargs)`: backend factory.

### Usage Example

```python
from kv_stores import of, KeyValueSchemaSpec

# In-memory store
mem = of("dict")
mem.set("user:1", {"name": "CJ", "age": 30})
mem.patch("user:1", {"city": "San Francisco"})
print(mem.get("user:1"))

# PostgreSQL store
schema = KeyValueSchemaSpec(
    columns={
        "id": "TEXT",
        "name": "TEXT",
        "age": "INT",
        "expires_at": "TIMESTAMPTZ"
    },
    indexed_columns=["name"],
)

pg = of(
    "postgres",
    dsn="postgresql://localhost:5432/postgres",
    table_name="user_profiles",
    key_columns="id",
    schema_spec=schema,
    create_if_not_exist=True,
)
pg.set("u1", {"name": "CJ", "age": 30})
print(pg.get("u1"))
```

## vector_stores Module

### Overview

`vector_stores` provides a unified vector database interface with a PostgreSQL `pgvector` implementation for upsert, similarity query, and delete operations.

### High-Level Objects

- `VectorStore` (base class):
  - `upsert`, `query`, `delete`
  - `upsert_many`
- `Match`: standardized query result object (`record_id`, `score`, `metadata`, `document`).
- `PGVectorStore`: PostgreSQL + `pgvector` implementation.
- `vector_stores.of(store_type, **kwargs)`: backend factory.

### Usage Example

```python
from vector_stores import of

store = of(
    "pgvector",
    dsn="postgresql://localhost:5432/postgres",
    table_name="documents_index",
    dimension=3,
    create_if_not_exist=True,
    distance="cosine",
)

store.upsert("doc-1", [0.1, 0.2, 0.3], metadata={"lang": "en"}, document="Hello world")
store.upsert("doc-2", [0.9, 0.1, 0.0], metadata={"lang": "fr"}, document="Bonjour le monde")

matches = store.query([0.1, 0.2, 0.25], top_k=5, metadata_filter={"lang": "en"})
for m in matches:
    print(m.record_id, m.score, m.metadata)
```

## sql_stores Module

### Overview

`sql_stores` provides a backend-agnostic SQL execution interface for read/write statements, including explicit transaction scope and context-manager support.

### High-Level Objects

- `SqlStore` (base class):
  - `execute`, `execute_many`, `query`
  - `transaction`, `close`
  - supports `with` statement via `__enter__` / `__exit__`
- `PostgresSqlStore`: `psycopg`-based implementation for PostgreSQL.
- `sql_stores.of(store_type, **kwargs)`: backend factory.

### Usage Example

```python
from sql_stores import of

with of("postgres", dsn="postgresql://localhost:5432/postgres") as db:
    db.execute("CREATE TABLE IF NOT EXISTS account (id TEXT PRIMARY KEY, balance INT NOT NULL)")

    with db.transaction():
        db.execute("INSERT INTO account (id, balance) VALUES (%s, %s)", ("A", 100))
        db.execute("INSERT INTO account (id, balance) VALUES (%s, %s)", ("B", 50))
        db.execute("UPDATE account SET balance = balance - %s WHERE id = %s", (20, "A"))
        db.execute("UPDATE account SET balance = balance + %s WHERE id = %s", (20, "B"))

    rows = db.query("SELECT id, balance FROM account ORDER BY id")
    print(rows)
```

## ml Module

### Overview

`ml` provides thin, reusable wrappers around common `scikit-learn` algorithms plus practical data-prep utilities. The API is task-oriented and exposed through task-specific factories:
- `ml.classifier_of(model_name, **kwargs)`
- `ml.regressor_of(model_name, **kwargs)`
- `ml.cluster_of(model_name, **kwargs)`

### High-Level Objects

- Core:
  - `MlModel`: common interface for `fit`, `predict`, params, and save/load.
- Submodules:
  - `classification.ClassifierModel` with concrete models:
    - `LogisticRegressionClassifier`
    - `RandomForestClassifierModel`
  - `regression.RegressionModel` with concrete models:
    - `LinearRegressionModel`
    - `RidgeRegressionModel`
    - `RandomForestRegressionModel`
  - `clustering.ClusterModel` with concrete models:
    - `KMeansClusterModel`
    - `DBSCANClusterModel`
- Data prep utilities:
  - `DatasetSplitter`
  - `FeatureScaler`
  - `Imputer`
  - `CategoricalEncoder`

### Usage Example

```python
import numpy as np
from ml import classifier_of, cluster_of, DatasetSplitter, FeatureScaler

# Classification example
X = np.array([[0.1, 1.2], [0.2, 1.0], [1.1, 0.2], [1.2, 0.1]])
y = np.array([0, 0, 1, 1])

split = DatasetSplitter.train_test(X, y, test_size=0.25, random_state=42, stratify=True)
scaler = FeatureScaler("standard").fit(split.X_train)

clf = classifier_of("logreg", max_iter=200)
clf.fit(scaler.transform(split.X_train), split.y_train)
pred = clf.predict(scaler.transform(split.X_test))
print("pred:", pred)

# Clustering example
cluster_model = cluster_of("kmeans", k=2, random_state=42)
labels = cluster_model.fit_predict(X)
print("cluster labels:", labels)
```

## Notes

- Local embedding models are downloaded automatically by `sentence-transformers` on first use.
- Hosted embedding/LLM providers require corresponding API keys (for example `OPENAI_API_KEY`).
- PostgreSQL modules require the `psycopg` package, and vector store usage requires `pgvector` installed in Postgres.
