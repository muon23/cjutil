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
# export GOOGLE_API_KEY="..."
# export REPLICATE_API_TOKEN="..."
# export HUGGINGFACEHUB_API_TOKEN="..."
# export DEEPINFRA_API_KEY="..."
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
- `src/main/vector_stores`: Vector store abstraction with PostgreSQL/pgvector backend.
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

## Notes

- Local embedding models are downloaded automatically by `sentence-transformers` on first use.
- Hosted embedding/LLM providers require corresponding API keys (for example `OPENAI_API_KEY`).
- PostgreSQL modules require the `psycopg` package, and vector store usage requires `pgvector` installed in Postgres.
