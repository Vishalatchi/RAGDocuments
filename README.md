# RAGofDocuments

## 1. Project Overview

This repository contains a FastAPI application titled `RAG system` and Python code that loads Markdown files, splits them into chunks, embeds those chunks, and stores them in Qdrant. It also contains hybrid retrieval, cross-encoder reranking, and answer generation with Groq.

`api/main.py` creates the FastAPI app, adds HTTP request logging middleware, and includes two routers. `src/utils/logging.py` configures logging. `scripts/run_ingestion.py` and `scripts/run_query.py` are empty.

`tests/test_chunking.py` loads Markdown files, prints the chunks, and calls `pipeline.chunk_embedding`. `tests/test_retrieval.py` calls `answer.generate_answer` and prints the response content. `generate_answer` calls `hybrid_search`.

## 2. Project Architecture / high level flow

**HTTP application**

- `api/main.py` creates `FastAPI(title="RAG system")`.
- HTTP middleware `log_request` reads header `X-Request-ID` or builds a request id, stores it in `request_id_var`, logs the method and URL, calls the next handler, and logs the status code and elapsed milliseconds.
- The app includes `RAGAnswerAPIRouter` from `api/routes/rag_answer.py`.
- The app includes `ingestion_router` from `api/routes/ingestion.py`.
- `GET /` returns `{"message":"app is running"}`.

**Logging**

- `src/utils/logging.py` defines `request_id_var`, `RequestIdFilter`, `JsonFormatter`, `setup_logging`, `get_logger`, and `log_timing`.
- `get_logger` calls `setup_logging` once, then returns `logging.getLogger(name)`.
- `setup_logging` writes to stdout and, when `LOG_DIR` is non-empty, to a rotating file named `app.log`.

**Document loading and chunking**

- `src/rag_app/ingestion/fileloader.py` defines `load_files(filepath)`.
- If `filepath` is a file and the path ends with `.md`, it chunks that file.
- If `filepath` is not a file, it walks the path and chunks files whose names end with `.md`.
- Each chunk receives metadata `source` (the file path) and `chunk_index`.
- `load_files` calls `chunk_maker` from `src.rag_app.ingestion.chunking`.
- `src/rag_app/ingestion/chunking.py` defines `chunk_maker(filepath)`. It reads the file as UTF-8, splits on Markdown headers, then splits those sections with `RecursiveCharacterTextSplitter`.

**Indexing**

- `src/rag_app/ingestion/pipeline.py` defines `create_collection`, `generate_point_id`, `chunk_embedding`, `delete_document`, and `create_payload_index`.
- `chunk_embedding` builds a `PointStruct` for each chunk, appends it to a batch, and calls `QdrantClient.upsert` when the batch reaches `batch_size` and again for any remaining points.
- `POST /ingestion/create_collection` calls `pipeline.create_collection` and then `pipeline.create_payload_index`, both with `os.getenv("QDRANT_COLLECTION_NAME")`.
- `POST /ingestion/load_files` calls `fileloader.load_files(filepath)` and then `pipeline.chunk_embedding(data)`.
- `POST /ingestion/delete_files` calls `pipeline.delete_document` with `QDRANT_COLLECTION_NAME` and `filepath`.
- `tests/test_chunking.py` also calls `fileloader.load_files` and `pipeline.chunk_embedding`. The calls to `pipeline.create_collection` and `pipeline.create_payload_index` in that file are comments.

**Query and answer**

- `src/rag_app/retrieval/hybrid_search.py` defines `hybrid_search(query, limit=5, retrieval=20)`.
- It builds a dense query vector and a sparse query vector, queries Qdrant using `QDRANT_COLLECTION_NAME`, then calls `reranker`.
- `src/rag_app/generation/answer.py` defines `filter_context_relative_response` and `generate_answer(query)`.
- `generate_answer` calls `hybrid_search(query)`, filters the reranked results, and sends a prompt to `ChatGroq`.
- `POST /ragapi/generate_answer` calls `answer.generate_answer(query)` and returns `{"answer": response}`.
- `tests/test_retrieval.py` calls `answer.generate_answer` at module level. It also defines `scrolldata`, and the module-level code does not call `scrolldata`.

## 3. Technologies and dependencies

`requirements.txt` lists these names, with no versions:

- fastapi
- uvicorn
- langgraph
- langchain
- python-dotenv
- pydantic
- qdrant-client
- langchain-groq
- langchain-ollama
- langchain-text-splitter
- fastembed
- ollama
- sentence_transformers
- pytest

Imports present in project source:

| Module | Imports used |
| --- | --- |
| `api/main.py` | `time`, `uuid`, `fastapi.FastAPI`, `api.routes.rag_answer.RAGAnswerAPIRouter`, `api.routes.ingestion.ingestion_router`, `src.utils.logging.get_logger`, `src.utils.logging.request_id_var` |
| `api/routes/ingestion.py` | `fastapi.APIRouter`, `src.rag_app.ingestion.pipeline`, `src.rag_app.ingestion.fileloader`, `dotenv.load_dotenv`, `src.utils.logging.get_logger`, `src.utils.logging.log_timing` |
| `api/routes/rag_answer.py` | `fastapi.FastAPI`, `fastapi.APIRouter`, `fastapi.HTTPException`, `src.rag_app.generation.answer`, `src.utils.logging.get_logger`, `src.utils.logging.log_timing` |
| `src/utils/logging.py` | `contextvars`, `json`, `logging`, `sys`, `time`, `contextlib.contextmanager`, `logging.handlers.RotatingFileHandler` |
| `src/rag_app/ingestion/chunking.py` | `langchain_text_splitters.RecursiveCharacterTextSplitter`, `langchain_text_splitters.MarkdownHeaderTextSplitter`, `src.utils.logging.get_logger`, `src.utils.logging.log_timing`, `fastapi.HTTPException` |
| `src/rag_app/ingestion/fileloader.py` | `langchain_core.documents.Document`, `src.rag_app.ingestion.chunking.chunk_maker`, `src.utils.logging.get_logger`, `src.utils.logging.log_timing`, `fastapi.HTTPException` |
| `src/rag_app/ingestion/pipeline.py` | `ollama`, `langchain_ollama.OllamaEmbeddings`, `fastembed.SparseTextEmbedding`, `qdrant_client.QdrantClient`, `qdrant_client.models`, `uuid`, `dotenv.load_dotenv`, `src.utils.logging.get_logger`, `src.utils.logging.log_timing`, `fastapi.HTTPException` |
| `src/rag_app/retrieval/hybrid_search.py` | `qdrant_client.QdrantClient`, `qdrant_client.models`, `dotenv.load_dotenv`, `ollama`, `fastembed.SparseTextEmbedding`, `src.rag_app.retrieval.reranker.reranker`, `src.utils.logging.get_logger`, `src.utils.logging.log_timing` |
| `src/rag_app/retrieval/reranker.py` | `sentence_transformers.CrossEncoder`, `src.utils.logging.get_logger`, `src.utils.logging.log_timing` |
| `src/rag_app/generation/answer.py` | `langchain_groq.ChatGroq`, `src.rag_app.retrieval.hybrid_search.hybrid_search`, `src.utils.logging.get_logger`, `src.utils.logging.log_timing`, `fastapi.HTTPException`, `dotenv.load_dotenv` |
| `tests/test_chunking.py` | `src.rag_app.ingestion.fileloader`, `src.rag_app.ingestion.pipeline`, `dotenv.load_dotenv` |
| `tests/test_retrieval.py` | `src.rag_app.generation.answer`, `qdrant_client.QdrantClient`, `qdrant_client.models`, `json`, `dotenv.load_dotenv` |

`log_timing` is imported by `ingestion.py`, `rag_answer.py`, `chunking.py`, `fileloader.py`, `pipeline.py`, `hybrid_search.py`, `reranker.py`, and `answer.py`. The only call shown in the repository is the example inside the docstring of `log_timing` in `src/utils/logging.py`.

`fastapi.FastAPI` is imported in `api/routes/rag_answer.py` and is not used in that file.

`uvicorn`, `langgraph`, `langchain`, `pydantic`, and `pytest` appear in `requirements.txt` and are not imported by the Python files in this repository.

There is no `pyproject.toml`.

## 4. Project Folder Structure

```text
RAGofDocuments/
├── .env
├── .gitignore
├── README.md
├── requirements.txt
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── routes/
│       ├── __init__.py
│       ├── ingestion.py
│       └── rag_answer.py
├── logs/
│   └── app.log
├── scripts/
│   ├── run_ingestion.py
│   └── run_query.py
├── src/
│   ├── rag_app/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── embedding/
│   │   │   ├── __init__.py
│   │   │   ├── dense.py
│   │   │   └── sparse.py
│   │   ├── generation/
│   │   │   ├── __init__.py
│   │   │   └── answer.py
│   │   ├── ingestion/
│   │   │   ├── __init__.py
│   │   │   ├── chunking.py
│   │   │   ├── fileloader.py
│   │   │   └── pipeline.py
│   │   ├── retrieval/
│   │   │   ├── __init__.py
│   │   │   ├── hybrid_search.py
│   │   │   └── reranker.py
│   │   └── vectorstore/
│   │       ├── __init__.py
│   │       ├── client.py
│   │       └── schema.py
│   └── utils/
│       ├── __init__.py
│       └── logging.py
└── tests/
    ├── __init__.py
    ├── test_chunking.py
    ├── test_generation.py
    └── test_retrieval.py
```

Empty files:

- `api/__init__.py`
- `api/routes/__init__.py`
- `scripts/run_ingestion.py`
- `scripts/run_query.py`
- `src/rag_app/__init__.py`
- `src/rag_app/config.py`
- `src/rag_app/embedding/__init__.py`
- `src/rag_app/embedding/dense.py`
- `src/rag_app/embedding/sparse.py`
- `src/rag_app/generation/__init__.py`
- `src/rag_app/ingestion/__init__.py`
- `src/rag_app/retrieval/__init__.py`
- `src/rag_app/vectorstore/__init__.py`
- `src/rag_app/vectorstore/client.py`
- `src/rag_app/vectorstore/schema.py`
- `src/utils/__init__.py`
- `tests/__init__.py`
- `tests/test_generation.py`

`logs/app.log` is present in the project directory.

## 5. Prerequisites

Not currently documented.

The implemented code constructs clients and models with these fixed values:

- `ollama.Client(host="localhost")` in `src/rag_app/ingestion/pipeline.py` and `src/rag_app/retrieval/hybrid_search.py`
- dense embedding model name `nomic-embed-text`
- sparse embedding model name `Qdrant/BM25`
- reranker model name `BAAI/bge-reranker-v2-m3`
- chat model name read from the `MODEL` environment variable

`ol_client` in `pipeline.py` is created and is not used. Dense embedding in that file uses `OllamaEmbeddings`.

## 6. Environment Variables required

Do not commit `.env`. `.gitignore` lists `.env`.

There is no `.env.example`.

Names read by `os.getenv` in source:

| Name | Where it is read | Default in code when unset |
| --- | --- | --- |
| `QDRANT_API_KEY` | `src/rag_app/ingestion/pipeline.py`, `src/rag_app/retrieval/hybrid_search.py`, `tests/test_retrieval.py` | none |
| `QDRANT_CLUSTER_ENDPOINT` | `src/rag_app/ingestion/pipeline.py`, `src/rag_app/retrieval/hybrid_search.py`, `tests/test_retrieval.py` | none |
| `QDRANT_COLLECTION_NAME` | `src/rag_app/ingestion/pipeline.py`, `api/routes/ingestion.py`, `src/rag_app/retrieval/hybrid_search.py`, `tests/test_retrieval.py` | none |
| `MODEL` | `src/rag_app/generation/answer.py` | none |
| `LOG_LEVEL` | `src/utils/logging.py` | `INFO` |
| `LOG_FORMAT` | `src/utils/logging.py` | `text` |
| `LOG_DIR` | `src/utils/logging.py` | `logs` |

`LOG_LEVEL` is converted with `.upper()`. The code comment in `logging.py` lists `DEBUG`, `INFO`, `WARNING`, and `ERROR`.

`LOG_FORMAT` is converted with `.lower()`. `setup_logging` uses `JsonFormatter` when the value is `json`. Any other value uses `logging.Formatter(TEXT_FORMAT)`.

`LOG_DIR` empty means `setup_logging` does not add a file handler. A non-empty value is created with `os.makedirs(log_dir, exist_ok=True)` and used as the folder for `app.log`.

`load_dotenv()` is called in `pipeline.py`, `hybrid_search.py`, `answer.py`, `api/routes/ingestion.py`, `tests/test_chunking.py`, and `tests/test_retrieval.py`.

Keys present in `.env` (names only):

- `QDRANT_API_KEY`
- `QDRANT_CLUSTER_ENDPOINT`
- `GROQ_API_KEY`
- `MODEL`
- `QDRANT_COLLECTION_NAME`
- `LOG_LEVEL`
- `LOG_FORMAT`
- `LOG_DIR`

`GROQ_API_KEY` is present in `.env` and is not referenced by name in the project source. `generate_answer` constructs `ChatGroq(model=model)` and does not pass an API key argument.

In `.env`, the key names `QDRANT_API_KEY`, `QDRANT_CLUSTER_ENDPOINT`, `GROQ_API_KEY`, `MODEL`, and `QDRANT_COLLECTION_NAME` each have a trailing space before `=`. `LOG_LEVEL`, `LOG_FORMAT`, and `LOG_DIR` do not.

Values are not listed here.

## 7. How to run the application

Not currently documented.

`api/main.py` defines the FastAPI application object `app`. It does not call a server. `scripts/run_ingestion.py` and `scripts/run_query.py` are empty. No Python file defines `if __name__ == "__main__"`. No host or port is set in the project source.

`tests/test_chunking.py` is a top-level script. On execution it:

1. Calls `load_dotenv()`.
2. Sets `path` to `C:\Users\Vinoth\Python\AI programs\Power Automate\power-automate-docs\articles\mobile`.
3. Calls `fileloader.load_files(path)`.
4. Prints each returned chunk.
5. Prints `file chunks created `.
6. Does not call `pipeline.create_collection`. That call is a comment: `pipeline.create_collection(os.getenv("QDRANT_COLLECTION_NAME"))`.
7. Calls `pipeline.chunk_embedding(data)`.
8. Does not call `pipeline.create_payload_index`. That call is a comment: `pipeline.create_payload_index(os.getenv("QDRANT_COLLECTION_NAME"))`.

`tests/test_retrieval.py` is a top-level script. On execution it:

1. Calls `load_dotenv()`.
2. Defines `scrolldata()` and does not call it.
3. Sets `query` to `how do I buy android phone`.
4. Calls `answer.generate_answer(query)`.
5. If the return value is a `str`, strips a leading `` ```json `` and a trailing `` ``` ``, then calls `json.loads`.
6. Otherwise uses the return value as `ans`.
7. Splits `ans.content` on the two-character sequence `\` and `n`, then prints each piece.

`scrolldata` in that file, when called, creates a `QdrantClient` from `QDRANT_API_KEY` and `QDRANT_CLUSTER_ENDPOINT`, then calls `scroll` on `QDRANT_COLLECTION_NAME` with `limit=10`, `with_vectors=True`, and `with_payload=True`. The scroll filter matches payload field `source` to `C:\Users\Vinoth\Python\AI programs\Power Automate\power-automate-docs\articles\mobile\manage-approvals.md`. It prints `points`.

## 8. RAG pipeline flow

**Chunking** (`chunk_maker`)

1. Log `Starting to chunk the file: {filepath}`.
2. Read the Markdown file with UTF-8 encoding.
3. Split with `MarkdownHeaderTextSplitter` on:
   - `#` mapped to metadata key `Header 1`
   - `##` mapped to metadata key `Header 2`
   - `###` mapped to metadata key `Header 3`
4. Split the resulting sections with `RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=200)`.
5. Log the file path and `len(file_chunk)`.
6. On exception, log it and raise `HTTPException(status_code=500, detail=str(e))`.

**Loading** (`load_files`)

1. Log `Loading files from '{filepath}'`.
2. Accept a file path or a directory path.
3. Process only paths or file names ending in `.md`.
4. Set `source` and `chunk_index` on each chunk.
5. Log the number of chunks and return the list.
6. On exception, log it and raise `HTTPException(status_code=500, detail=str(e))`.

**Collection creation** (`create_collection`)

1. If `qd_client.collection_exists(collection_name)` is false, log that the collection is being created and call `create_collection`.
2. Named dense vector `dense`: size `768`, distance `COSINE`.
3. Named sparse vector `sparse`: modifier `IDF`.
4. After creation, log that the collection was created.
5. If the collection already exists, log that it already exists.
6. On exception, log it and raise `HTTPException(status_code=500, detail=str(e))`.

The HTTP handler `POST /ingestion/create_collection` calls this function and then `create_payload_index`.

**Point identity** (`generate_point_id`)

- Logs the source, chunk index, and `f"{source}:{chunk_index}"`.
- ID string is `uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{chunk_index}")`.
- On exception, log it and raise `HTTPException(status_code=500, detail=str(e))`.

**Embedding and upsert** (`chunk_embedding`)

1. Log the chunk count and `batch_size`.
2. For each chunk, print `chunks started`.
3. Join non-empty metadata values `Header 1`, `Header 2`, and `Header 3` with newlines.
4. Embedding text is that header text, then two newlines, then `page_content`.
5. Dense vector: `OllamaEmbeddings(model="nomic-embed-text").embed_query(embedding_text)`.
6. Sparse vector: `SparseTextEmbedding(model_name="Qdrant/BM25")`, then `models.SparseVector` from `indices` and `values`.
7. Point id comes from `generate_point_id(file_chunk.metadata["source"], file_chunk.metadata["chunk_index"])`.
8. Build a `models.PointStruct` and append it to `points`.
9. Payload fields: `source`, `chunk_index`, `Header 1`, `Header 2`, `Header 3`, `text` (`page_content`).
10. `batch_size` default is `50`. When `len(points) >= batch_size`, the batch is upserted, the count and collection name are logged, and `points` is cleared.
11. After the loop, any remaining points are upserted and the remaining count is logged.
12. Upsert uses `collection_name=os.getenv("QDRANT_COLLECTION_NAME")` and `wait=True`.
13. The function returns `points`. A batch that triggered `points.clear()` is not included in that return value. The final remainder, if any, is returned after it has been upserted.
14. On exception, log it and raise `HTTPException(status_code=500, detail=str(e))`.

`POST /ingestion/load_files` calls `chunk_embedding` and does not use its return value.

**Deletion** (`delete_document`)

- Deletes points in the given collection whose payload field `source` matches the given `source`, with `wait=True`.
- Logs before and after the delete.
- On exception, log it and raise `HTTPException(status_code=500, detail=str(e))`.
- `POST /ingestion/delete_files` passes its `filepath` argument as that `source`.

**Payload index** (`create_payload_index`)

- Logs that a payload index is being created.
- Creates a payload index on field `source` with schema `KEYWORD`.
- On exception, log it and raise `HTTPException(status_code=500, detail=str(e))`.

**Hybrid search** (`hybrid_search`)

1. Signature is `hybrid_search(query, limit=5, retrieval=20)`. The docstring lists a `collection_name` parameter. The function signature does not include `collection_name`.
2. Dense query vector from `ollama.Client(host="localhost").embeddings(model="nomic-embed-text", prompt=query)`, using the `embedding` field.
3. Sparse query vector from `SparseTextEmbedding(model_name="Qdrant/BM25")`.
4. `QdrantClient` is created with `api_key=os.getenv("QDRANT_API_KEY")` and `url=os.getenv("QDRANT_CLUSTER_ENDPOINT")`.
5. `query_points` uses `collection_name=os.getenv("QDRANT_COLLECTION_NAME")` and two prefetches:
   - dense query, `using="dense"`, `limit=retrieval` (default `20`)
   - sparse query, `using="sparse"`, `limit=retrieval` (default `20`)
6. Fusion query is `RrfQuery` with `Rrf(k=60)`.
7. Result `limit` defaults to `5`. `with_payload=True`.
8. `results.points` are passed to `reranker(query, candidates, top_k=limit)`.

**Reranking** (`reranker`)

1. `CrossEncoder("BAAI/bge-reranker-v2-m3")`.
2. Pairs are `(query, candidate.payload["text"])`.
3. Scores come from `predict`.
4. Results are sorted by score descending and cut to `top_k` (default `5`).
5. Return value is a list of `(candidate, score)` pairs.

**Answer generation** (`generate_answer`)

1. Signature is `generate_answer(query)`. The docstring also lists `:param reranked_result:`. The function does not take `reranked_result`.
2. `model = os.getenv("MODEL")`.
3. `reranked_result = hybrid_search(query)`.
4. `filter_context_relative_response(reranked_result)` is called with defaults `margin=0.25`, `min_query=0.05`, `max_chunks=3`.
5. The prompt is an f-string. It tells the model to answer only from the provided context, not to hallucinate or invent an answer, to say it does not have information when it does not, and to provide the answer in JSON format. It then includes `Context:`, the `context_chunks` value, `Question:`, the query, and `Answer:`.
6. `context_chunks` is the list returned by `filter_context_relative_response`. It is inserted into the f-string directly.
7. `ChatGroq(model=model).invoke(prompt)` is returned.
8. On exception, log the query and exception, then raise `HTTPException(status_code=500, detail=str(e))`.

`filter_context_relative_response` returns `[]` when `not min_query or result_rank==[]` is true. Otherwise it keeps pairs whose score is within `margin` of `result_rank[0][1]`, reads `point.payload["text"]`, and returns at most `max_chunks` strings. On exception, it logs and raises `HTTPException(status_code=500, detail=str(e))`.

**Request logging** (`log_request`)

1. `request_id` is `request.headers.get("X-Request-ID")` when that header is present.
2. Otherwise `request_id` is `uuid.uuid5(uuid.NAMESPACE_URL, "Request").hex[:12]`.
3. The id is stored with `request_id_var.set`.
4. The start log line is `Request startedRequest ID: {request_id} - {request.method} {request.url}`.
5. Elapsed time is `(time.perf_counter() - start) * 1000`.
6. On success, the completion log includes `response.status_code` and the elapsed time, and the response is returned.
7. On exception, the log message includes `response.status_code` and the elapsed time, then the exception is re-raised.
8. `request_id_var.reset(token)` runs in `finally`.

## 9. API endpoints, if present

`api/main.py` sets the application title to `RAG system`. No request model is defined in the project. No authentication dependency is attached to the routes.

| Method | Path | Function | Declared parameters | Return value in source |
| --- | --- | --- | --- | --- |
| `GET` | `/` | `root` | none | `{"message":"app is running"}` |
| `POST` | `/ingestion/create_collection` | `create_collection` | none | `{"message": "Data ingestion successful"}` or `{"message": f"Error creating collection: {str(e)}"}` |
| `POST` | `/ingestion/load_files` | `load_files` | `filepath: str` | `{"message": "Data ingestion successful"}` or `{"message": f"Error loading files: {str(e)}"}` |
| `POST` | `/ingestion/delete_files` | `delete_files` | `filepath: str` | `{"message": "Data deletion successful"}` or `{"message": f"Error deleting files: {str(e)}"}` |
| `POST` | `/ragapi/generate_answer` | `generate_answer` | `query: str` | `{"answer": response}` |

`ingestion_router` uses `prefix="/ingestion"` and `tags=["ingestion"]`.

`RAGAnswerAPIRouter` uses `prefix="/ragapi"` and `tags=["RAG Answer API"]`. The `generate_answer` route function is `async`.

`POST /ingestion/create_collection` calls `pipeline.create_collection` and `pipeline.create_payload_index` with `os.getenv("QDRANT_COLLECTION_NAME")`. Exceptions are logged and returned in the message string above.

`POST /ingestion/load_files` calls `fileloader.load_files(filepath)` and `pipeline.chunk_embedding(data)`. On success it logs the filepath and collection name. Exceptions are logged and returned in the message string above.

`POST /ingestion/delete_files` calls `pipeline.delete_document(os.getenv("QDRANT_COLLECTION_NAME"), filepath)`. On success it logs the filepath and collection name. Exceptions are logged and returned in the message string above.

`POST /ragapi/generate_answer` logs the query, sets `response` to the return value of `answer.generate_answer(query)`, logs that response, and returns `{"answer": response}`. On exception it logs the query and raises `HTTPException(status_code=500, detail=str(e))`.

## 10. Installation steps

Not currently documented.

Dependencies are listed, without versions or an install command, in `requirements.txt`.

## 11. Qdrant configuration and collection details, if present

Client construction in `pipeline.py`, `hybrid_search.py`, and `scrolldata`:

- `QdrantClient(api_key=os.getenv("QDRANT_API_KEY"), url=os.getenv("QDRANT_CLUSTER_ENDPOINT"))`

`create_collection(collection_name)` creates a collection only when it does not already exist:

| Setting | Value in code |
| --- | --- |
| Dense vector name | `dense` |
| Dense vector size | `768` |
| Dense distance | `COSINE` |
| Sparse vector name | `sparse` |
| Sparse modifier | `IDF` |

The collection name used by `chunk_embedding` upsert, `hybrid_search` `query_points`, and the ingestion routes is `os.getenv("QDRANT_COLLECTION_NAME")`.

`tests/test_retrieval.py` `scrolldata` also uses `os.getenv("QDRANT_COLLECTION_NAME")`.

Point payload fields written by `chunk_embedding`:

- `source`
- `chunk_index`
- `Header 1`
- `Header 2`
- `Header 3`
- `text`

`create_payload_index` indexes payload field `source` as `KEYWORD`.

`delete_document` filters on payload field `source`.

`scrolldata` filters on payload field `source` with `models.MatchValue` and calls `scroll` with `limit=10`, `with_vectors=True`, and `with_payload=True`.

Query fusion in `hybrid_search`: reciprocal rank fusion with `k=60`, prefetch limit default `20`, final limit default `5`.

## 12. Embedding and retrieval approach, if present

**Indexing embeddings**

- Dense: `OllamaEmbeddings(model="nomic-embed-text")`, method `embed_query`.
- Sparse: `fastembed.SparseTextEmbedding(model_name="Qdrant/BM25")`.
- Text embedded for each chunk is the non-empty header metadata plus `page_content`, as described in section 8.
- Each embedded chunk is stored as a `PointStruct` with vectors `dense` and `sparse`.

**Query embeddings**

- Dense: `ollama.Client(host="localhost").embeddings(model="nomic-embed-text", prompt=query)`.
- Sparse: `SparseTextEmbedding(model_name="Qdrant/BM25")` on the query string.

**Retrieval**

- Qdrant `query_points` uses the collection name from `QDRANT_COLLECTION_NAME` and prefetches the named vectors `dense` and `sparse`.
- Fusion is `models.RrfQuery(rrf=models.Rrf(k=60))`.
- Candidates are reranked by `sentence_transformers.CrossEncoder("BAAI/bge-reranker-v2-m3")` using payload field `text`.
- `generate_answer` passes the query to `hybrid_search` and then filters the returned `(point, score)` pairs by score margin.

`src/rag_app/embedding/dense.py` and `src/rag_app/embedding/sparse.py` are empty. Embedding code used by the project is in `pipeline.py` and `hybrid_search.py`.

## 13. Example request/response, based only on the actual API

No sample HTTP request or captured HTTP response is stored in the repository. The route functions return these values:

`GET /`:

```json
{"message": "app is running"}
```

`POST /ingestion/create_collection` returns one of:

```json
{"message": "Data ingestion successful"}
```

```json
{"message": "Error creating collection: <exception text>"}
```

`POST /ingestion/load_files` returns one of:

```json
{"message": "Data ingestion successful"}
```

```json
{"message": "Error loading files: <exception text>"}
```

`POST /ingestion/delete_files` returns one of:

```json
{"message": "Data deletion successful"}
```

```json
{"message": "Error deleting files: <exception text>"}
```

`POST /ragapi/generate_answer` returns:

```json
{"answer": "<return value of answer.generate_answer>"}
```

The project does not contain a stored value for that answer. `answer.generate_answer` returns the object from `ChatGroq.invoke`. On exception the route raises `HTTPException` with status code `500` and `detail=str(e)`.

`tests/test_retrieval.py` uses this query string with `answer.generate_answer` directly, not through the HTTP route:

```text
how do I buy android phone
```

No captured response for that call is stored in the repository.

## 14. Testing instructions, if tests exist

Not currently documented.

`pytest` is listed in `requirements.txt`. No Python file imports `pytest`. No test command is defined in the repository.

Files under `tests/`:

| File | Contents |
| --- | --- |
| `tests/__init__.py` | Empty |
| `tests/test_chunking.py` | Script that calls `fileloader.load_files` on `C:\Users\Vinoth\Python\AI programs\Power Automate\power-automate-docs\articles\mobile`, prints each chunk, and calls `pipeline.chunk_embedding` |
| `tests/test_generation.py` | Empty |
| `tests/test_retrieval.py` | Script that defines `scrolldata` and, at module level, calls `answer.generate_answer` with `how do I buy android phone`, then prints pieces of `ans.content` |

No assertion is defined in the repository.

## 15. Security considerations

- `.gitignore` contains `.env` and `.idea`. It does not list `logs/`.
- Qdrant access uses `QDRANT_API_KEY` and `QDRANT_CLUSTER_ENDPOINT` from the environment after `load_dotenv()`.
- `MODEL` is read from the environment for `ChatGroq`.
- `GROQ_API_KEY` is stored in `.env`. Project source does not reference that name.
- The FastAPI routes do not declare authentication or authorization.
- `POST /ingestion/load_files` passes `filepath` to `fileloader.load_files`, which reads a file or walks a directory.
- `POST /ingestion/delete_files` passes `filepath` to `pipeline.delete_document` as the `source` filter value.
- Ingestion error responses include `str(e)` in the returned message.
- `POST /ragapi/generate_answer` logs the query and the generated response. On failure, `HTTPException` detail is `str(e)`.
- `log_request` logs `request.method` and `request.url`.
- `setup_logging` writes log records to stdout and to `logs/app.log` when `LOG_DIR` is the default or another non-empty directory. `app.log` uses `RotatingFileHandler` with `maxBytes=5_000_000`, `backupCount=3`, and `encoding="utf-8"`.
- `tests/test_chunking.py` contains an absolute local directory path.
- `tests/test_retrieval.py` contains an absolute local file path inside `scrolldata` and prints the generated answer content.
- `chunk_embedding` prints `chunks started` for each chunk. Point ids are written to the logger.

## 16. Known limitations

- `scripts/run_ingestion.py`, `scripts/run_query.py`, `config.py`, `embedding/dense.py`, `embedding/sparse.py`, `vectorstore/client.py`, and `vectorstore/schema.py` are empty.
- `requirements.txt` has no versions. `uvicorn`, `langgraph`, `langchain`, `pydantic`, and `pytest` are listed and are not imported.
- No `.env.example`, install command, or server start command is in the repository. `api/main.py` does not set a host or port.
- In `.env`, `QDRANT_API_KEY`, `QDRANT_CLUSTER_ENDPOINT`, `GROQ_API_KEY`, `MODEL`, and `QDRANT_COLLECTION_NAME` include a trailing space. `os.getenv` calls use names without that trailing space.
- `create_collection`, `delete_document`, and `create_payload_index` take a `collection_name` argument. `chunk_embedding` and `hybrid_search` use `QDRANT_COLLECTION_NAME` instead of a function argument.
- `hybrid_search` docstring lists `collection_name`. The function signature does not.
- `generate_answer` docstring lists `reranked_result`. The function signature is `generate_answer(query)`.
- `chunk_embedding` returns the points still held in `points` after upsert. Batches cleared at `batch_size` are not part of that return value. The load-files route does not use that return value.
- `ol_client` is created in `pipeline.py` and is not used.
- `fastapi.FastAPI` is imported in `api/routes/rag_answer.py` and is not used.
- `log_timing` is defined in `src/utils/logging.py` and is imported by other modules. Those modules do not call it.
- `load_files` only handles `.md` files.
- Ollama host, embedding model names, vector size, chunk size, chunk overlap, RRF `k`, and the reranker model name are fixed in code.
- When `X-Request-ID` is absent, `log_request` uses `uuid.uuid5(uuid.NAMESPACE_URL, "Request").hex[:12]`.
- The `except` branch of `log_request` formats `response.status_code`. `response` is assigned only when `call_next(request)` returns.
- Ingestion routes catch `Exception` and return a message dict. `generate_answer` and several ingestion functions raise `HTTPException` with status code `500`.
- `POST /ingestion/create_collection` returns `{"message": "Data ingestion successful"}` after calling collection creation and payload-index creation.
- `tests/test_chunking.py` hardcodes a local directory path, prints chunks, and calls `chunk_embedding`. The `create_collection` and `create_payload_index` calls in that file are comments. The file contains no assertions.
- `tests/test_retrieval.py` hardcodes a query and defines `scrolldata` without calling it. The file contains no assertions.
- `test_generation.py` contains no tests.
- `filter_context_relative_response` returns `[]` when `result_rank` is `[]`. With the default `min_query=0.05`, a non-empty `result_rank` is filtered by `margin` and `max_chunks`.
- The HTTP answer route returns the `ChatGroq.invoke` object inside `{"answer": response}`. `tests/test_retrieval.py` reads `.content` from that kind of object when the return value is not a `str`.
- `setup_logging` sets loggers named `httpx`, `httpcore`, `urllib3`, and `uvicorn.access` to `logging.WARNING`.

## 17. Future improvements

Not currently documented.

No TODO, FIXME, or planned-work notes are present in the project source.
