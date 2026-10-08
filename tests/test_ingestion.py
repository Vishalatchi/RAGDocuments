import sys
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def ingestion(monkeypatch):
    monkeypatch.setattr("qdrant_client.QdrantClient", MagicMock())
    monkeypatch.setattr("ollama.Client", MagicMock())
    monkeypatch.setattr("langchain_ollama.OllamaEmbeddings", MagicMock())
    monkeypatch.setattr("fastembed.SparseTextEmbedding", MagicMock())
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "unit-test-collection")
    sys.modules.pop("src.rag_app.ingestion.pipeline", None)
    sys.modules.pop("api.routes.ingestion", None)
    from api.routes import ingestion as ingestion_mod

    ingestion_mod.pipeline = MagicMock()
    ingestion_mod.fileloader = MagicMock()
    return ingestion_mod


@pytest.fixture
def client(ingestion):
    app = FastAPI()
    app.include_router(ingestion.ingestion_router)
    return TestClient(app)


def test_router_prefix_and_tag(ingestion):
    assert ingestion.ingestion_router.prefix == "/ingestion"
    assert ingestion.ingestion_router.tags == ["ingestion"]


def test_create_collection_creates_collection_and_payload_index(ingestion, client):
    response = client.post("/ingestion/create_collection")

    assert response.status_code == 200
    assert response.json() == {"message": "Data ingestion successful"}
    ingestion.pipeline.create_collection.assert_called_once_with("unit-test-collection")
    ingestion.pipeline.create_payload_index.assert_called_once_with("unit-test-collection")


def test_create_collection_passes_none_when_collection_name_is_unset(ingestion, client, monkeypatch):
    monkeypatch.delenv("QDRANT_COLLECTION_NAME", raising=False)

    response = client.post("/ingestion/create_collection")

    assert response.status_code == 200
    assert response.json() == {"message": "Data ingestion successful"}
    ingestion.pipeline.create_collection.assert_called_once_with(None)
    ingestion.pipeline.create_payload_index.assert_called_once_with(None)


def test_create_collection_returns_error_message_and_skips_index(ingestion, client):
    ingestion.pipeline.create_collection.side_effect = RuntimeError("create failed")

    response = client.post("/ingestion/create_collection")

    assert response.status_code == 200
    assert response.json() == {"message": "Error creating collection: create failed"}
    ingestion.pipeline.create_payload_index.assert_not_called()


def test_create_collection_returns_error_when_payload_index_fails(ingestion, client):
    ingestion.pipeline.create_payload_index.side_effect = RuntimeError("index failed")

    response = client.post("/ingestion/create_collection")

    assert response.status_code == 200
    assert response.json() == {"message": "Error creating collection: index failed"}
    ingestion.pipeline.create_collection.assert_called_once_with("unit-test-collection")


def test_load_files_loads_path_and_embeds_returned_chunks(ingestion, client):
    chunks = ["chunk-a", "chunk-b"]
    ingestion.fileloader.load_files.return_value = chunks

    response = client.post("/ingestion/load_files", params={"filepath": r"C:\docs\guide.md"})

    assert response.status_code == 200
    assert response.json() == {"message": "Data ingestion successful"}
    ingestion.fileloader.load_files.assert_called_once_with(r"C:\docs\guide.md")
    ingestion.pipeline.chunk_embedding.assert_called_once_with(chunks)


def test_load_files_embeds_empty_chunk_list(ingestion, client):
    ingestion.fileloader.load_files.return_value = []

    response = client.post("/ingestion/load_files", params={"filepath": r"C:\empty"})

    assert response.status_code == 200
    assert response.json() == {"message": "Data ingestion successful"}
    ingestion.pipeline.chunk_embedding.assert_called_once_with([])


def test_load_files_returns_error_and_does_not_embed_when_load_fails(ingestion, client):
    ingestion.fileloader.load_files.side_effect = RuntimeError("read failed")

    response = client.post("/ingestion/load_files", params={"filepath": "missing.md"})

    assert response.status_code == 200
    assert response.json() == {"message": "Error loading files: read failed"}
    ingestion.pipeline.chunk_embedding.assert_not_called()


def test_load_files_returns_error_when_embedding_fails(ingestion, client):
    ingestion.fileloader.load_files.return_value = ["chunk"]
    ingestion.pipeline.chunk_embedding.side_effect = RuntimeError("embed failed")

    response = client.post("/ingestion/load_files", params={"filepath": "guide.md"})

    assert response.status_code == 200
    assert response.json() == {"message": "Error loading files: embed failed"}


def test_load_files_requires_filepath(client):
    response = client.post("/ingestion/load_files")

    assert response.status_code == 422


def test_delete_files_deletes_source_from_collection(ingestion, client):
    response = client.post("/ingestion/delete_files", params={"filepath": r"C:\docs\guide.md"})

    assert response.status_code == 200
    assert response.json() == {"message": "Data deletion successful"}
    ingestion.pipeline.delete_document.assert_called_once_with(
        "unit-test-collection", r"C:\docs\guide.md"
    )


def test_delete_files_passes_none_collection_when_unset(ingestion, client, monkeypatch):
    monkeypatch.delenv("QDRANT_COLLECTION_NAME", raising=False)

    response = client.post("/ingestion/delete_files", params={"filepath": "guide.md"})

    assert response.status_code == 200
    assert response.json() == {"message": "Data deletion successful"}
    ingestion.pipeline.delete_document.assert_called_once_with(None, "guide.md")


def test_delete_files_returns_error_message(ingestion, client):
    ingestion.pipeline.delete_document.side_effect = RuntimeError("delete failed")

    response = client.post("/ingestion/delete_files", params={"filepath": "guide.md"})

    assert response.status_code == 200
    assert response.json() == {"message": "Error deleting files: delete failed"}


def test_delete_files_requires_filepath(client):
    response = client.post("/ingestion/delete_files")

    assert response.status_code == 422
