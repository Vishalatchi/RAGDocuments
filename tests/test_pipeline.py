import sys
import uuid
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from langchain_core.documents import Document
from qdrant_client import models


@pytest.fixture
def pipeline(monkeypatch):
    mock_client = MagicMock()
    dense = MagicMock()
    dense.embed_query.return_value = [0.1, 0.2, 0.3]
    sparse_vector = MagicMock()
    sparse_vector.indices.tolist.return_value = [1, 4]
    sparse_vector.values.tolist.return_value = [0.6, 0.4]
    sparse_model = MagicMock()
    sparse_model.embed.side_effect = lambda text: iter([sparse_vector])

    monkeypatch.setattr("qdrant_client.QdrantClient", MagicMock(return_value=mock_client))
    monkeypatch.setattr("ollama.Client", MagicMock())
    monkeypatch.setattr("langchain_ollama.OllamaEmbeddings", MagicMock(return_value=dense))
    monkeypatch.setattr("fastembed.SparseTextEmbedding", MagicMock(return_value=sparse_model))
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "unit-test-collection")
    monkeypatch.setenv("QDRANT_API_KEY", "unit-test-key")
    monkeypatch.setenv("QDRANT_CLUSTER_ENDPOINT", "http://qdrant.example")

    sys.modules.pop("src.rag_app.ingestion.pipeline", None)
    import src.rag_app.ingestion.pipeline as pipeline_mod

    pipeline_mod.qd_client = mock_client
    pipeline_mod.dense_embed = dense
    pipeline_mod.sparse_embed = sparse_model
    return pipeline_mod


def _document(text, source="guide.md", chunk_index=0, header_1="Guide", header_2=None, header_3=None):
    return Document(
        page_content=text,
        metadata={
            "source": source,
            "chunk_index": chunk_index,
            "Header 1": header_1,
            "Header 2": header_2,
            "Header 3": header_3,
        },
    )


def _capture_upserts(pipeline_mod):
    saved = []

    def upsert(*args, **kwargs):
        saved.append(list(kwargs["points"]))

    pipeline_mod.qd_client.upsert.side_effect = upsert
    return saved


def test_generate_point_id_uses_uuid5_of_source_and_chunk_index(pipeline):
    point_id = pipeline.generate_point_id("guide.md", 2)

    assert point_id == str(uuid.uuid5(uuid.NAMESPACE_URL, "guide.md:2"))
    assert pipeline.generate_point_id("guide.md", 2) == point_id
    assert pipeline.generate_point_id("guide.md", 3) != point_id


def test_generate_point_id_raises_http_exception(pipeline, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("uuid failed")

    monkeypatch.setattr(pipeline.uuid, "uuid5", fail)

    with pytest.raises(HTTPException) as exc_info:
        pipeline.generate_point_id("guide.md", 0)

    assert exc_info.value.status_code == 500
    assert "uuid failed" in exc_info.value.detail


def test_create_collection_creates_dense_and_sparse_vectors_when_missing(pipeline):
    pipeline.qd_client.collection_exists.return_value = False

    assert pipeline.create_collection("docs") is None

    kwargs = pipeline.qd_client.create_collection.call_args.kwargs
    assert kwargs["collection_name"] == "docs"
    assert kwargs["vectors_config"]["dense"].size == 768
    assert kwargs["vectors_config"]["dense"].distance == models.Distance.COSINE
    assert kwargs["sparse_vectors_config"]["sparse"].modifier == models.Modifier.IDF


def test_create_collection_does_not_create_existing_collection(pipeline):
    pipeline.qd_client.collection_exists.return_value = True

    pipeline.create_collection("docs")

    pipeline.qd_client.create_collection.assert_not_called()


def test_create_collection_raises_http_exception(pipeline):
    pipeline.qd_client.collection_exists.side_effect = RuntimeError("qdrant down")

    with pytest.raises(HTTPException) as exc_info:
        pipeline.create_collection("docs")

    assert exc_info.value.status_code == 500
    assert "qdrant down" in exc_info.value.detail


def test_chunk_embedding_builds_point_from_headers_and_text(pipeline):
    saved = _capture_upserts(pipeline)
    chunk = _document("Place the widget.", header_1="Guide", header_2=None, header_3="Usage")

    returned = pipeline.chunk_embedding([chunk], batch_size=50)

    assert len(saved) == 1
    point = saved[0][0]
    assert returned == saved[0]
    assert point.id == str(uuid.uuid5(uuid.NAMESPACE_URL, "guide.md:0"))
    assert point.vector["dense"] == [0.1, 0.2, 0.3]
    assert point.vector["sparse"].indices == [1, 4]
    assert point.vector["sparse"].values == [0.6, 0.4]
    assert point.payload == {
        "source": "guide.md",
        "chunk_index": 0,
        "Header 1": "Guide",
        "Header 2": None,
        "Header 3": "Usage",
        "text": "Place the widget.",
    }
    pipeline.dense_embed.embed_query.assert_called_once_with("Guide\nUsage\n\nPlace the widget.")
    pipeline.qd_client.upsert.assert_called_once()
    assert pipeline.qd_client.upsert.call_args.kwargs["collection_name"] == "unit-test-collection"
    assert pipeline.qd_client.upsert.call_args.kwargs["wait"] is True


def test_chunk_embedding_uses_page_content_when_headers_are_missing(pipeline):
    saved = _capture_upserts(pipeline)
    chunk = Document(page_content="Only text.", metadata={"source": "a.md", "chunk_index": 1})

    pipeline.chunk_embedding([chunk], batch_size=50)

    pipeline.dense_embed.embed_query.assert_called_once_with("\n\nOnly text.")
    assert saved[0][0].payload["Header 1"] is None
    assert saved[0][0].payload["text"] == "Only text."


def test_chunk_embedding_upserts_full_batches_and_returns_remainder(pipeline):
    saved = _capture_upserts(pipeline)
    chunks = [_document(f"text {index}", chunk_index=index) for index in range(5)]

    returned = pipeline.chunk_embedding(chunks, batch_size=2)

    assert [len(batch) for batch in saved] == [2, 2, 1]
    assert len(returned) == 1
    assert returned[0].payload["text"] == "text 4"
    assert all(
        call.kwargs["collection_name"] == "unit-test-collection" and call.kwargs["wait"] is True
        for call in pipeline.qd_client.upsert.call_args_list
    )


def test_chunk_embedding_returns_empty_list_when_count_matches_batch_size(pipeline):
    saved = _capture_upserts(pipeline)
    chunks = [_document(f"text {index}", chunk_index=index) for index in range(2)]

    returned = pipeline.chunk_embedding(chunks, batch_size=2)

    assert len(saved) == 1
    assert len(saved[0]) == 2
    assert returned == []


def test_chunk_embedding_empty_input_does_not_upsert(pipeline):
    returned = pipeline.chunk_embedding([], batch_size=50)

    assert returned == []
    pipeline.qd_client.upsert.assert_not_called()


def test_chunk_embedding_raises_http_exception_without_source(pipeline):
    chunk = Document(page_content="text", metadata={"chunk_index": 0})

    with pytest.raises(HTTPException) as exc_info:
        pipeline.chunk_embedding([chunk])

    assert exc_info.value.status_code == 500
    pipeline.qd_client.upsert.assert_not_called()


def test_chunk_embedding_raises_http_exception_when_sparse_embed_is_empty(pipeline):
    pipeline.sparse_embed.embed.side_effect = lambda text: iter([])

    with pytest.raises(HTTPException) as exc_info:
        pipeline.chunk_embedding([_document("text")])

    assert exc_info.value.status_code == 500


def test_delete_document_filters_on_source(pipeline):
    pipeline.delete_document("docs", r"C:\docs\guide.md")

    kwargs = pipeline.qd_client.delete.call_args.kwargs
    assert kwargs["collection_name"] == "docs"
    assert kwargs["wait"] is True
    condition = kwargs["points_selector"].must[0]
    assert condition.key == "source"
    assert condition.match.value == r"C:\docs\guide.md"


def test_delete_document_raises_http_exception(pipeline):
    pipeline.qd_client.delete.side_effect = RuntimeError("delete failed")

    with pytest.raises(HTTPException) as exc_info:
        pipeline.delete_document("docs", "guide.md")

    assert exc_info.value.status_code == 500
    assert "delete failed" in exc_info.value.detail


def test_create_payload_index_uses_keyword_schema_for_source(pipeline):
    pipeline.create_payload_index("docs")

    kwargs = pipeline.qd_client.create_payload_index.call_args.kwargs
    assert kwargs["collection_name"] == "docs"
    assert kwargs["field_name"] == "source"
    assert kwargs["field_schema"] == models.PayloadSchemaType.KEYWORD


def test_create_payload_index_raises_http_exception(pipeline):
    pipeline.qd_client.create_payload_index.side_effect = RuntimeError("index failed")

    with pytest.raises(HTTPException) as exc_info:
        pipeline.create_payload_index("docs")

    assert exc_info.value.status_code == 500
    assert "index failed" in exc_info.value.detail
