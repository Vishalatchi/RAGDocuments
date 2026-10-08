from unittest.mock import MagicMock

import pytest
from qdrant_client import models

from src.rag_app.retrieval import hybrid_search


def _patch_clients(monkeypatch, embedding=None, sparse_items=None, points=None):
    if embedding is None:
        embedding = [0.1, 0.2]
    if sparse_items is None:
        sparse_item = MagicMock()
        sparse_item.indices.tolist.return_value = [3]
        sparse_item.values.tolist.return_value = [1.5]
        sparse_items = [sparse_item]
    if points is None:
        points = [MagicMock(name="point")]

    ol_client = MagicMock()
    ol_client.embeddings.return_value = {"embedding": embedding}
    client_cls = MagicMock(return_value=ol_client)
    monkeypatch.setattr(hybrid_search.ollama, "Client", client_cls)

    sparse_model = MagicMock()
    sparse_model.embed.return_value = sparse_items
    sparse_cls = MagicMock(return_value=sparse_model)
    monkeypatch.setattr(hybrid_search, "SparseTextEmbedding", sparse_cls)

    qd_client = MagicMock()
    qd_client.query_points.return_value = MagicMock(points=points)
    qdrant_cls = MagicMock(return_value=qd_client)
    monkeypatch.setattr(hybrid_search, "QdrantClient", qdrant_cls)

    reranked = [(points[0], 0.9)] if points else []
    reranker = MagicMock(return_value=reranked)
    monkeypatch.setattr(hybrid_search, "reranker", reranker)

    monkeypatch.setenv("QDRANT_API_KEY", "unit-test-key")
    monkeypatch.setenv("QDRANT_CLUSTER_ENDPOINT", "http://qdrant.example")
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "unit-test-collection")
    return client_cls, ol_client, sparse_cls, qdrant_cls, qd_client, reranker, points


def test_hybrid_search_queries_dense_and_sparse_vectors_then_reranks(monkeypatch):
    client_cls, ol_client, sparse_cls, qdrant_cls, qd_client, reranker, points = _patch_clients(
        monkeypatch
    )

    result = hybrid_search.hybrid_search("place a widget", limit=2, retrieval=7)

    client_cls.assert_called_once_with(host="localhost")
    ol_client.embeddings.assert_called_once_with(model="nomic-embed-text", prompt="place a widget")
    sparse_cls.assert_called_once_with(model_name="Qdrant/BM25")
    qdrant_cls.assert_called_once_with(api_key="unit-test-key", url="http://qdrant.example")

    kwargs = qd_client.query_points.call_args.kwargs
    assert kwargs["collection_name"] == "unit-test-collection"
    assert kwargs["limit"] == 2
    assert kwargs["with_payload"] is True
    assert kwargs["query"].rrf.k == 60
    assert kwargs["prefetch"][0].using == "dense"
    assert kwargs["prefetch"][0].query == [0.1, 0.2]
    assert kwargs["prefetch"][0].limit == 7
    assert kwargs["prefetch"][1].using == "sparse"
    assert kwargs["prefetch"][1].limit == 7
    assert isinstance(kwargs["prefetch"][1].query, models.SparseVector)
    assert kwargs["prefetch"][1].query.indices == [3]
    assert kwargs["prefetch"][1].query.values == [1.5]
    reranker.assert_called_once_with("place a widget", points, top_k=2)
    assert result == reranker.return_value


def test_hybrid_search_uses_default_limits(monkeypatch):
    _, _, _, _, qd_client, reranker, points = _patch_clients(monkeypatch)

    hybrid_search.hybrid_search("query")

    kwargs = qd_client.query_points.call_args.kwargs
    assert kwargs["limit"] == 5
    assert kwargs["prefetch"][0].limit == 20
    assert kwargs["prefetch"][1].limit == 20
    reranker.assert_called_once_with("query", points, top_k=5)


def test_hybrid_search_passes_empty_candidate_list_to_reranker(monkeypatch):
    _, _, _, _, _, reranker, _ = _patch_clients(monkeypatch, points=[])
    reranker.return_value = []

    assert hybrid_search.hybrid_search("query") == []
    reranker.assert_called_once_with("query", [], top_k=5)


def test_hybrid_search_raises_key_error_when_embedding_is_missing(monkeypatch):
    _, ol_client, _, _, _, _, _ = _patch_clients(monkeypatch)
    ol_client.embeddings.return_value = {}

    with pytest.raises(KeyError):
        hybrid_search.hybrid_search("query")


def test_hybrid_search_raises_index_error_when_sparse_embed_is_empty(monkeypatch):
    _patch_clients(monkeypatch, sparse_items=[])

    with pytest.raises(IndexError):
        hybrid_search.hybrid_search("query")
