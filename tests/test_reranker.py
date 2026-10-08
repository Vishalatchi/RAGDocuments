from unittest.mock import MagicMock

import pytest

from src.rag_app.retrieval import reranker as reranker_module


class _Candidate:
    def __init__(self, text):
        self.payload = {"text": text}


def _patch_encoder(monkeypatch, scores):
    model = MagicMock()
    model.predict.return_value = scores
    encoder_cls = MagicMock(return_value=model)
    monkeypatch.setattr(reranker_module, "CrossEncoder", encoder_cls)
    return encoder_cls, model


def test_reranker_sorts_by_score_descending_and_limits_top_k(monkeypatch):
    first = _Candidate("alpha")
    second = _Candidate("beta")
    third = _Candidate("gamma")
    encoder_cls, model = _patch_encoder(monkeypatch, [0.2, 0.9, 0.4])

    result = reranker_module.reranker("widget", [first, second, third], top_k=2)

    encoder_cls.assert_called_once_with("BAAI/bge-reranker-v2-m3")
    model.predict.assert_called_once_with(
        [("widget", "alpha"), ("widget", "beta"), ("widget", "gamma")]
    )
    assert result == [(second, 0.9), (third, 0.4)]


def test_reranker_default_top_k_is_five(monkeypatch):
    candidates = [_Candidate(f"text {index}") for index in range(6)]
    _patch_encoder(monkeypatch, [0.1, 0.6, 0.2, 0.5, 0.4, 0.3])

    result = reranker_module.reranker("query", candidates)

    assert len(result) == 5
    assert result[0][0] is candidates[1]
    assert result[0][1] == 0.6


def test_reranker_returns_all_candidates_when_top_k_is_larger(monkeypatch):
    candidates = [_Candidate("a"), _Candidate("b")]
    _patch_encoder(monkeypatch, [0.1, 0.8])

    result = reranker_module.reranker("query", candidates, top_k=10)

    assert result == [(candidates[1], 0.8), (candidates[0], 0.1)]


def test_reranker_returns_empty_list_for_top_k_zero(monkeypatch):
    _patch_encoder(monkeypatch, [0.4, 0.9])

    assert reranker_module.reranker("query", [_Candidate("a"), _Candidate("b")], top_k=0) == []


def test_reranker_returns_empty_list_for_no_candidates(monkeypatch):
    _, model = _patch_encoder(monkeypatch, [])

    assert reranker_module.reranker("query", []) == []
    model.predict.assert_called_once_with([])


def test_reranker_keeps_original_order_for_equal_scores(monkeypatch):
    first = _Candidate("first")
    second = _Candidate("second")
    _patch_encoder(monkeypatch, [0.5, 0.5])

    result = reranker_module.reranker("query", [first, second], top_k=2)

    assert result == [(first, 0.5), (second, 0.5)]


def test_reranker_raises_key_error_when_text_payload_is_missing(monkeypatch):
    candidate = MagicMock()
    candidate.payload = {}
    _patch_encoder(monkeypatch, [])

    with pytest.raises(KeyError):
        reranker_module.reranker("query", [candidate])
