from unittest.mock import MagicMock
import pytest
from fastapi import HTTPException
from src.rag_app.generation import answer


class _Point:
    def __init__(self, text):
        self.payload = {"text": text}


def test_filter_returns_empty_list_for_empty_results():
    assert answer.filter_context_relative_response([]) == []


def test_filter_returns_empty_list_when_min_query_is_falsy():
    ranked = [(_Point("kept"), 0.9)]

    assert answer.filter_context_relative_response(ranked, min_query=0) == []
    assert answer.filter_context_relative_response(ranked, min_query=None) == []


def test_filter_uses_first_score_as_margin_reference():
    ranked = [
        (_Point("first"), 0.50),
        (_Point("higher"), 0.90),
        (_Point("lower"), 0.10),
    ]

    assert answer.filter_context_relative_response(ranked, margin=0.25, max_chunks=3) == [
        "first",
        "higher",
    ]


def test_filter_stops_at_max_chunks():
    ranked = [
        (_Point("one"), 1.0),
        (_Point("two"), 0.9),
        (_Point("three"), 0.8),
        (_Point("four"), 0.8),
    ]

    assert answer.filter_context_relative_response(ranked, margin=0.25, max_chunks=2) == [
        "one",
        "two",
    ]


def test_filter_with_max_chunks_zero_returns_empty_list():
    ranked = [(_Point("one"), 1.0)]

    assert answer.filter_context_relative_response(ranked, max_chunks=0) == []


def test_filter_keeps_only_scores_within_zero_margin():
    ranked = [
        (_Point("base"), 0.4),
        (_Point("lower"), 0.3),
        (_Point("equal"), 0.4),
    ]

    assert answer.filter_context_relative_response(ranked, margin=0, max_chunks=3) == [
        "base",
        "equal",
    ]


def test_filter_raises_http_exception_when_text_is_missing():
    point = MagicMock()
    point.payload = {}

    with pytest.raises(HTTPException) as exc_info:
        answer.filter_context_relative_response([(point, 1.0)])

    assert exc_info.value.status_code == 500


def test_filter_raises_http_exception_for_none_results():
    with pytest.raises(HTTPException) as exc_info:
        answer.filter_context_relative_response(None)

    assert exc_info.value.status_code == 500


def test_generate_answer_sends_filtered_context_and_query_to_chat_model(monkeypatch):
    monkeypatch.setenv("MODEL", "unit-test-model")
    monkeypatch.setattr(
        answer,
        "hybrid_search",
        MagicMock(return_value=[(_Point("from the document"), 1.0)]),
    )
    chat = MagicMock()
    chat.invoke.return_value = "model-response"
    chat_cls = MagicMock(return_value=chat)
    monkeypatch.setattr(answer, "ChatGroq", chat_cls)

    result = answer.generate_answer("where is the widget")

    assert result == "model-response"
    answer.hybrid_search.assert_called_once_with("where is the widget")
    chat_cls.assert_called_once_with(model="unit-test-model")
    prompt = chat.invoke.call_args.args[0]
    assert "from the document" in prompt
    assert "where is the widget" in prompt
    assert "do not hallucinate and do not invent answer" in prompt
    assert "provide answer in json format" in prompt
    assert "Context:" in prompt
    assert "Question:" in prompt
    assert "Answer:" in prompt


def test_generate_answer_inserts_empty_context_list(monkeypatch):
    monkeypatch.setenv("MODEL", "unit-test-model")
    monkeypatch.setattr(answer, "hybrid_search", MagicMock(return_value=[]))
    chat = MagicMock()
    chat.invoke.return_value = "no-context"
    monkeypatch.setattr(answer, "ChatGroq", MagicMock(return_value=chat))

    assert answer.generate_answer("unknown") == "no-context"
    prompt = chat.invoke.call_args.args[0]
    assert "[]" in prompt
    assert "unknown" in prompt


def test_generate_answer_uses_model_from_environment_when_unset(monkeypatch):
    monkeypatch.delenv("MODEL", raising=False)
    monkeypatch.setattr(answer, "hybrid_search", MagicMock(return_value=[]))
    chat_cls = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(answer, "ChatGroq", chat_cls)

    answer.generate_answer("query")

    chat_cls.assert_called_once_with(model=None)


def test_generate_answer_raises_http_exception_when_search_fails(monkeypatch):
    monkeypatch.setenv("MODEL", "unit-test-model")
    monkeypatch.setattr(answer, "hybrid_search", MagicMock(side_effect=RuntimeError("search failed")))

    with pytest.raises(HTTPException) as exc_info:
        answer.generate_answer("query")

    assert exc_info.value.status_code == 500
    assert "search failed" in exc_info.value.detail
