from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import rag_answer


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(rag_answer.RAGAnswerAPIRouter)
    return TestClient(app)


def test_router_prefix_and_tag():
    assert rag_answer.RAGAnswerAPIRouter.prefix == "/ragapi"
    assert rag_answer.RAGAnswerAPIRouter.tags == ["RAG Answer API"]


def test_generate_answer_returns_model_response(client, monkeypatch):
    generate = MagicMock(return_value="from the documents")
    monkeypatch.setattr(rag_answer.answer, "generate_answer", generate)

    response = client.post("/ragapi/generate_answer", params={"query": "where is the widget"})

    assert response.status_code == 200
    assert response.json() == {"answer": "from the documents"}
    generate.assert_called_once_with("where is the widget")


def test_generate_answer_returns_non_string_response(client, monkeypatch):
    generate = MagicMock(return_value={"text": "json answer"})
    monkeypatch.setattr(rag_answer.answer, "generate_answer", generate)

    response = client.post("/ragapi/generate_answer", params={"query": "query"})

    assert response.status_code == 200
    assert response.json() == {"answer": {"text": "json answer"}}


def test_generate_answer_passes_empty_query(client, monkeypatch):
    generate = MagicMock(return_value="")
    monkeypatch.setattr(rag_answer.answer, "generate_answer", generate)

    response = client.post("/ragapi/generate_answer", params={"query": ""})

    assert response.status_code == 200
    assert response.json() == {"answer": ""}
    generate.assert_called_once_with("")


def test_generate_answer_requires_query(client):
    response = client.post("/ragapi/generate_answer")

    assert response.status_code == 422


def test_generate_answer_returns_500_when_generation_fails(client, monkeypatch):
    monkeypatch.setattr(
        rag_answer.answer,
        "generate_answer",
        MagicMock(side_effect=RuntimeError("search failed")),
    )

    response = client.post("/ragapi/generate_answer", params={"query": "where is the widget"})

    assert response.status_code == 500
    assert response.json() == {"detail": "search failed"}
