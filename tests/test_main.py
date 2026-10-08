import asyncio
import logging
import sys
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


class _Headers(dict):
    def get(self, key, default=None):
        return super().get(key, default)


class _Request:
    def __init__(self, method, url, request_id=None, include_header=True):
        if include_header:
            self.headers = _Headers({"X-Request-ID": request_id})
        else:
            self.headers = _Headers()
        self.method = method
        self.url = url


class _Response:
    def __init__(self, status_code):
        self.status_code = status_code


@pytest.fixture
def main(monkeypatch):
    monkeypatch.setattr("qdrant_client.QdrantClient", MagicMock())
    monkeypatch.setattr("ollama.Client", MagicMock())
    monkeypatch.setattr("langchain_ollama.OllamaEmbeddings", MagicMock())
    monkeypatch.setattr("fastembed.SparseTextEmbedding", MagicMock())
    sys.modules.pop("src.rag_app.ingestion.pipeline", None)
    sys.modules.pop("api.routes.ingestion", None)
    sys.modules.pop("api.main", None)
    from api import main as main_mod

    return main_mod


@pytest.fixture
def client(main):
    return TestClient(main.app)


def _messages(caplog):
    return [record.message for record in caplog.records if record.name == "api.main"]


def _request_id_from_start_log(message):
    prefix = "Request startedRequest ID: "
    assert message.startswith(prefix)
    return message[len(prefix):].split(" - ", 1)[0]


def test_app_title_is_rag_system(main):
    assert main.app.title == "RAG system"


def test_app_includes_ingestion_and_answer_routers(main):
    included = [
        route.original_router
        for route in main.app.routes
        if hasattr(route, "original_router")
    ]

    assert main.ingestion_router in included
    assert main.RAGAnswerAPIRouter in included


def test_root_returns_running_message(main):
    assert main.root() == {"message": "app is running"}


def test_get_root_returns_running_message(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "app is running"}


def test_log_request_uses_request_id_header_and_logs_success(main, caplog):
    caplog.set_level(logging.INFO, logger="api.main")
    request = _Request("GET", "http://app.example/", request_id="req-123")
    response = _Response(200)

    async def call_next(received):
        assert received is request
        assert main.request_id_var.get() == "req-123"
        return response

    result = asyncio.run(main.log_request(request, call_next))

    assert result is response
    assert _messages(caplog)[0] == "Request startedRequest ID: req-123 - GET http://app.example/"
    completed = _messages(caplog)[1]
    assert completed.startswith("Request completed. ID: req-123 - response status code : 200 Time taken: ")
    assert completed.endswith(" ms")
    assert main.request_id_var.get() == "-"


def test_log_request_builds_new_fallback_id_when_header_is_missing(main, caplog):
    caplog.set_level(logging.INFO, logger="api.main")
    request = _Request("GET", "http://app.example/", include_header=False)
    seen = []

    async def call_next(received):
        seen.append(main.request_id_var.get())
        return _Response(200)

    asyncio.run(main.log_request(request, call_next))
    asyncio.run(main.log_request(request, call_next))

    assert len(seen[0]) == 12
    assert seen[0] != seen[1]
    assert int(seen[0], 16) >= 0
    assert _messages(caplog)[0] == f"Request startedRequest ID: {seen[0]} - GET http://app.example/"
    assert main.request_id_var.get() == "-"


def test_log_request_builds_fallback_id_when_header_is_empty(main, caplog):
    caplog.set_level(logging.INFO, logger="api.main")
    request = _Request("GET", "http://app.example/", request_id="")
    seen = []

    async def call_next(received):
        seen.append(main.request_id_var.get())
        return _Response(200)

    asyncio.run(main.log_request(request, call_next))

    request_id = seen[0]
    assert request_id != ""
    assert len(request_id) == 12
    assert int(request_id, 16) >= 0
    assert _request_id_from_start_log(_messages(caplog)[0]) == request_id


def test_log_request_logs_error_and_reraises_when_call_next_raises(main, caplog):
    caplog.set_level(logging.INFO, logger="api.main")
    request = _Request("GET", "http://app.example/crash", request_id="req-crash")

    async def call_next(received):
        raise RuntimeError("route failed")

    with pytest.raises(RuntimeError, match="route failed"):
        asyncio.run(main.log_request(request, call_next))

    messages = _messages(caplog)
    assert messages[0] == "Request startedRequest ID: req-crash - GET http://app.example/crash"
    assert messages[1].startswith("Request crashed. ID: req-crash - Error: route failed elapsed time: ")
    assert messages[1].endswith(" ms")
    assert main.request_id_var.get() == "-"
