from pathlib import Path

import pytest
from fastapi import HTTPException

from src.rag_app.ingestion.chunking import chunk_maker

WIDGETS_MD = Path(__file__).parent / "test_file_widgets.md"


def test_chunk_maker_splits_project_markdown_on_headers():
    chunks = chunk_maker(str(WIDGETS_MD))

    assert isinstance(chunks, list)
    assert chunks
    assert any(chunk.metadata.get("Header 1") == "Create widgets for flows" for chunk in chunks)
    assert any(chunk.metadata.get("Header 2") == "Create a widget on an iOS device" for chunk in chunks)
    assert any("Long-press the home screen" in chunk.page_content for chunk in chunks)
    assert all(chunk.page_content for chunk in chunks)


def test_chunk_maker_keeps_short_section_in_one_chunk(tmp_path):
    path = tmp_path / "short.md"
    path.write_text("# Title\n\nA short paragraph.\n", encoding="utf-8")

    chunks = chunk_maker(str(path))

    assert len(chunks) == 1
    assert chunks[0].metadata.get("Header 1") == "Title"
    assert "A short paragraph." in chunks[0].page_content


def test_chunk_maker_records_nested_headers(tmp_path):
    path = tmp_path / "nested.md"
    path.write_text(
        "# Guide\n\n## Install\n\nInstall the app.\n\n### Usage\n\nRun the widget.\n",
        encoding="utf-8",
    )

    chunks = chunk_maker(str(path))
    usage = [chunk for chunk in chunks if chunk.metadata.get("Header 3") == "Usage"]

    assert usage
    assert usage[0].metadata.get("Header 1") == "Guide"
    assert usage[0].metadata.get("Header 2") == "Install"
    assert "Run the widget." in usage[0].page_content


def test_chunk_maker_splits_text_longer_than_chunk_size(tmp_path):
    path = tmp_path / "long.md"
    path.write_text("# Long\n\n" + ("word " * 800), encoding="utf-8")

    chunks = chunk_maker(str(path))

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 750 for chunk in chunks)
    assert all(chunk.metadata.get("Header 1") == "Long" for chunk in chunks)
    combined = " ".join(chunk.page_content for chunk in chunks)
    assert "word" in combined


def test_chunk_maker_reads_utf8_text(tmp_path):
    path = tmp_path / "unicode.md"
    path.write_text("# Título\n\nविजेट\n", encoding="utf-8")

    chunks = chunk_maker(str(path))

    assert chunks[0].metadata.get("Header 1") == "Título"
    assert "विजेट" in chunks[0].page_content


def test_chunk_maker_returns_empty_list_for_empty_file(tmp_path):
    path = tmp_path / "empty.md"
    path.write_text("", encoding="utf-8")

    assert chunk_maker(str(path)) == []


def test_chunk_maker_reads_non_markdown_extension(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("plain text without headers\n", encoding="utf-8")

    chunks = chunk_maker(str(path))

    assert len(chunks) == 1
    assert chunks[0].page_content == "plain text without headers"
    assert "Header 1" not in chunks[0].metadata


def test_chunk_maker_raises_http_exception_for_missing_file(tmp_path):
    missing = tmp_path / "missing.md"

    with pytest.raises(HTTPException) as exc_info:
        chunk_maker(str(missing))

    assert exc_info.value.status_code == 500
    assert "No such file or directory" in exc_info.value.detail
    assert missing.name in exc_info.value.detail


def test_chunk_maker_raises_http_exception_for_invalid_utf8(tmp_path):
    path = tmp_path / "bad.md"
    path.write_bytes(b"\xff\xfe not utf-8")

    with pytest.raises(HTTPException) as exc_info:
        chunk_maker(str(path))

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail
