import pytest
from fastapi import HTTPException

from src.rag_app.ingestion import fileloader


def test_load_files_sets_source_and_chunk_index_for_one_markdown_file(tmp_path):
    path = tmp_path / "guide.md"
    path.write_text("# Guide\n\nPlace the widget.\n", encoding="utf-8")

    chunks = fileloader.load_files(str(path))

    assert len(chunks) == 1
    assert chunks[0].metadata["source"] == str(path)
    assert chunks[0].metadata["chunk_index"] == 0
    assert "Place the widget." in chunks[0].page_content


def test_load_files_numbers_chunks_from_one_file(tmp_path):
    path = tmp_path / "long.md"
    path.write_text("# Long\n\n" + ("sentence " * 500), encoding="utf-8")

    chunks = fileloader.load_files(str(path))

    assert len(chunks) > 1
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.metadata["source"] == str(path) for chunk in chunks)


def test_load_files_ignores_non_markdown_file(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("not markdown\n", encoding="utf-8")

    assert fileloader.load_files(str(path)) == []


def test_load_files_ignores_uppercase_md_extension(tmp_path):
    path = tmp_path / "notes.MD"
    path.write_text("# Title\n\nBody\n", encoding="utf-8")

    assert fileloader.load_files(str(path)) == []


def test_load_files_walks_directory_and_skips_non_markdown(tmp_path):
    (tmp_path / "keep.md").write_text("# Keep\n\nKept.\n", encoding="utf-8")
    (tmp_path / "skip.txt").write_text("skipped\n", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "child.md").write_text("# Child\n\nChild text.\n", encoding="utf-8")

    chunks = fileloader.load_files(str(tmp_path))
    sources = {chunk.metadata["source"] for chunk in chunks}

    assert sources == {str(tmp_path / "keep.md"), str(nested / "child.md")}
    assert all(chunk.metadata["chunk_index"] == 0 for chunk in chunks)


def test_load_files_restarts_chunk_index_for_each_file(tmp_path):
    for name in ("a.md", "b.md"):
        (tmp_path / name).write_text("# Title\n\n" + ("word " * 800), encoding="utf-8")

    chunks = fileloader.load_files(str(tmp_path))
    by_source = {}
    for chunk in chunks:
        by_source.setdefault(chunk.metadata["source"], []).append(chunk.metadata["chunk_index"])

    assert len(by_source) == 2
    for indexes in by_source.values():
        assert indexes == list(range(len(indexes)))
        assert indexes[0] == 0


def test_load_files_returns_empty_list_for_empty_directory(tmp_path):
    assert fileloader.load_files(str(tmp_path)) == []


def test_load_files_returns_empty_list_for_missing_path(tmp_path):
    missing = tmp_path / "does-not-exist"

    assert fileloader.load_files(str(missing)) == []


def test_load_files_raises_http_exception_when_chunk_maker_fails(tmp_path, monkeypatch):
    path = tmp_path / "guide.md"
    path.write_text("# Guide\n\nBody\n", encoding="utf-8")

    def fail(filepath):
        raise ValueError("bad chunk")

    monkeypatch.setattr(fileloader, "chunk_maker", fail)

    with pytest.raises(HTTPException) as exc_info:
        fileloader.load_files(str(path))

    assert exc_info.value.status_code == 500
    assert "bad chunk" in exc_info.value.detail
