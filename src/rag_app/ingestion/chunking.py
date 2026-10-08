from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from src.utils.logging import get_logger,log_timing
from fastapi import HTTPException

logger = get_logger(__name__)
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
)

def chunk_maker(filepath: str) -> list:
    try:
        logger.info(f"Starting to chunk the file: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = f.read()
        sections = markdown_splitter.split_text(data)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=200)
        file_chunk = text_splitter.split_documents(sections)
        logger.info(f"Completed chunking the file: {filepath}. Total chunks created: {len(file_chunk)}")
        return file_chunk
    except Exception as e:
        logger.exception(f"Error in chunk_maker for file '{filepath}': {e}")
        raise HTTPException(status_code=500, detail=str(e))