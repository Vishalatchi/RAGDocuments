import os
from langchain_core.documents import Document
from src.rag_app.ingestion.chunking import chunk_maker
from src.utils.logging import get_logger,log_timing
from fastapi import HTTPException

logger =get_logger(__name__)
def load_files(filepath:str)->list[Document]:
    file_chunks=[]
    try:
        logger.info(f"Loading files from '{filepath}'")
        if os.path.isfile(filepath):
            if filepath.endswith(".md"):
                for i , chunk in enumerate(chunk_maker(filepath)):
                    chunk.metadata["source"]=filepath
                    chunk.metadata["chunk_index"]=i
                    file_chunks.append(chunk)
        else:
            for root,_,files in os.walk(filepath):
                for file in files:
                    if file.endswith(".md"):
                        full_path = os.path.join(root, file)
                        # Process the markdown file
                        for i, chunk in enumerate(chunk_maker(full_path)):
                            chunk.metadata["source"] = full_path
                            chunk.metadata["chunk_index"] = i
                            file_chunks.append(chunk)
        logger.info(f"Loaded {len(file_chunks)} chunks from '{filepath}'")
        return file_chunks
    except Exception as e:
        logger.exception(f"Error loading files from '{filepath}': {e}")
        raise HTTPException(status_code=500, detail=str(e))

