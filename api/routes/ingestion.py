from fastapi import APIRouter
from src.rag_app.ingestion import pipeline,fileloader
from dotenv import load_dotenv
import os
from src.utils.logging import get_logger, log_timing

logger = get_logger(__name__)
load_dotenv()

ingestion_router=APIRouter(tags=["ingestion"], prefix="/ingestion")

@ingestion_router.post("/create_collection")
def create_collection():
    """
    Create a collection in the Qdrant database. create collection and payload index for the collection.
    :return:
    """
    try:

        pipeline.create_collection(os.getenv("QDRANT_COLLECTION_NAME"))

        pipeline.create_payload_index(os.getenv("QDRANT_COLLECTION_NAME"))

        return {"message": "Data ingestion successful"}
    except Exception as e:
        logger.exception(f"Error creating collection: {e}")
        return {"message": f"Error creating collection: {str(e)}"}


@ingestion_router.post("/load_files")
def load_files(filepath:str):
    """
    Ingest data into the Qdrant database. Ingest data from the source path to the collection.
    :return:
    """
    try:
        data = fileloader.load_files(filepath)
        pipeline.chunk_embedding(data)
        logger.info(f"Successfully ingested data from '{filepath}' into collection '{os.getenv('QDRANT_COLLECTION_NAME')}'.")
        return {"message": "Data ingestion successful"}
    except Exception as e:
        logger.exception(f"Error loading files: {e}")
        return {"message": f"Error loading files: {str(e)}"}


@ingestion_router.post("/delete_files")
def delete_files(filepath:str):
    """
    Delete all data from the Qdrant database. Delete all data from the collection.
    :return:
    """
    try:
        pipeline.delete_document(os.getenv("QDRANT_COLLECTION_NAME"),filepath)
        logger.info(f"Successfully deleted data from '{filepath}' in collection '{os.getenv('QDRANT_COLLECTION_NAME')}'.")
        return {"message": "Data deletion successful"}
    except Exception as e:
        logger.exception(f"Error deleting files: {e}")
        return {"message": f"Error deleting files: {str(e)}"}


