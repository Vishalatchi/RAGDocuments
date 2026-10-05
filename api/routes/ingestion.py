from fastapi import APIRouter
from src.rag_app.ingestion import pipeline,fileloader
from dotenv import load_dotenv
import os
load_dotenv()

ingestion_router=APIRouter(tags=["ingestion"], prefix="/ingestion")

@ingestion_router.post("/create_collection")
def create_collection():
    """
    Create a collection in the Qdrant database. create collection and payload index for the collection.
    :return:
    """
    pipeline.create_collection(os.getenv("QDRANT_COLLECTION_NAME"))

    pipeline.create_payload_index(os.getenv("QDRANT_COLLECTION_NAME"))

    return {"message": "Data ingestion successful"}


@ingestion_router.post("/load_files")
def load_files(filepath:str):
    """
    Ingest data into the Qdrant database. Ingest data from the source path to the collection.
    :return:
    """
    data = fileloader.load_files(filepath)
    pipeline.chunk_embedding(data)
    return {"message": "Data ingestion successful"}


@ingestion_router.post("/delete_files")
def delete_files(filepath:str):
    """
    Delete all data from the Qdrant database. Delete all data from the collection.
    :return:
    """
    pipeline.delete_document(os.getenv("QDRANT_COLLECTION_NAME"),filepath)
    return {"message": "Data deletion successful"}



