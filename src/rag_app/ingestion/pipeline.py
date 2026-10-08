import ollama
from langchain_ollama import OllamaEmbeddings
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
import os
import uuid
from dotenv import load_dotenv
from src.utils.logging import get_logger, log_timing
from fastapi import HTTPException

logger = get_logger(__name__)

load_dotenv()

qd_client=QdrantClient(
    api_key=os.getenv("QDRANT_API_KEY"), url=os.getenv("QDRANT_CLUSTER_ENDPOINT")
)

ol_client=ollama.Client(host="localhost")

dense_embed =OllamaEmbeddings(
    model="nomic-embed-text")

sparse_embed =SparseTextEmbedding(
    model_name="Qdrant/BM25")


def create_collection(collection_name:str):
    """
    create collection in qdrant. Two separate vector configurations are created for dense and sparse embeddings.
    :param collection_name:
    :return:
    """
    try:
        if not qd_client.collection_exists(collection_name=collection_name):
            logger.info(f"Creating collection '{collection_name}' in Qdrant.")
            qd_client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": models.VectorParams(size=768, distance=models.Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )
            logger.info(f"Collection '{collection_name}' created successfully.")
        else:
            logger.info(f"Collection '{collection_name}' already exists in Qdrant.")
    except Exception as e:
        logger.exception(f"Error creating collection '{collection_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_point_id(source: str, chunk_index: int) -> str:
    """
    Generate a unique point ID based on the source and chunk index.
    :param source: The source of the document (e.g., file path).
    :param chunk_index: The index of the chunk within the document.
    :return: A unique point ID as a string.
    """
    try:
        value = f"{source}:{chunk_index}"
        logger.info(f"Generating point ID for source '{source}' and chunk index '{chunk_index}': {value}")
        return str(uuid.uuid5(uuid.NAMESPACE_URL, value))
    except Exception as e:
        logger.exception(f"Error generating point ID for source '{source}' and chunk index '{chunk_index}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


def chunk_embedding(chunks, batch_size=50):
    """
    Embed the given chunks using both dense and sparse embeddings.
    :param chunks: List of document chunks to embed.
    :param batch_size: The size of batches for embedding.
    :return: List of points with embeddings and metadata.
    """

    try:
        points = []
        logger.info(f"Starting chunk embedding for {len(chunks)} chunks with batch size {batch_size}.")
        for file_chunk in chunks:
            print(f"chunks started")
            header_text = "\n".join(
                filter(
                    None,
                    [
                        file_chunk.metadata.get("Header 1"),
                        file_chunk.metadata.get("Header 2"),
                        file_chunk.metadata.get("Header 3"),
                    ],
                )
            )
            logger.info(f"Embedding text for chunk with source '{file_chunk.metadata.get('source')}' and chunk index '{file_chunk.metadata.get('chunk_index')}'.")
            embedding_text = f"{header_text}\n\n{file_chunk.page_content}"

            # Dense Embedding
            logger.info(f"Generating dense embedding for chunk with source '{file_chunk.metadata.get('source')}' and chunk index '{file_chunk.metadata.get('chunk_index')}'.")
            dense_vector = dense_embed.embed_query(embedding_text)

            # Sparse Embedding
            logger.info(f"Generating sparse embedding for chunk with source '{file_chunk.metadata.get('source')}' and chunk index '{file_chunk.metadata.get('chunk_index')}'.")
            sparse_embedding = list(sparse_embed.embed(embedding_text))[0]
            sparse_vector = models.SparseVector(
                indices=sparse_embedding.indices.tolist(),
                values=sparse_embedding.values.tolist(),
            )

            point_id = generate_point_id(file_chunk.metadata["source"], file_chunk.metadata["chunk_index"])
            logger.info(f"Generated point ID '{point_id}' for chunk with source '{file_chunk.metadata.get('source')}' and chunk index '{file_chunk.metadata.get('chunk_index')}'.")
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={"dense": dense_vector, "sparse": sparse_vector},
                    payload={
                        "source": file_chunk.metadata.get("source"),
                        "chunk_index": file_chunk.metadata.get("chunk_index"),
                        "Header 1": file_chunk.metadata.get("Header 1"),
                        "Header 2": file_chunk.metadata.get("Header 2"),
                        "Header 3": file_chunk.metadata.get("Header 3"),
                        "text": file_chunk.page_content,
                    }
                )
            )
            if len(points) >= batch_size:
                qd_client.upsert(
                    collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
                    points=points,
                    wait=True)
                logger.info(f"Upserted {len(points)} points to collection '{os.getenv('QDRANT_COLLECTION_NAME')}'.")
                points.clear()
        if points:
            qd_client.upsert(
                collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
                points=points,
                wait=True)
            logger.info(f"Upserted remaining {len(points)} points to collection '{os.getenv('QDRANT_COLLECTION_NAME')}'.")
        return points
    except Exception as e:
        logger.exception(f"Error during chunk embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def delete_document(collection_name: str, source: str):
    """
    Delete all points in the collection that match the given source.
    :param collection_name: The name of the collection in Qdrant.
    :param source: The source of the document to delete.
    """
    try:
        filter_condition = models.Filter(
            must=[
                models.FieldCondition(
                    key="source",
                    match=models.MatchValue(value=source)
                )
            ]
        )
        logger.info(f"Deleting points from collection '{collection_name}' with source '{source}'.")
        qd_client.delete(
            collection_name=collection_name,
            points_selector=filter_condition,
            wait=True
        )
        logger.info(f"Deleted points from collection '{collection_name}' with source '{source}'.")
    except Exception as e:
        logger.exception(f"Error deleting points from collection '{collection_name}' with source '{source}': {e}")
        raise HTTPException(status_code=500, detail=str(e))

def create_payload_index(collection_name: str):
    """
    Create a payload index for the 'source' field in the collection.
    :param collection_name: The name of the collection in Qdrant.
    """
    try:
        logger.info(f"Creating payload index for 'source' field in collection '{collection_name}'.")
        qd_client.create_payload_index(
            collection_name=collection_name,
            field_name="source",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
    except Exception  as e:
        logger.exception(f"Error creating payload index for 'source' field in collection '{collection_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))
