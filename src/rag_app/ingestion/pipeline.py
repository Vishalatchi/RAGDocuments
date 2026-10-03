import ollama
from langchain_ollama import OllamaEmbeddings
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
import os
import uuid
from dotenv import load_dotenv
from collections import Counter

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
    if not qd_client.collection_exists(collection_name=collection_name):
        qd_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(size=768, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        print("collection created")

def generate_point_id(source: str, chunk_index: int) -> str:
    """
    Generate a unique point ID based on the source and chunk index.
    :param source: The source of the document (e.g., file path).
    :param chunk_index: The index of the chunk within the document.
    :return: A unique point ID as a string.
    """
    value = f"{source}:{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))

def chunk_embedding(chunks, batch_size=50):
    """
    Embed the given chunks using both dense and sparse embeddings.
    :param chunks: List of document chunks to embed.
    :param batch_size: The size of batches for embedding.
    :return: List of points with embeddings and metadata.
    """
    points = []

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
        embedding_text = f"{header_text}\n\n{file_chunk.page_content}"

        # Dense Embedding
        dense_vector = dense_embed.embed_query(embedding_text)

        # Sparse Embedding
        sparse_embedding = list(sparse_embed.embed(embedding_text))[0]
        sparse_vector = models.SparseVector(
            indices=sparse_embedding.indices.tolist(),
            values=sparse_embedding.values.tolist(),
        )

        point_id = generate_point_id(file_chunk.metadata["source"], file_chunk.metadata["chunk_index"])
        print(f"point id generated: {point_id}")
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
            print(f"{len(points)} loaded")
            points.clear()
    if points:
        qd_client.upsert(
            collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
            points=points,
            wait=True)
        print(f"{len(points)} loaded")
    return points

def delete_document(collection_name: str, source: str):
    """
    Delete all points in the collection that match the given source.
    :param collection_name: The name of the collection in Qdrant.
    :param source: The source of the document to delete.
    """
    filter_condition = models.Filter(
        must=[
            models.FieldCondition(
                key="source",
                match=models.MatchValue(value=source)
            )
        ]
    )
    qd_client.delete(
        collection_name=collection_name,
        points_selector=filter_condition,
        wait=True
    )

def create_payload_index(collection_name: str):
    """
    Create a payload index for the 'source' field in the collection.
    :param collection_name: The name of the collection in Qdrant.
    """
    qd_client.create_payload_index(
        collection_name=collection_name,
        field_name="source",
        field_schema=models.PayloadSchemaType.KEYWORD
    )
