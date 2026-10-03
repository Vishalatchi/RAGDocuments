from qdrant_client import QdrantClient, models
import os
from dotenv import load_dotenv
import ollama
from fastembed import SparseTextEmbedding
from src.rag_app.retrieval.reranker import reranker

load_dotenv()

def hybrid_search(query, limit=5, retrieval=20):
    """
    Perform a hybrid search using dense and sparse embeddings.
    :param query: The search query.
    :param collection_name: The name of the Qdrant collection to search in.
    :param limit: The maximum number of results to return.
    :param retrieval: The number of candidates to retrieve for reranking.
    :return: A list of top-k reranked results.
    """
    # dense query vector
    ol_client =ollama.Client(host="localhost")
    dense = ol_client.embeddings(model="nomic-embed-text", prompt=query)
    dense_query = dense["embedding"]

    # sparse query vector
    sparse = SparseTextEmbedding(model_name="Qdrant/BM25")
    sparse_embedding = list(sparse.embed(query))[0]
    sparse_query = models.SparseVector(
        indices=sparse_embedding.indices.tolist(),
        values=sparse_embedding.values.tolist(),
    )

    # hybrid query
    qd_client = QdrantClient(
        api_key=os.getenv("QDRANT_API_KEY"), url=os.getenv("QDRANT_CLUSTER_ENDPOINT")
    )
    results = qd_client.query_points(
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        prefetch=[
            models.Prefetch(query=dense_query, using="dense", limit=retrieval),
            models.Prefetch(query=sparse_query, using="sparse", limit=retrieval),
        ],
        query=models.RrfQuery(rrf=models.Rrf(k=60)),
        limit=limit,
        with_payload=True,
    )

    candidates = results.points
    reranked_result=reranker(query, candidates, top_k=limit)
    return reranked_result




