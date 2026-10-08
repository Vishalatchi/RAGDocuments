from sentence_transformers import CrossEncoder
from src.utils.logging import get_logger, log_timing

logger = get_logger(__name__)
def reranker(query, candidate, top_k=5):
    """
    candidates: list of quadrant scorepoint objects
    """
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
    pairs = [(query, can.payload["text"]) for can in candidate]
    scores = reranker.predict(pairs)
    scored = list(zip(candidate, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

