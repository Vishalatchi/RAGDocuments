from fastapi import FastAPI
from fastapi import APIRouter
from fastapi import HTTPException
from src.rag_app.generation import answer
from src.utils.logging import get_logger, log_timing

logger = get_logger(__name__)
RAGAnswerAPIRouter = APIRouter(prefix="/ragapi", tags=["RAG Answer API"])
@RAGAnswerAPIRouter.post("/generate_answer")
async def generate_answer(query: str):
    """
    Generate an answer based on the provided query.
    :param query: The search query.
    :return: A JSON response containing the generated answer.
    """
    logger.info(f"Received query in generate_answer: {query}")
    try:
        response = answer.generate_answer(query)
    except Exception as e:
        logger.exception(f"Error generating answer for query '{query}': {e}")
        raise HTTPException(status_code=500, detail=str(e))
    logger.info(f"Generated answer for query '{query}': {response}")
    return {"answer": response}
