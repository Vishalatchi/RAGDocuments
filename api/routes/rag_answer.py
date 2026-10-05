from fastapi import FastAPI
from fastapi import APIRouter
from src.rag_app.generation import answer


RAGAnswerAPIRouter = APIRouter(prefix="/ragapi", tags=["RAG Answer API"])

@RAGAnswerAPIRouter.post("/generate_answer")
async def generate_answer(query: str):
    """
    Generate an answer based on the provided query.
    :param query: The search query.
    :return: A JSON response containing the generated answer.
    """
    response = answer.generate_answer(query)
    return {"answer": response}
