import time
import uuid
from fastapi import FastAPI
from api.routes.rag_answer import RAGAnswerAPIRouter
from api.routes.ingestion import ingestion_router
from src.utils.logging import get_logger,request_id_var

logger = get_logger(__name__)
app = FastAPI(
    title ="RAG system"
)

@app.middleware("http")
async def log_request(request, call_next):
    """
    Middleware to log incoming requests and their processing time.
    :param request:
    :param call_next:
    :return:
    """
    request_id = request.headers.get("X-Request-ID") or uuid.uuid5(uuid.NAMESPACE_URL,str(uuid.uuid4())).hex[:12]
    token = request_id_var.set(request_id)
    start =time.perf_counter()
    logger.info(f"Request startedRequest ID: {request_id} - {request.method} {request.url}")
    try:
        response = await call_next(request)
    except Exception  as e:
        elapsed =(time.perf_counter() - start) *1000
        logger.exception(f"Request crashed. ID: {request_id} - Error: {e} elapsed time: {elapsed:.2f} ms")
        raise
    else:
        elapsed =(time.perf_counter() - start) *1000
        logger.info(f"Request completed. ID: {request_id} - response status code : {response.status_code} Time taken: {elapsed:.2f} ms")
        return response
    finally:
        request_id_var.reset(token)

app.include_router(RAGAnswerAPIRouter)
app.include_router(ingestion_router)

@app.get("/")
def root():
    return {"message":"app is running"}
