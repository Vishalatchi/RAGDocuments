
from fastapi import FastAPI
from api.routes.rag_answer import RAGAnswerAPIRouter
from api.routes.ingestion import ingestion_router
app = FastAPI(
    title ="RAG system"
)
app.include_router(RAGAnswerAPIRouter)
app.include_router(ingestion_router)

@app.get("/")
def root():
    return {"message":"app is running"}
