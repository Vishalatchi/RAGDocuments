from langchain_groq import ChatGroq
from src.rag_app.retrieval.hybrid_search import hybrid_search
import os
from dotenv import load_dotenv
load_dotenv()

def filter_context_relative_response(result_rank,margin=0.25, min_query=0.05,max_chunks =3):
    """
    Filter the context relative response based on the provided parameters.
    :param result_rank:
    :param margin:
    :param min_query:
    :param max_chunks:
    :return:
    """
    if not min_query or result_rank==[]:
        return []
    else:
        related_response = [ point.payload["text"] for point,score in result_rank if(result_rank[0][1]-score <= margin )][:max_chunks]
        return related_response

def generate_answer(query):
    """
    Generate an answer based on the provided query and reranked results.
    :param query:
    :param reranked_result:
    :return:
    """
    model= os.getenv("MODEL")
    reranked_result =hybrid_search(query)
    context_chunks = filter_context_relative_response(reranked_result)
    prompt = f"""
              You are an assistant. review the user query and provide answer only
              from the context provided. do not hallucinate and do not invent answer. 
              if you don't have information, say you don't have information. provide answer in json format.
              Context:\n{context_chunks}\n\nQuestion:\n{query}\n\nAnswer:"""
    chat_groq = ChatGroq(model=model)
    response = chat_groq.invoke(prompt)
    return response

