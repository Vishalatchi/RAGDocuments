from src.rag_app.generation import answer
from qdrant_client import QdrantClient, models
import json
import os
from dotenv import  load_dotenv

load_dotenv()

def scrolldata():
    gd_client = (
        QdrantClient(
            api_key=os.getenv("QDRANT_API_KEY"),
            url=os.getenv("QDRANT_CLUSTER_ENDPOINT")
        )
    )
    source_path = r""

    points, next_page = gd_client.scroll(
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        scroll_filter=models.Filter(
             must=[
                models.FieldCondition(
                key="source", match=models.MatchValue(value=source_path)
                )
            ]
        ),
        limit=10,
        with_vectors=True,
        with_payload=True,
    )

    print(points)



if __name__ == "__main__":
    query = ("how do I buy android phone")
    response = answer.generate_answer(query)
    if isinstance(response, str):
        data = response.strip().removeprefix("```json").removesuffix("```").strip()
        ans = json.loads(data)
    else:
        ans = response

    datas=ans.content.split("\\n")
    for d in datas:
        print(d)


