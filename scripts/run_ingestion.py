from src.rag_app.ingestion import fileloader
from src.rag_app.ingestion import pipeline
import os
from dotenv import load_dotenv
load_dotenv()
path=(r"C:\Users\Vinoth\Python\AI programs\Power Automate\power-automate-docs\articles\mobile")

data = fileloader.load_files(path)
for d in data:
   print (d)

print("file chunks created ")

#pipeline.create_collection(os.getenv("QDRANT_COLLECTION_NAME"))

pipeline.chunk_embedding(data)


#pipeline.create_payload_index(os.getenv("QDRANT_COLLECTION_NAME"))