import pinecone
import os
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file
env_vars = load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
env =  os.getenv("PINECONE_ENVIRONMENT")

index_name = "langchain-pinecone-search"

pinecone.init(api_key=api_key, environment=env)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder="./models")

vectorStore = Pinecone.from_existing_index(index_name, embeddings)

query = "O que é herança?"
docs = vectorStore.similarity_search(query)

print(docs[0].page_content)