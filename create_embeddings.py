import os
import pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file
env_vars = load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
env =  os.getenv("PINECONE_ENVIRONMENT")

index_name = "langchain-pinecone-search"
delete_index = False

pinecone.init(api_key=api_key, environment=env)

if delete_index:
    pinecone.delete_index(name="langchain-pinecone-search")

# Check if index already exists, create it if it doesn't
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=384, metric='euclidean')

# create the index
# pinecone.create_index(
#     name=index_name,
#     dimension=1536,  # dimensionality of dense model
#     metric="dotproduct",  # sparse values supported only for dotproduct
#     pod_type="s1",
#     metadata_config={"indexed": []},  # see explaination above
# )
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder="./models")

loader = TextLoader("./docs/herança.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(documents)

Pinecone.from_documents(documents, embeddings, index_name=index_name)