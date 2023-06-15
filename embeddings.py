# todo: i can prob use text loader embeddings for now

from langchain.embeddings import LlamaCppEmbeddings

local_path = './models/ggml-gpt4all-l13b-snoozy.bin'

llama_embeddings = LlamaCppEmbeddings(model_path=local_path)

text = "This is a test document."

query_result = llama_embeddings.embed_query(text)

print(query_result)

doc_result = llama_embeddings.embed_documents([text])

print(doc_result)