import pinecone
import os
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
import time
from langchain.chains.question_answering import load_qa_chain

start_time = time.time()


# Load environment variables from .env file
env_vars = load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
env =  os.getenv("PINECONE_ENVIRONMENT")

index_name = "langchain-pinecone-search"

pinecone.init(api_key=api_key, environment=env)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder="./models")

vectorStore = Pinecone.from_existing_index(index_name, embeddings)

gpt4all_model_path = './models/ggml-gpt4all-l13b-snoozy.bin'
# gpt4all_model_path = './models/ggml-gpt4all-j-v1.3-groovy.bin'

callbacks = [StreamingStdOutCallbackHandler()]
llm = GPT4All(model=gpt4all_model_path, callbacks=callbacks, verbose=True, n_threads=6)




prompt_template = """Use o contexto disponibilizado como base para sua resposta.

{context}

Questão: {question}
Responda em português:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)



query = "O que é herança?"
docs = vectorStore.similarity_search(query, k=1)

print(PROMPT.format(context=docs[0].page_content, question=query))


chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)
chain({"input_documents": docs, "question": query}, return_only_outputs=True)





end_time = time.time()
execution_time = end_time - start_time

print("Time taken:", execution_time, "seconds")