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

start_time = time.time()


# Load environment variables from .env file
env_vars = load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
env =  os.getenv("PINECONE_ENVIRONMENT")

index_name = "langchain-pinecone-search"

pinecone.init(api_key=api_key, environment=env)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder="./models")

vectorStore = Pinecone.from_existing_index(index_name, embeddings)




# TODO: abstract pinecone and PromptTemplate to their own files
# gpt4all_model_path = './models/ggml-gpt4all-l13b-snoozy.bin'
gpt4all_model_path = './models/ggml-gpt4all-j-v1.3-groovy.bin'

template = """A partir de agora você sempre responderá em português no contexto de programação de computadores.

Use o contexto disponibilizado como base para sua resposta.

Questão: {question}
Responda em português:"""

prompt = PromptTemplate(template=template, input_variables=["question"])

question = prompt.format(question="O que é herança?")




callbacks = [StreamingStdOutCallbackHandler()]
llm = GPT4All(model=gpt4all_model_path, callbacks=callbacks, verbose=True, n_threads=6)

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorStore.as_retriever(search_kwargs={"k": 1}))
qa.run(question)


end_time = time.time()
execution_time = end_time - start_time

print("Time taken:", execution_time, "seconds")