from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

gpt4all_model_path = './models/ggml-gpt4all-l13b-snoozy.bin'

loader = TextLoader("./docs/herança.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder="./models")
vectorstore = Chroma.from_documents(documents, embeddings)

callbacks = [StreamingStdOutCallbackHandler()]
llm = GPT4All(model=gpt4all_model_path, callbacks=callbacks, verbose=False)


qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(search_kwargs={"k": 1}))
# qa.run("O que é herança?")

res = qa(f"""
    A partir de agora você responderá em português no contexto de programação de computadores. Use o texto disponibilizado como base para sua resposta.
    O que é herança?
""")
print(res["result"])