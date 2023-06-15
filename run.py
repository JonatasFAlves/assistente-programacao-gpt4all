from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

local_path = './models/ggml-gpt4all-l13b-snoozy.bin'

template = """A partir de agora você sempre responderá em português no contexto de programação de computadores.

Você sempre irá gerar um código de exemplo quando necessário.

Questão: {question}

Resposta: Explicarei de uma maneira simples."""

prompt = PromptTemplate(template=template, input_variables=["question"])

# Callbacks support token-wise streaming
callbacks = [StreamingStdOutCallbackHandler()]
# Verbose is required to pass to the callback manager
llm = GPT4All(model=local_path, callbacks=callbacks, verbose=False)
# If you want to use a custom model add the backend parameter
# Check https://docs.gpt4all.io/gpt4all_python.html for supported backends
# llm = GPT4All(model=local_path, backend='gptj', callbacks=callbacks, verbose=True)

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "O que é herança em java?"

llm_chain.run(question)




# gptj = gpt4all.GPT4All("ggml-gpt4all-l13b-snoozy.bin", model_path="./models/")
# messages = [{"role":"system", "content":"A partir de agora você responderá em português no contexto de programação."},{"role": "user", "content": "O que é herança em java?"}]
# gptj.chat_completion(messages)