import gpt4all
gptj = gpt4all.GPT4All("ggml-gpt4all-l13b-snoozy.bin", model_path="./models/")
messages = [{"role":"system", "content":"A partir de agora você responderá em português no contexto de programação."},{"role": "user", "content": "O que é herança em java?"}]
gptj.chat_completion(messages)