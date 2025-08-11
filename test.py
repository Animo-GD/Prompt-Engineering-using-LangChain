from langchain_community.llms import GPT4All
import os
model_filename = "qwen2.5-coder-7b-instruct-q4_0.gguf"
model_dir = os.path.join(os.path.expanduser("~"),r"AppData\Local\nomic.ai\GPT4All",model_filename)

llm = GPT4All(model=model_dir)

print(llm.invoke("Once upon a time,"))