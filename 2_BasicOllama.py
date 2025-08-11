from langchain_community.llms import Ollama
llm = Ollama(model="mistral")
response = llm.invoke("Explain AI in a simple way")
print(response)