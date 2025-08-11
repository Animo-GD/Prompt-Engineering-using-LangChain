from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="mistral")

template = "Translate the following text to {language}: {text}"
prompt_template = PromptTemplate.from_template(template)

prompt = prompt_template.format(language="French",text="I love you")

print(llm.invoke(prompt))