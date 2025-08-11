from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

llm = OllamaLLM(model="mistral")

template_message = "\n".join([
    "Explain for me",
    "{subject}",
    "like I am",
    "{age}"
])
prompt_temp = PromptTemplate(
    template=template_message,
    input_variables=["subject","age"]
    )

prompt = prompt_temp.format(subject="AI",age=5)
reponse = llm.invoke(prompt)
print(reponse)