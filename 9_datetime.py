from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

llm = OllamaLLM(model="mistral")
formating_instruction = "Replay Only with a datetime format DAY/MONTH/YEAR like 11/02/2001"
prompt_temp = PromptTemplate(
    template="When was {person} born?\n{formating_instruction}",
    input_variables=["person","formating_instruction"]
)

prompt = prompt_temp.format(person="Elon Musk",formating_instruction=formating_instruction)

response = llm.invoke(prompt)

print(response)