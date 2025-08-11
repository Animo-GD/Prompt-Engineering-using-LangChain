from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="mistral",temperature=.5)

demo_temp = PromptTemplate(
    template="Country Name {country} | Capital Name {capital}",
    input_variables=["country","capital"],
) 
examples = [
    {"country":"Egypt","capital":"Cairo"},
    {"country":"Sudia Arabia","capital":"Riyadh"},
    {"country":"Singapore","capital":"Singapore"}
]

fewshot_prompt = FewShotPromptTemplate(
    example_prompt = demo_temp,
    examples = examples,
    suffix = "Country Name {country} |",
    input_variables=["country"]
)

user_country = "Australia"

prompt = fewshot_prompt.format(country=user_country)

print(llm.invoke(prompt))