from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="mistral")

output_parser = CommaSeparatedListOutputParser()

format_instruction = output_parser.get_format_instructions()

prompt_template = PromptTemplate(
    template="List Three popular {type} plates.\n{format_instruction}",
    input_variables=["type"],
    partial_variables={"format_instruction":format_instruction}
)

prompt = prompt_template.format(type="Arabian")
reponse = llm.invoke(prompt)
print("Comma Seprated Output: ",reponse)

output = output_parser.parse(reponse)

print("Parsed Output: ",output)