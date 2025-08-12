from langchain_ollama import OllamaLLM
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

from pydantic import BaseModel,Field
from typing import List

# task = "List three popular Arabian plates."
llm = OllamaLLM(model="mistral")

class Plate(BaseModel):
    plate_name: str = Field(...,description="The Name Of The Plate",max_length=50)
    ingredient: List[str] = Field(...,description="The Ingredient Of the Plate")

parser = PydanticOutputParser(pydantic_object=Plate)
format_instructions = parser.get_format_instructions() 

prompt_template = PromptTemplate(
    template="State The Following\n{plate}\n{format_instructions}",
    input_variables=["plate"],
    partial_variables={"format_instructions":format_instructions}
)

fav_plate = "Shawerma"

prompt = prompt_template.format(plate=fav_plate)

response = llm.invoke(prompt)
output = parser.parse(response)
print("Plate Name : ",output.plate_name)
print("Ingredient : ",output.ingredient)
