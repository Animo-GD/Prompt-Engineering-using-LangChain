from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

llm = OllamaLLM(model="mistral")

message_1 = "List Three Places From Egypt"
message_2 = "Which Place has the most population"
message_3 = "Can I visit it in winter?"
conversation = ConversationChain(
    llm=llm,
    memory = ConversationBufferMemory()
)

output_1 = conversation.predict(input=message_1)
output_2 = conversation.predict(input=message_2)
output_3 = conversation.predict(input= message_3)

print(f"{output_1}\n{output_2}\n{output_3}")

conversation.memory.clear()