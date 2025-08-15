from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory,ConversationBufferWindowMemory,ConversationSummaryMemory
from langchain.chains import ConversationChain



llm = OllamaLLM(model="mistral")

message_1 = "List Three Places From Egypt"
message_2 = "Which Place has the most population"
message_3 = "Can I visit it in winter?"

# ConversationBufferMemory -> If you want to send all previous messages and that is cost a lot of tokens
conversation = ConversationChain(
    llm=llm,
    memory = ConversationBufferMemory(),
    verbose= True
)


output_1 = conversation.predict(input=message_1)
output_2 = conversation.predict(input=message_2)
output_3 = conversation.predict(input= message_3)


conversation.memory.clear()

# ConversationBufferWindowMemory -> If you want `k` messages to be sent
conversation = ConversationChain(
    llm=llm,
    memory = ConversationBufferWindowMemory(k=1),
    verbose= True
)
output_1 = conversation.predict(input=message_1)
output_2 = conversation.predict(input=message_2)
output_3 = conversation.predict(input= message_3)


# ConversationSummaryMemory -> Use LLM To the summary the conversation to use it for the context
conversation.memory.clear()

conversation = ConversationChain(
    llm = llm,
    memory = ConversationSummaryMemory(llm=llm),
    verbose=True
)

output_1 = conversation.predict(input=message_1)
output_2 = conversation.predict(input=message_2)
output_3 = conversation.predict(input= message_3)

conversation.memory.clear()
# Entity Memory -> Use the entities that was presented in previous conversation afte summarize it

from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE

conversation = ConversationChain(
    llm = llm,
    memory = ConversationEntityMemory(llm=llm),
    prompt = ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    verbose=True
)

# Save Conversation to file
from langchain.schema import message_to_dict,messages_from_dict
import json

conv_dict = message_to_dict(conversation.memory.chat_memory.messages)

with open("conversation-memory.json","w") as dest:
    dest.write(json.dumps(conv_dict))

# Load saved conversation
from langchain.memory import ChatMessageHistory
with open("conversation-memory.json") as src:
    saved_history = json.loads(src.read())

history = ChatMessageHistory()
history.messages = messages_from_dict(saved_history)

memory = ConversationSummaryMemory(llm=llm)
memory = memory.from_messages(chat_memory=history,llm=llm)

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)