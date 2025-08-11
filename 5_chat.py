from langchain_ollama import OllamaLLM
from langchain_core.messages import AIMessage,SystemMessage,HumanMessage
llm = OllamaLLM(model="mistral")
# base_conversation = [
#     SystemMessage("You are a professional Programmer, you should ask me some question to specify my specialization."),
#     AIMessage("What is your favirot language?"),
#     HumanMessage("Python"),
#     AIMessage("Do you have a problem solving skills?"),
#     HumanMessage("Not too much, just basics"),
#     AIMessage("Do you have any degrees BSc, MSc?"),
#     HumanMessage("Yes I have a BSC in computer science and engineering")
# ]
# response = llm.invoke(base_conversation)
# print(response)

conversation = [
    SystemMessage(
        "You are a professional programmer interviewer. "
        "Ask exactly ONE question at a time. "
        "Do not include more than one question in your reply. "
        "After I answer, ask the next question."
        "All questions must be very simple"
    )
    ]
for _ in range(3):
    # AI Response
    ai_response = llm.invoke(conversation)
    first_line = ai_response.splitlines()[0].strip()
    print("AI: ",first_line)
    conversation.append(AIMessage(first_line))
    print("\n\n")
    # Human Response
    human_reponse = input("Human: ").strip()
    conversation.append(HumanMessage(human_reponse))
    print("\n\n")
final_msg = "Now After you have all these questions quess my specialization."
conversation.append(HumanMessage(final_msg))
ai_answer = llm.invoke(conversation)
print(ai_answer)