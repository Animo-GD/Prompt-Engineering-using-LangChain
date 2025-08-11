from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate,FewShotPromptTemplate

llm = OllamaLLM(model="mistral",temperature=0.2)

template = PromptTemplate(
    template="Question : {question}\n------\nAnswer : {answer}",
    input_variables=["question","answer"]
)
examples = [
    {"question":"When I was seven, my sister was twice my age. Now I am seventy years old, how old can my sister be?",
     "answer":"\n".join([
        "We will followup some questions to get the answer.",
        "Follow up: How old was your sister when you were seven?",
        "Intermediate answer: Twice, which mean 14 years.",
        "Follow up: What is the difference between your age and your sister's age?",
        "Intermediate answer: 14 years - 7 years = 7 years.",
        "Follow up: When you were seventy years old, how old would your sister be?",
        "Intermediate answer: my age (70) +  The difference between me and my sister's age (7) = 77 years.",
        "Final Answer: 77 years."
     ])},
    {"question":"I have five oranges, and the sum of what my sister and brother have is three times what I have plus thirty-five?",
     "answer":"\n".join([
        "We will followup some questions to get the answer.",
        "Follow up: How many oranges do you have?",
        "Intermediate answer: 5 oranges.",
        "Follow up: How many oranges does your sister and brother have?",
        "Intermediate answer: three times: = 3 * 5 = 15. Also, we need to add thirty-five to them: 15 + 35 = 50 ",
        "Final Answer: 50 oranges."
     ])}
]
fewShot_template = FewShotPromptTemplate(
    examples = examples,
    example_prompt=template,
    suffix = "Question : {question}",
    input_variables = ["question"]
)

user_question = "Total with what my family is 50 oranges. If we subtract one-fifth of the number from them, and add ten more oranges, how many oranges will there be in the end?"
prompt = fewShot_template.format(question=user_question)

answer = llm.invoke(prompt)

print(answer)