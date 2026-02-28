from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model = 'llama3.2')

template = """
You are an epert in answering questions about a pizza restaurant

Here are some relevant: {reviews}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

result = chain.invoke({"reviews": [] ,"question":"what is the best pizza place in town"})
print(result)