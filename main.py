from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = ChatOllama(model="llama3.2:3b")

template = """
You are an expert in answering questions about a pizza restaurant.

Here are some relevant reviews:
{reviews}

Question: {question}

Answer based only on the reviews above.
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model


while True:

    print("\n....................")

    question = input("Ask your question (q to quit): ")

    if question.lower() == "q":
        break


    # retrieve documents
    docs = retriever.invoke(question)

    # convert docs to text
    reviews_text = "\n\n".join(
        doc.page_content for doc in docs
    )

    # run LLM
    result = chain.invoke({
        "reviews": reviews_text,
        "question": question
    })

    print("\nAnswer:")
    print(result.content)