from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are an expert in Software Project Management.

Here are some relevant excerpts from official sources:
{context}

Based on these, answer the following question clearly and concisely:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def format_docs_with_sources(docs):
    output = []
    for doc in docs:
        source = doc.metadata.get("source_file", "unknown.pdf")
        chunk = doc.metadata.get("chunk", "n/a")
        content = doc.page_content.strip()
        output.append(f"[Source: {source}, Chunk: {chunk}]\n{content}")
    return "\n\n".join(output)

while True:
    print("\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    if question.lower() == "q":
        break

    docs = retriever.invoke(question)
    print("\n--- Retrieved Chunks ---\n")
    for doc in docs:
        print(f"[{doc.metadata['source_file']} - Chunk {doc.metadata['chunk']}]\n{doc.page_content[:300]}...\n")

    context = format_docs_with_sources(docs)
    result = chain.invoke({"context": context, "question": question})
    print("\nAnswer:\n", result)
