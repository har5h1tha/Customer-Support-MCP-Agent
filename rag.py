import chromadb
import ollama

# 1. Connect to the existing ChromaDB database
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection(name="support_knowledge")


def retrieve_context(query: str, top_k: int = 3) -> list[str]:
    # 2. Convert the user's question into an embedding
    response = ollama.embed(
        model="nomic-embed-text",
        input=query
    )

    query_embedding = response["embeddings"][0]

    # 3. Retrieve the most relevant document chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count())
    )

    return results["documents"][0]


def ask_question(question: str) -> str:
    # 4. Retrieve relevant information
    chunks = retrieve_context(question)

    context = "\n\n".join(chunks)

    # 5. Ask the LLM to answer using the retrieved context
    prompt = f"""
You are a customer support assistant.

Answer the user's question using ONLY the provided context.
If the context does not contain the answer, clearly say that
you do not have enough information.

Context:
{context}

User question:
{question}

Answer:
"""

    response = ollama.chat(
        model="llama3.2:latest",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]


if __name__ == "__main__":
    while True:
        question = input("\nAsk a support question (or type 'exit'): ")

        if question.strip().lower() == "exit":
            break

        if not question.strip():
            continue

        answer = ask_question(question)
        print("\nAnswer:", answer)