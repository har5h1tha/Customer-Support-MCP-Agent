from pathlib import Path

import chromadb
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Read the troubleshooting document
file_path = Path("data/troubleshooting.txt")
text = file_path.read_text(encoding="utf-8")

# 2. Split the document into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

# 3. Connect to persistent ChromaDB storage
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="support_knowledge"
)

# 4. Generate embeddings and store the chunks
for index, chunk in enumerate(chunks):
    response = ollama.embed(
        model="nomic-embed-text",
        input=chunk
    )

    embedding = response["embeddings"][0]

    collection.upsert(
        ids=[f"support-{index}"],
        documents=[chunk],
        embeddings=[embedding],
        metadatas=[{"source": file_path.name}]
    )

print(f"Indexed {len(chunks)} chunks successfully.")