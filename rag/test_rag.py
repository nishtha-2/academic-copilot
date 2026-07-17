import os

from pdf_loader import extract_text
from chunker import chunk_text
from embedder import create_embedding
from vector_store import build_index
from retriever import search
from generator import build_context

pdf_folder = "../data/documents"

all_chunks = []
all_embeddings = []

print("\nLoading PDFs...\n")

for file in os.listdir(pdf_folder):

    if file.endswith(".pdf"):

        path = os.path.join(pdf_folder, file)

        text = extract_text(path)

        chunks = chunk_text(text)

        print(f"PDF: {file}")
        print(f"Text Length: {len(text)}")
        print(f"Chunks: {len(chunks)}")
        for chunk in chunks:

            # Add source information
            chunk["source"] = file

            all_chunks.append(chunk)

            embedding = create_embedding(
                chunk["text"]
            )

            all_embeddings.append(embedding)

print("\n" + "=" * 50)

print(f"Total Chunks: {len(all_chunks)}")
print(f"Total Embeddings: {len(all_embeddings)}")

print("\nBuilding FAISS Index...")

index = build_index(all_embeddings)

print(f"Vectors Stored: {index.ntotal}")

print("\nKnowledge Base Ready!")

while True:

    query = input("\nAsk a question (or type 'exit'): ")

    if query.lower() == "exit":
        break

    query_embedding = create_embedding(query)

    distances, results = search(
        index,
        query_embedding,
        k=3
    )

    retrieved_chunks = []

    for idx in results:
        retrieved_chunks.append(
            all_chunks[idx]
        )

    context = build_context(
        retrieved_chunks
    )

    print("\nContext Length:")
    print(len(context))

    print("\nTop Results:\n")

    for rank, idx in enumerate(results, start=1):

        chunk = all_chunks[idx]

        print(f"\nResult {rank}")
        print("-" * 50)

        print(f"Similarity Score: {distances[rank-1]:.4f}")
        print(f"Source: {chunk['source']}")
        print(f"Chunk ID: {chunk['chunk_id']}")

        print()
        print(chunk["text"][:500])