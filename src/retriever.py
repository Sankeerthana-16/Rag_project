import faiss
import numpy as np
import pickle
import os
from src.generator import generate_answer
from src.embedding import load_embedding_model


def load_vector_store():

    index = faiss.read_index("data/faiss_index.bin")

    with open("data/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


def retrieve_documents(query, index, chunks, model, k=3):
    print("Model type:", type(model))

    query_embedding = model.encode([query])

    query_embedding = np.array(
        query_embedding,
        dtype="float32"
    )

    distances, indices = index.search(query_embedding, k)

    results = []

    for i in indices[0]:

        if i != -1 and i < len(chunks):
            results.append(chunks[i])

    return results
if __name__ == "__main__":

    # Load embedding model
    model = load_embedding_model()

    # Load FAISS database
    index, chunks = load_vector_store()

    # Ask a question
    question = "What is transformer architecture?"

    # Retrieve relevant chunks
    results = retrieve_documents(
        question,
        index,
        chunks,
        model,
        k=3
    )

    # Display results
    print("\nRetrieved Context:\n")

    for i, chunk in enumerate(results):

        print(f"\nChunk {i+1}:")
        print(chunk)

    # Combine retrieved chunks into context
    context = "\n\n".join(results)

    # Generate final answer using Groq
    answer = generate_answer(question, context)

    # Display final answer
    print("\n========== AI GENERATED ANSWER ==========")
    print(answer)