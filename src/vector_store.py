import faiss
import numpy as np
import pickle
import os


def create_vector_store(embeddings, chunks):

    os.makedirs("data", exist_ok=True)

    embeddings = np.array(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(
        index,
        "data/faiss_index.bin"
    )

    # Save chunks
    with open("data/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("FAISS database created successfully!")
    print("Total vectors stored:", index.ntotal)

    return index


def search_vectors(index, query_embedding, k=3):

    query_embedding = np.array(
        [query_embedding],
        dtype="float32"
    )

    distances, indices = index.search(
        query_embedding, k
    )

    return distances, indices
if __name__ == "__main__":

    from src.pdf_loader import load_pdf
    from src.chuncking import split_text
    from src.embedding import (
        load_embedding_model,
        create_embeddings
    )

    # Load PDF
    pages = load_pdf(
        "data/attention-all-you-need-Paper.pdf"
    )

    # Extract text
    full_text = "\n".join(
        page["text"] for page in pages
    )

    # Create chunks
    chunks = split_text(full_text)

    # Generate embeddings
    model = load_embedding_model()
    embeddings = create_embeddings(chunks, model)

    # Create FAISS database
    index = create_vector_store(embeddings,chunks)

    print("FAISS database created successfully!")
    print("Total vectors stored:", index.ntotal)