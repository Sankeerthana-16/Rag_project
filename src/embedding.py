from sentence_transformers import SentenceTransformer


# Load embedding model
def load_embedding_model():

    model = SentenceTransformer("all-MiniLM-L6-v2")

    return model


# Convert text chunks into embeddings
def create_embeddings(chunks, model):

    embeddings = model.encode(chunks)

    return embeddings
if __name__ == "__main__":

    from src.pdf_loader import load_pdf
    from src.chuncking import split_text

    pdf_path = "data/attention-all-you-need-Paper.pdf"

    pages = load_pdf(pdf_path)

    full_text = "\n".join(
        page["text"] for page in pages
    )

    chunks = split_text(full_text)

    model = load_embedding_model()

    embeddings = create_embeddings(chunks, model)

    print("Total chunks:", len(chunks))

    print("Embedding shape:", embeddings.shape)

    print("First embedding:")
    print(embeddings[0][:10])