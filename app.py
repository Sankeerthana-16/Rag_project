

import streamlit as st
import numpy as np
import faiss
import pymupdf

from src.embedding import load_embedding_model, create_embeddings
from src.chuncking import split_text
from src.retriever import load_vector_store, retrieve_documents
from src.generator import generate_answer

st.set_page_config(page_title="RAG Document Assistant", layout="wide")

st.title("📄 RAG Document Assistant")
st.write("Upload a PDF to ask questions about it, or query the default document (*Attention Is All You Need*).")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

# Load default models and vector store
@st.cache_resource
def load_resources():
    model = load_embedding_model()
    index, chunks = load_vector_store()
    return model, index, chunks

model, default_index, default_chunks = load_resources()

# Cache processing of uploaded PDFs
@st.cache_resource
def process_uploaded_pdf(file_bytes, filename):
    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    pages_text = []
    for page in doc:
        text = page.get_text("text").strip()
        if text:
            pages_text.append(text)
    doc.close()

    full_text = "\n".join(pages_text)
    if not full_text.strip():
        return None, []

    chunks = split_text(full_text)
    embeddings = create_embeddings(chunks, model)
    embeddings = np.array(embeddings, dtype="float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, chunks

if uploaded_file is not None:
    with st.spinner("Processing uploaded PDF..."):
        active_index, active_chunks = process_uploaded_pdf(uploaded_file.getvalue(), uploaded_file.name)
    if active_index is None or not active_chunks:
        st.error("Could not extract any text from the uploaded PDF. Please make sure it contains selectable text.")
        active_index, active_chunks = default_index, default_chunks
    else:
        st.success(f"Loaded '{uploaded_file.name}' with {len(active_chunks)} chunks.")
else:
    active_index, active_chunks = default_index, default_chunks
    st.info("Currently using default document: *Attention Is All You Need*")

# User question
question = st.text_input("Ask a question about your document:")

if st.button("Generate Answer"):

    if question.strip():
        with st.spinner("Searching document and generating answer..."):
            results = retrieve_documents(
                question,
                active_index,
                active_chunks,
                model,
                k=3
            )

            context = "\n\n".join(results)
            answer = generate_answer(question, context)

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Retrieved Context"):
            st.write(context)

    else:
        st.warning("Please enter a question.")
