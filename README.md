# RAG Document Assistant

A Retrieval-Augmented Generation (RAG) assistant built using Streamlit, Sentence Transformers, FAISS, and Groq API.

## Features
- **PDF Ingestion & Text Extraction**: Extracts text from PDF files using PyMuPDF.
- **Text Chunking**: Configurable chunk size and overlap for context preservation.
- **Embeddings & Vector Search**: Generates embeddings using `sentence-transformers/all-MiniLM-L6-v2` and searches with FAISS.
- **LLM Question Answering**: Generates answers based on retrieved context using Groq (`openai/gpt-oss-120b`).
- **Interactive UI**: Upload any PDF directly or query the default document (*Attention Is All You Need*).

## Setup & Running

1. **Activate virtual environment**:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. **Configure API Key**:
   Ensure `.env` contains your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

3. **Run the Streamlit application**:
   ```powershell
   streamlit run app.py
   ```
   Or using the virtual environment's Python/Streamlit:
   ```powershell
   .\venv\Scripts\streamlit.exe run app.py
   ```

