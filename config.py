import os

# Paths
DATA_DIR = "data"
DEFAULT_PDF = os.path.join(DATA_DIR, "attention-all-you-need-Paper.pdf")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.pkl")

# Model configurations
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "openai/gpt-oss-120b"

# Chunking configurations
CHUNK_SIZE = 700
CHUNK_OVERLAP = 150
TOP_K = 3

