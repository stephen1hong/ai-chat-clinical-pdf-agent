"""Configuration settings for the Clinical PDF Agent."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"
FAISS_INDEX_PATH = FAISS_INDEX_DIR / "index.faiss"
METADATA_PATH = FAISS_INDEX_DIR / "metadata.json"

# Embedding configuration
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Chunking configuration
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Retrieval configuration
TOP_K_RESULTS = 5

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_TEMPERATURE = 0.7

# Dataset configuration
DATASET_NAME = "zhengyun21/PMC-Patients"
DATASET_SPLIT = "train"
MAX_DOCUMENTS = 1000  # Limit for Phase 1 prototype
