"""Script to build FAISS index from PMC-Patients dataset."""
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import faiss
from datasets import load_dataset

import config
from app.embeddings import get_embedding_model
from app.chunking import chunk_documents


def load_pmc_patients_dataset(max_docs: int = None):
    """Load PMC-Patients dataset from Hugging Face.

    Args:
        max_docs: Maximum number of documents to load (for testing).

    Returns:
        Tuple of (texts, metadatas).
    """
    print(f"Loading dataset: {config.DATASET_NAME}")
    print(f"Split: {config.DATASET_SPLIT}")

    try:
        dataset = load_dataset(config.DATASET_NAME, split=config.DATASET_SPLIT)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nNote: If the dataset is not available, you can:")
        print("1. Use a different dataset")
        print("2. Use custom PDFs with app.pdf_parser")
        raise

    max_docs = max_docs or config.MAX_DOCUMENTS
    if max_docs:
        dataset = dataset.select(range(min(max_docs, len(dataset))))

    print(f"Loaded {len(dataset)} documents")

    texts = []
    metadatas = []

    # Extract text and metadata from dataset
    for i, item in enumerate(dataset):
        # PMC-Patients structure: patient cases with sections
        # Combine relevant text fields
        text_parts = []

        # Extract patient information
        if 'patient' in item and item['patient']:
            text_parts.append(f"Patient: {item['patient']}")

        # Extract article title/context
        if 'title' in item and item['title']:
            text_parts.append(f"Title: {item['title']}")

        # Extract main content
        if 'patient_case' in item and item['patient_case']:
            text_parts.append(item['patient_case'])

        # Combine all text
        full_text = "\n\n".join(text_parts)

        if full_text.strip():
            texts.append(full_text)

            # Create metadata
            metadata = {
                'source': f"PMC-Patients-{i}",
                'doc_index': i
            }

            # Add article ID if available
            if 'pmid' in item and item['pmid']:
                metadata['pmid'] = item['pmid']
                metadata['source'] = f"PMID:{item['pmid']}"

            metadatas.append(metadata)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} documents...")

    print(f"Extracted text from {len(texts)} documents")
    return texts, metadatas


def build_faiss_index(embeddings: np.ndarray):
    """Build FAISS index from embeddings.

    Args:
        embeddings: Numpy array of embeddings (n_docs, embedding_dim).

    Returns:
        FAISS index.
    """
    dimension = embeddings.shape[1]
    print(f"Building FAISS index with dimension {dimension}")

    # Use IndexFlatL2 for exact nearest neighbor search
    # For larger datasets, consider IndexIVFFlat for approximate search
    index = faiss.IndexFlatL2(dimension)

    # Add embeddings to index
    embeddings = embeddings.astype('float32')
    index.add(embeddings)

    print(f"Index built with {index.ntotal} vectors")
    return index


def main():
    """Main function to build and save FAISS index."""
    print("=" * 60)
    print("Building FAISS Index for Clinical PDF Agent")
    print("=" * 60)

    # Create output directory
    config.FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load dataset
    print("\n[Step 1/5] Loading PMC-Patients dataset...")
    texts, metadatas = load_pmc_patients_dataset()

    if not texts:
        print("Error: No documents loaded. Exiting.")
        return

    # Step 2: Chunk documents
    print("\n[Step 2/5] Chunking documents...")
    chunked_texts, chunked_metadatas = chunk_documents(texts, metadatas)

    # Step 3: Generate embeddings
    print("\n[Step 3/5] Generating embeddings...")
    embedding_model = get_embedding_model()
    embeddings = embedding_model.encode(chunked_texts, batch_size=32)
    print(f"Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")

    # Step 4: Build FAISS index
    print("\n[Step 4/5] Building FAISS index...")
    index = build_faiss_index(embeddings)

    # Step 5: Save index and metadata
    print("\n[Step 5/5] Saving index and metadata...")

    # Save FAISS index
    faiss.write_index(index, str(config.FAISS_INDEX_PATH))
    print(f"FAISS index saved to: {config.FAISS_INDEX_PATH}")

    # Save metadata
    with open(config.METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(chunked_metadatas, f, indent=2, ensure_ascii=False)
    print(f"Metadata saved to: {config.METADATA_PATH}")

    # Print summary
    print("\n" + "=" * 60)
    print("Index Building Complete!")
    print("=" * 60)
    print(f"Total documents: {len(texts)}")
    print(f"Total chunks: {len(chunked_texts)}")
    print(f"Index size: {index.ntotal} vectors")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print("\nYou can now start the API server with:")
    print("  uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()
