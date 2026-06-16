"""FAISS-based document retrieval."""
from typing import List, Dict
import json
import faiss
import numpy as np
from pathlib import Path

import config
from app.embeddings import get_embedding_model


class FAISSRetriever:
    """FAISS-based retriever for document chunks."""

    def __init__(self, index_path: str = None, metadata_path: str = None):
        """Initialize the FAISS retriever.

        Args:
            index_path: Path to the FAISS index file.
                       Defaults to config.FAISS_INDEX_PATH.
            metadata_path: Path to the metadata JSON file.
                          Defaults to config.METADATA_PATH.
        """
        self.index_path = Path(index_path or config.FAISS_INDEX_PATH)
        self.metadata_path = Path(metadata_path or config.METADATA_PATH)

        self.index = None
        self.metadata = None
        self.embedding_model = get_embedding_model()

        self._load_index()

    def _load_index(self):
        """Load the FAISS index and metadata from disk."""
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {self.index_path}. "
                "Please run scripts/build_index.py first."
            )

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found at {self.metadata_path}. "
                "Please run scripts/build_index.py first."
            )

        print(f"Loading FAISS index from {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))

        print(f"Loading metadata from {self.metadata_path}")
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        print(f"Index loaded: {self.index.ntotal} vectors, {len(self.metadata)} metadata entries")

    def retrieve(self, query: str, k: int = None) -> List[Dict]:
        """Retrieve top-k most relevant documents for a query.

        Args:
            query: Query string.
            k: Number of documents to retrieve.
               Defaults to config.TOP_K_RESULTS.

        Returns:
            List of dicts with 'text', 'source', and 'score' keys.
        """
        k = k or config.TOP_K_RESULTS
        k = min(k, self.index.ntotal)  # Don't request more than available

        # Encode query
        query_embedding = self.embedding_model.encode(query)

        # Reshape for FAISS (needs 2D array)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')

        # Search index
        distances, indices = self.index.search(query_embedding, k)

        # Retrieve metadata for results
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['score'] = float(distance)
                results.append(result)

        return results


# Global instance (lazy-loaded)
_retriever = None


def get_retriever() -> FAISSRetriever:
    """Get or create the global retriever instance.

    Returns:
        Singleton FAISSRetriever instance.
    """
    global _retriever
    if _retriever is None:
        _retriever = FAISSRetriever()
    return _retriever
