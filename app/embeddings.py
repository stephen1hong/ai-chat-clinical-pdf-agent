"""Embedding generation using sentence-transformers."""
from typing import List, Union
from sentence_transformers import SentenceTransformer
import numpy as np

import config


class EmbeddingModel:
    """Wrapper for sentence-transformers embedding model."""

    def __init__(self, model_name: str = None):
        """Initialize the embedding model.

        Args:
            model_name: Name of the sentence-transformers model to use.
                       Defaults to config.EMBEDDING_MODEL_NAME.
        """
        model_name = model_name or config.EMBEDDING_MODEL_NAME
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("Embedding model loaded successfully")

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """Encode text(s) into embeddings.

        Args:
            texts: Single text string or list of text strings to encode.
            batch_size: Batch size for encoding.

        Returns:
            Numpy array of embeddings with shape (n_texts, embedding_dim).
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True
        )

        return embeddings


# Global instance (lazy-loaded)
_embedding_model = None


def get_embedding_model() -> EmbeddingModel:
    """Get or create the global embedding model instance.

    Returns:
        Singleton EmbeddingModel instance.
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model
