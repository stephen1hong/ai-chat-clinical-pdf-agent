"""Text chunking utilities using LangChain text splitters."""
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def chunk_documents(
    texts: List[str],
    metadatas: List[Dict] = None,
    chunk_size: int = None,
    chunk_overlap: int = None
) -> tuple[List[str], List[Dict]]:
    """Chunk documents into smaller pieces for embedding.

    Args:
        texts: List of text strings to chunk.
        metadatas: Optional list of metadata dicts (one per text).
        chunk_size: Maximum size of each chunk in characters.
                   Defaults to config.CHUNK_SIZE.
        chunk_overlap: Number of overlapping characters between chunks.
                      Defaults to config.CHUNK_OVERLAP.

    Returns:
        Tuple of (chunked_texts, chunked_metadatas).
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunked_texts = []
    chunked_metadatas = []

    # Process each document
    for i, text in enumerate(texts):
        chunks = text_splitter.split_text(text)
        chunked_texts.extend(chunks)

        # Propagate metadata to all chunks
        if metadatas:
            base_metadata = metadatas[i].copy()
            for chunk_idx, chunk in enumerate(chunks):
                chunk_metadata = base_metadata.copy()
                chunk_metadata['chunk_index'] = chunk_idx
                chunk_metadata['text'] = chunk
                chunked_metadatas.append(chunk_metadata)
        else:
            # Create basic metadata if none provided
            for chunk_idx, chunk in enumerate(chunks):
                chunked_metadatas.append({
                    'doc_index': i,
                    'chunk_index': chunk_idx,
                    'text': chunk
                })

    print(f"Chunked {len(texts)} documents into {len(chunked_texts)} chunks")
    return chunked_texts, chunked_metadatas
