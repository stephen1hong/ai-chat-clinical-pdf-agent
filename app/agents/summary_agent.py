"""Summarization agent using OpenAI API."""
from typing import List, Dict
from openai import OpenAI

import config


def summarize(docs: List[Dict]) -> Dict:
    """Summarize the provided documents.

    Args:
        docs: List of documents with 'text' and 'source' keys.

    Returns:
        Dict with 'answer' and 'sources' keys.
    """
    if not config.OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY not set. Please add it to your .env file."
        )

    # Build text from documents
    doc_parts = []
    sources = []

    for i, doc in enumerate(docs):
        doc_parts.append(f"[Document {i+1}]")
        doc_parts.append(doc['text'])
        doc_parts.append("")  # Empty line between docs

        # Extract source information
        source = doc.get('source', 'Unknown')
        if source not in sources:
            sources.append(source)

    documents_text = "\n".join(doc_parts)

    # Build prompt
    system_prompt = """You are a helpful medical AI assistant specialized in summarizing clinical information.

Instructions:
- Provide a clear, concise summary of the key clinical information from the documents
- Focus on important medical facts, symptoms, diagnoses, and treatments
- Organize the summary in a logical structure
- Be accurate and avoid adding information not present in the documents
- Keep the summary concise but comprehensive"""

    user_prompt = f"""Please summarize the following clinical documents:

{documents_text}

Provide a comprehensive summary of the key clinical information."""

    # Call OpenAI API
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    try:
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=config.OPENAI_TEMPERATURE,
            max_tokens=700
        )

        summary = response.choices[0].message.content.strip()

        return {
            "answer": summary,
            "sources": sources
        }

    except Exception as e:
        return {
            "answer": f"Error generating summary: {str(e)}",
            "sources": []
        }
