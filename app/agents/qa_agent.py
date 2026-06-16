"""Question-answering agent using OpenAI API."""
from typing import List, Dict
from openai import OpenAI

import config


def answer_question(query: str, context_docs: List[Dict]) -> Dict:
    """Answer a question using retrieved context documents.

    Args:
        query: User's question.
        context_docs: List of retrieved documents with 'text' and 'source' keys.

    Returns:
        Dict with 'answer' and 'sources' keys.
    """
    if not config.OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY not set. Please add it to your .env file."
        )

    # Build context from retrieved documents
    context_parts = []
    sources = []

    for i, doc in enumerate(context_docs):
        context_parts.append(f"[Document {i+1}]")
        context_parts.append(doc['text'])
        context_parts.append("")  # Empty line between docs

        # Extract source information
        source = doc.get('source', 'Unknown')
        if source not in sources:
            sources.append(source)

    context_text = "\n".join(context_parts)

    # Build prompt
    system_prompt = """You are a helpful medical AI assistant. Your role is to answer questions based on the provided clinical documents.

Instructions:
- Answer the question using ONLY the information from the provided documents
- Be accurate and concise
- If the documents don't contain enough information to answer the question, say so
- Cite document numbers when possible (e.g., "According to Document 2...")
- Focus on clinical accuracy and clarity"""

    user_prompt = f"""Context documents:
{context_text}

Question: {query}

Please provide a clear, accurate answer based on the context documents above."""

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
            max_tokens=500
        )

        answer = response.choices[0].message.content.strip()

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        return {
            "answer": f"Error generating answer: {str(e)}",
            "sources": []
        }
