"""Query routing logic to determine which agent to use."""
from typing import List, Dict

from app.agents.qa_agent import answer_question
from app.agents.summary_agent import summarize


def route_query(query: str, retrieved_docs: List[Dict]) -> Dict:
    """Route the query to the appropriate agent.

    Args:
        query: User's query string.
        retrieved_docs: List of retrieved documents from FAISS.

    Returns:
        Dict with 'answer' and 'sources' keys from the selected agent.
    """
    query_lower = query.lower()

    # Simple keyword-based routing
    summary_keywords = ['summarize', 'summary', 'overview', 'brief']

    # Check if query indicates summarization
    if any(keyword in query_lower for keyword in summary_keywords):
        print("Routing to summary agent")
        return summarize(retrieved_docs)
    else:
        print("Routing to QA agent")
        return answer_question(query, retrieved_docs)
