"""FastAPI application for the Clinical PDF Agent."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import uvicorn

from app.retriever import get_retriever
from app.router import route_query


# Request/Response models
class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    query: str = Field(..., min_length=1, description="User's question or request")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    answer: str = Field(..., description="Generated answer or summary")
    sources: List[str] = Field(..., description="List of source documents used")


# Initialize FastAPI app
app = FastAPI(
    title="Clinical PDF Agent",
    description="AI chat system for clinical documents with RAG-based Q&A and summarization",
    version="1.0.0"
)


# Global retriever instance
retriever = None


@app.on_event("startup")
async def startup_event():
    """Initialize resources on application startup."""
    global retriever
    print("Starting up Clinical PDF Agent...")
    try:
        retriever = get_retriever()
        print("Application ready!")
    except Exception as e:
        print(f"Error during startup: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Clinical PDF Agent",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/chat (POST)",
            "health": "/health (GET)",
            "docs": "/docs (GET)"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "retriever_loaded": retriever is not None,
        "index_size": retriever.index.ntotal if retriever else 0
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint for Q&A and summarization.

    Args:
        request: ChatRequest with user query.

    Returns:
        ChatResponse with answer and sources.
    """
    try:
        # Retrieve relevant documents
        print(f"Processing query: {request.query}")
        retrieved_docs = retriever.retrieve(request.query)
        print(f"Retrieved {len(retrieved_docs)} documents")

        if not retrieved_docs:
            raise HTTPException(
                status_code=404,
                detail="No relevant documents found for the query"
            )

        # Route to appropriate agent
        result = route_query(request.query, retrieved_docs)

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"]
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
