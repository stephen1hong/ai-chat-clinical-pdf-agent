"""FastAPI application for the Clinical PDF Agent."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import uvicorn
import os
import subprocess
import sys
import asyncio
import threading
from pathlib import Path

from app.retriever import get_retriever
from app.router import route_query
import config


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

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://voranex.ai",
        "http://voranex.ai",
        "https://stephen1hong.github.io",
        "http://localhost:3000",
        "http://localhost:5500",
        "http://127.0.0.1:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global retriever instance
retriever = None
index_building = False


def build_index_background():
    """Build FAISS index in background thread."""
    global retriever, index_building
    index_building = True

    try:
        print("FAISS index not found. Building index from PMC-Patients dataset...")
        print(f"Using MAX_DOCUMENTS={config.MAX_DOCUMENTS} (this may take a few minutes)...")

        # Ensure data directory exists
        config.FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

        # Run the build_index script
        build_script = Path(__file__).parent.parent / "scripts" / "build_index.py"
        result = subprocess.run(
            [sys.executable, str(build_script)],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        print("Index built successfully!")

        # Load the retriever
        retriever = get_retriever()
        print("Application ready!")

    except subprocess.CalledProcessError as e:
        print(f"Error building index: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error during index building: {e}")
    finally:
        index_building = False


@app.on_event("startup")
async def startup_event():
    """Initialize resources on application startup."""
    global retriever
    print("Starting up Clinical PDF Agent...")

    # Check if FAISS index exists
    if not config.FAISS_INDEX_PATH.exists() or not config.METADATA_PATH.exists():
        # Build index in background thread so server can start immediately
        thread = threading.Thread(target=build_index_background, daemon=True)
        thread.start()
        print("Index building started in background. Server is ready to accept health checks.")
    else:
        # Load existing index
        print("FAISS index found. Loading...")
        try:
            retriever = get_retriever()
            print("Application ready!")
        except Exception as e:
            print(f"Error loading retriever: {e}")
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
    """Health check endpoint - returns healthy even during index building."""
    if retriever is None:
        state = "building" if index_building else "starting"
        message = "Index is being built, please wait..." if index_building else "Starting up..."

        return {
            "status": "healthy",
            "state": state,
            "message": message,
            "retriever_loaded": False,
            "index_size": 0
        }

    try:
        index_size = retriever.index.ntotal if hasattr(retriever, 'index') else 0
    except:
        index_size = 0

    return {
        "status": "healthy",
        "state": "ready",
        "message": "Ready to answer questions",
        "retriever_loaded": True,
        "index_size": index_size
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
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
