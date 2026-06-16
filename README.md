# Clinical PDF Agent

AI chat system for clinical documents with RAG-based Q&A and summarization capabilities. Built with FastAPI, FAISS, and OpenAI.

## Features

- **RAG-based Question Answering**: Ask questions about clinical documents and get accurate, cited answers
- **Document Summarization**: Generate concise summaries of clinical information
- **PMC-Patients Dataset**: Pre-configured to use fully open clinical data from Hugging Face
- **PDF Support**: Parse and process PDF documents (ready for future integration)
- **Fast Retrieval**: FAISS-based vector search for efficient document retrieval
- **REST API**: Easy-to-use FastAPI endpoints

## Architecture

```
User Query → Embedding → FAISS Retrieval → Router → Agent (QA/Summary) → Response
```

- **Embeddings**: BAAI/bge-small-en-v1.5 (sentence-transformers)
- **Vector Store**: FAISS (CPU version)
- **Chunking**: LangChain RecursiveCharacterTextSplitter
- **LLM**: OpenAI GPT-3.5-turbo
- **Framework**: FastAPI

## Installation

### Prerequisites

- Python 3.8+
- OpenAI API key

### Setup

1. Clone the repository:
```bash
cd ai-chat-clinical-pdf-agent
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-key-here
```

## Usage

### Step 1: Build the FAISS Index

Load the PMC-Patients dataset and build the vector index:

```bash
python scripts/build_index.py
```

This will:
- Download PMC-Patients dataset from Hugging Face (first 1000 documents by default)
- Chunk documents into 512-character segments
- Generate embeddings using BAAI/bge-small-en-v1.5
- Build and save FAISS index to `data/faiss_index/`

**Note**: This process may take several minutes on first run.

### Step 2: Start the API Server

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

### Step 3: Make Requests

#### Question Answering

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are common symptoms of diabetes?"}'
```

#### Summarization

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize diabetes cases"}'
```

#### Response Format

```json
{
  "answer": "According to the retrieved documents...",
  "sources": ["PMID:12345678", "PMID:87654321"]
}
```

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Endpoints

#### `POST /chat`
Main endpoint for Q&A and summarization.

**Request Body:**
```json
{
  "query": "Your question or summarization request"
}
```

**Response:**
```json
{
  "answer": "Generated response",
  "sources": ["List of source documents"]
}
```

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "retriever_loaded": true,
  "index_size": 5000
}
```

## Configuration

Edit `config.py` to customize:

- `EMBEDDING_MODEL_NAME`: Embedding model (default: BAAI/bge-small-en-v1.5)
- `CHUNK_SIZE`: Text chunk size (default: 512)
- `CHUNK_OVERLAP`: Chunk overlap (default: 50)
- `TOP_K_RESULTS`: Number of documents to retrieve (default: 5)
- `OPENAI_MODEL`: OpenAI model (default: gpt-3.5-turbo)
- `OPENAI_TEMPERATURE`: Generation temperature (default: 0.7)
- `MAX_DOCUMENTS`: Dataset size limit (default: 1000)

## Project Structure

```
clinical-pdf-agent/
├── requirements.txt        # Python dependencies
├── config.py              # Configuration settings
├── .env                   # Environment variables (not in git)
├── .env.example           # Environment template
├── .gitignore            # Git ignore rules
├── README.md             # This file
├── scripts/
│   └── build_index.py    # Index building script
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   ├── embeddings.py     # Embedding model
│   ├── retriever.py      # FAISS retrieval
│   ├── chunking.py       # Document chunking
│   ├── pdf_parser.py     # PDF text extraction
│   ├── router.py         # Query routing logic
│   └── agents/
│       ├── __init__.py
│       ├── qa_agent.py       # Question answering
│       └── summary_agent.py  # Summarization
└── data/
    └── faiss_index/      # Generated FAISS index (not in git)
```

## Example Queries

### Question Answering

```bash
# Symptom questions
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the symptoms of hypertension?"}'

# Treatment questions
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What treatments are available for type 2 diabetes?"}'

# Diagnosis questions
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How is pneumonia diagnosed?"}'
```

### Summarization

```bash
# General summary
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize cardiovascular disease cases"}'

# Focused summary
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Give me an overview of cancer treatment approaches"}'
```

## Routing Logic

The system automatically routes queries to the appropriate agent:

- **Summary Agent**: Triggered by keywords like "summarize", "summary", "overview", "brief"
- **QA Agent**: Used for all other queries (default)

## PDF Processing (Future Enhancement)

The system includes PDF parsing capabilities (`app/pdf_parser.py`) ready for integration:

```python
from app.pdf_parser import extract_text_from_pdf

# Extract text from PDF
text = extract_text_from_pdf("path/to/document.pdf")

# Add to index building pipeline
# (Requires modification to build_index.py)
```

## Development

### Running Tests

```bash
# Test the API is running
curl http://localhost:8000/health

# Test QA functionality
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is diabetes?"}'

# Test summarization
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize diabetes information"}'
```

### Rebuilding the Index

If you modify chunking parameters or want to use a different dataset:

```bash
# Delete existing index
rm -rf data/faiss_index/

# Rebuild
python scripts/build_index.py
```

## Phase 1 Limitations

This is a prototype implementation with intentional limitations:

- **Simple Routing**: Keyword-based (not ML-based)
- **No PDF Upload UI**: Parsing code ready, but no upload endpoint
- **No Authentication**: Open API endpoints
- **No Rate Limiting**: Suitable for development only
- **No Conversation History**: Each query is independent
- **Fixed Dataset**: PMC-Patients only (configurable)
- **Basic Error Handling**: Minimal validation

## Future Enhancements

- [ ] PDF upload endpoint
- [ ] Conversation memory/context
- [ ] Multi-turn dialogue support
- [ ] User authentication
- [ ] Rate limiting
- [ ] Improved routing (ML-based)
- [ ] Support for multiple datasets
- [ ] Streaming responses
- [ ] Batch processing
- [ ] Advanced error handling

## Troubleshooting

### "FAISS index not found"

Run the index building script first:
```bash
python scripts/build_index.py
```

### "OPENAI_API_KEY not set"

Make sure you've created a `.env` file with your API key:
```bash
cp .env.example .env
# Edit .env and add your key
```

### "Dataset not available"

The PMC-Patients dataset should download automatically. If it fails:
1. Check your internet connection
2. Try again (datasets are cached after first download)
3. Use a different dataset or custom PDFs

### Port 8000 already in use

Change the port in the command:
```bash
uvicorn app.main:app --reload --port 8001
```

## License

This project is for educational and research purposes. Ensure compliance with:
- PMC-Patients dataset license
- OpenAI API terms of service
- Any applicable healthcare data regulations

## Citation

If you use the PMC-Patients dataset, please cite:
```
@article{pmc-patients,
  title={PMC-Patients: A Large-scale Dataset of Patient Summaries},
  author={...},
  journal={...},
  year={2023}
}
```

## Support

For issues and questions:
1. Check this README
2. Review API documentation at `/docs`
3. Check configuration in `config.py`
4. Verify environment variables in `.env`

## Acknowledgments

- PMC-Patients dataset from Hugging Face
- BAAI for bge-small-en-v1.5 embeddings
- OpenAI for GPT models
- FAISS by Facebook Research
- FastAPI framework
