# ASKVUTA

An intelligent Vietnamese RAG chatbot for discovering information about Vũng Tàu city.

## OpenRag Integration

This project integrates OpenRag as the core framework for building the embedding and retrieval pipeline.
- Utilized OpenRag for document ingestion, embedding generation, and indexing.
- Built a FAISS-based vector search system for efficient semantic retrieval.
- Processed crawled Vietnamese articles into structured embeddings for downstream querying.
- Enabled a modular and scalable RAG pipeline with support for flexible retrieval strategies and embedding models.
- OpenRag repository: [![OpenRag](https://img.shields.io/badge/OpenRag-GitHub-blue)](https://github.com/incidentfox/OpenRag)
## Project Structure

```
ASKVUTA/
├── backend/
│   ├── app/                    # FastAPI application
│   │   ├── api/routes.py       # API endpoints (/health, /search, /chat)
│   │   ├── core/               # Config & VectorStore
│   │   ├── models/             # Pydantic request/response models
│   │   ├── services/           # LLM, Search, RAG services
│   │   └── utils/              # Prompt builder
│   ├── crawler/                # Web scraper for Vũng Tàu articles
│   └── loaders/                # OpenRAG selective import loader
│
├── frontend/                   # React + Vite + Tailwind UI
│
├── data/
│   ├── dataset/                # Crawled JSON articles by topic
│   └── embeddings/             # FAISS vector store (vungtau_knowledge.pkl)
│
├── scripts/
│   └── build_rag.py            # Build FAISS index from dataset
│
├── OpenRag/                    # OpenRAG library (submodule)
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Clone OpenRag 
```bash
# Clone the repo
git clone https://github.com/incidentfox/OpenRag.git
cd OpenRag
# Install requirements
pip install -r requirements.txt
```

### 3. Crawl articles (optional – dataset already included)
```bash
python backend/crawler/crawl_articles.py
```

### 4. Build the RAG index
```bash
python scripts/build_rag.py
```

### 5. Run the backend
```bash
python backend/app/main.py
# or from backend/ directory:
uvicorn app.main:app --reload
```

### 5. Run the frontend
```bash
cd frontend
npm install
npm run dev
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server health check |
| GET | `/info` | Vector store info |
| GET | `/search?query=...` | Semantic document search |
| POST | `/chat` | RAG-powered Q&A |

## Tech Stack

- **Backend**: FastAPI + FAISS (OpenRAG) + Arcee-VyLinh-3B
- **Embeddings**: paraphrase-multilingual-mpnet-base-v2 (SentenceTransformers)
- **Frontend**: React 18 + Vite + Tailwind CSS + Radix UI
