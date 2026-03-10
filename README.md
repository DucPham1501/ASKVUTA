# ASKVUTA

An intelligent Vietnamese RAG chatbot for discovering information about Vũng Tàu city.

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

### 2. Crawl articles (optional – dataset already included)
```bash
python backend/crawler/crawl_articles.py
```

### 3. Build the RAG index
```bash
python scripts/build_rag.py
```

### 4. Run the backend
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

- **Backend**: FastAPI + FAISS (OpenRAG) + Qwen2.5-0.5B-Instruct
- **Embeddings**: paraphrase-multilingual-mpnet-base-v2 (SentenceTransformers)
- **Frontend**: React 18 + Vite + Tailwind CSS + Radix UI
