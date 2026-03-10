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

All endpoints are served under the `/api` prefix.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Railway health check — returns `{"status": "ok"}` |
| GET | `/api/info` | Vector store metadata |
| GET | `/api/search?query=...` | Semantic document search |
| POST | `/api/chat` | RAG-powered Q&A |

### Response Format

```json
// /api/health (flat)
{"status": "ok", "vector_store_loaded": true, ...}

// All other endpoints (envelope)
{"success": true, "data": { ... }}
```

## Deployment

### Backend → Railway
1. Connect your GitHub repo to Railway
2. Set environment variables in Railway dashboard (see `.env.example`)
3. Railway auto-runs: `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`

### Frontend → Vercel
1. Set `VITE_BACKEND_URL=https://your-backend.railway.app` in Vercel env vars
2. Deploy the `frontend/` directory

### Environment Variables

| Variable | Where | Description |
|----------|-------|-------------|
| `BACKEND_URL` | Railway | Your Railway backend URL |
| `FRONTEND_URL` | Railway | Your Vercel frontend URL (for CORS) |
| `PORT` | Railway | Set automatically by Railway |
| `VITE_BACKEND_URL` | Vercel | Your Railway backend URL |

## Tech Stack

- **Backend**: FastAPI + FAISS (OpenRAG) + Qwen2.5-0.5B-Instruct
- **Embeddings**: paraphrase-multilingual-mpnet-base-v2 (SentenceTransformers)
- **Frontend**: React 18 + Vite + Tailwind CSS + Radix UI
