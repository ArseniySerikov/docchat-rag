# DocChat RAG — AI Q&A over Web Documents (FastAPI)

DocChat RAG is a small production-style backend that turns web pages into a searchable knowledge base and answers user questions with references to the sources.

**What it does:**
1) Ingests a URL (downloads a page, extracts text, splits into chunks)
2) Creates embeddings and stores them in a vector database (Chroma)
3) Retrieves the most relevant chunks for a user question (top-k)
4) Uses an LLM to generate an answer and returns **citations/sources**

**Why it matters (business value):**
- Enables “chat with documentation / articles / internal pages”
- Faster onboarding & support: answers are grounded in the ingested content
- Clear traceability: every answer comes with sources

---

## Key features
- FastAPI REST API + Swagger UI
- URL ingestion (HTML parsing → chunking)
- Embeddings + vector search (Chroma)
- Answer generation with returned source references
- Config via `.env` / `.env.example`
- Docker support (Dockerfile + docker compose)

---

## Tech stack
Python, FastAPI, LangChain, Chroma, OpenAI API, Docker, Git

---

## Quick start (local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger UI:
- http://127.0.0.1:8000/docs

API examples
Ingest URL

POST /ingest/url

```bash
curl -X POST "http://127.0.0.1:8000/ingest/url" \
  -H "Content-Type: application/json" \
  -d '{"url":"https://en.wikipedia.org/wiki/Prompt_engineering"}'
```

Query
POST /query
```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is prompt engineering? Answer with citations.","k":5}'
```
## Demo (2-step)
1) Ingest: POST `/ingest/url` with a URL  
2) Ask: POST `/query` with `question` and optional `k`  
You get an answer + a list of source chunks used for grounding.

## Run with Docker
```bash
cp .env.example .env
# set OPENAI_API_KEY in .env
docker compose up --build

## Notes / limitations
- URL ingestion works best for articles/docs pages (not heavy JS websites).
- Current store: Chroma (local). Can be swapped to Postgres + pgvector.

## Roadmap (next improvements)
- PDF ingestion
- Reranking (optional)
- Postgres + pgvector backend
- Logging/metrics and basic auth