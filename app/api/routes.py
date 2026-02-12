from fastapi import APIRouter, HTTPException
from app.core.schemas import IngestUrlRequest, IngestResponse, QueryRequest, QueryResponse, SourceItem
from app.infra.loaders import WebLoader
from app.infra.splitters import Chunker
from app.infra.vector_store import VectorStore
from app.services.ingest_service import IngestService
from app.services.query_service import QueryService

router = APIRouter()
store = VectorStore()
ingest_service = IngestService(WebLoader(), Chunker(), store)
query_service = QueryService(store)


@router.get("/health")
def health():
    return {"status": "ok"}

@router.post("/ingest/url", response_model=IngestResponse)
def ingest(req: IngestUrlRequest):
    try:
        added = ingest_service.ingest_url(req.url)
        return IngestResponse(added_chunks=added, source=req.url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="question is empty")
    k = req.k or 5
    out = query_service.answer(q, k)
    sources = []
    for i, d in enumerate(out["docs"]):
        m = d.metadata or {}
        sources.append(
            SourceItem(
                source=m.get("source", ""),
                title=m.get("title", "") or "",
                chunk_id=m.get("chunk_id", ""),
                score=float(out["sims"][i]) if i < len(out["sims"]) else 0.0,
            )
        )
    return QueryResponse(answer=str(out["answer"]), sources=sources)
