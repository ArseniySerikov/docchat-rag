from pydantic import BaseModel, Field
from typing import List, Optional


class IngestUrlRequest(BaseModel):
    url: str


class IngestResponse(BaseModel):
    added_chunks: int
    source: str


class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = None


class SourceItem(BaseModel):
    source: str
    title: str = ""
    chunk_id: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem] = Field(default_factory=list)
