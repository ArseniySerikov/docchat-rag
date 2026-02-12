from typing import List
from langchain_core.documents import Document

from app.infra.loaders import WebLoader
from app.infra.splitters import Chunker
from app.infra.vector_store import VectorStore


class IngestService:
    def __init__(self, loader: WebLoader, chunker: Chunker, store: VectorStore):
        self.loader = loader
        self.chunker = chunker
        self.store = store

    def _enrich_chunks(self, chunks: List[Document], source_url: str) -> List[Document]:
        out: List[Document] = []
        for i, ch in enumerate(chunks):
            ch.metadata = ch.metadata or {}
            ch.metadata["source"] = ch.metadata.get("source", source_url) or source_url
            ch.metadata["title"] = ch.metadata.get("title", "") or ""
            ch.metadata["chunk_index"] = i
            ch.metadata["chunk_id"] = self.store.make_chunk_id(source_url, i, ch.page_content)
            out.append(ch)
        return out

    def ingest_url(self, url: str) -> int:
        docs = self.loader.load(url)
        chunks = self.chunker.split(docs)
        chunks = self._enrich_chunks(chunks, url)
        return self.store.add_documents(chunks)
