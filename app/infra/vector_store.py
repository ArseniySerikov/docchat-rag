import hashlib
from typing import List, Tuple

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from app.core.config import settings


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class VectorStore:
    def __init__(self):
        self._store = Chroma(
            collection_name=settings.collection_name,
            persist_directory=settings.chroma_dir,
            embedding_function=OpenAIEmbeddings(api_key=settings.openai_api_key),
        )

    @staticmethod
    def make_chunk_id(source: str, chunk_index: int, text: str) -> str:
        h = _hash_text(text)[:12]
        return f"{source}::c{chunk_index}::{h}"

    def _existing_chunk_ids(self) -> set[str]:
        try:
            existing = self._store.get(include=["metadatas"])
            metas = existing.get("metadatas", []) or []
            return {m.get("chunk_id") for m in metas if isinstance(m, dict) and m.get("chunk_id")}
        except Exception:
            return set()

    def add_documents(self, docs: List[Document]) -> int:
        if not docs:
            return 0

        existing = self._existing_chunk_ids()
        fresh: List[Document] = []
        for d in docs:
            cid = (d.metadata or {}).get("chunk_id")
            if cid and cid not in existing:
                fresh.append(d)

        if not fresh:
            return 0

        self._store.add_documents(fresh)
        self._store.persist()
        return len(fresh)

    def similarity_with_score(self, query: str, k: int) -> List[Tuple[Document, float]]:
        return self._store.similarity_search_with_score(query, k=k)

    def mmr(self, query: str, k: int, fetch_k: int) -> List[Document]:
        return self._store.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)
