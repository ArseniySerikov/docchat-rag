from typing import List, Tuple
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.config import settings
from app.infra.vector_store import VectorStore
from app.infra.prompts import QA_SYSTEM, QA_USER_TEMPLATE


class QueryService:
    def __init__(self, store: VectorStore):
        self.store = store
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.model_name,
            temperature=settings.temperature,
        )

    @staticmethod
    def _score_to_similarity(distance: float) -> float:
        try:
            return 1.0 / (1.0 + float(distance))
        except Exception:
            return 0.0

    def _build_context(self, docs: List[Document]) -> str:
        parts = []
        total = 0
        for d in docs:
            chunk = d.page_content.strip()
            if not chunk:
                continue
            meta = d.metadata or {}
            src = meta.get("source", "")
            cid = meta.get("chunk_id", "")
            piece = f"[source={src} chunk_id={cid}]\n{chunk}\n"
            if total + len(piece) > settings.max_context_chars:
                break
            parts.append(piece)
            total += len(piece)
        return "\n".join(parts)

    def retrieve(self, question: str, k: int) -> Tuple[List[Document], List[float]]:
        if settings.use_mmr:
            docs = self.store.mmr(question, k=k, fetch_k=settings.fetch_k)
            return docs, [0.0 for _ in docs]

        pairs = self.store.similarity_with_score(question, k=k)
        docs = [d for d, _ in pairs]
        sims = [self._score_to_similarity(dist) for _, dist in pairs]
        return docs, sims

    def answer(self, question: str, k: int) -> dict:
        docs, sims = self.retrieve(question, k)

        if not settings.use_mmr and sims:
            if max(sims) < settings.score_threshold:
                return {"answer": "I don't know based on the provided documents.", "docs": [], "sims": []}

        context = self._build_context(docs)
        user_msg = QA_USER_TEMPLATE.format(context=context, question=question)

        resp = self.llm.invoke([SystemMessage(content=QA_SYSTEM), HumanMessage(content=user_msg)])
        return {"answer": resp.content, "docs": docs, "sims": sims}
