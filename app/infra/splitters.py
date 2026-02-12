from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.core.config import settings


class Chunker:
    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size or settings.chunk_size,
            chunk_overlap=chunk_overlap or settings.chunk_overlap,
            add_start_index=True,
        )

    def split(self, docs: List[Document]) -> List[Document]:
        return self.splitter.split_documents(docs)
