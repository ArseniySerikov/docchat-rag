from typing import List
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document


class WebLoader:
    def load(self, url: str) -> List[Document]:
        loader = WebBaseLoader(web_paths=(url,))
        docs = loader.load()
        for d in docs:
            d.metadata = d.metadata or {}
            d.metadata["source"] = url
        return docs
