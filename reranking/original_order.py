from langchain_core.documents import Document
from .base import BaseReranker


class OriginalOrderReranker(BaseReranker):

    def rerank(self, documents: list[Document]) -> list[Document]:
        return documents