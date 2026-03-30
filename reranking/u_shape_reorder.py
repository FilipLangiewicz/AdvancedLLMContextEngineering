from langchain_core.documents import Document
from .base import BaseReranker


class UShapeReranker(BaseReranker):

    def rerank(self, documents: list[Document]) -> list[Document]:
        if len(documents) <= 2:
            return documents

        result = [None] * len(documents)
        left_ptr = 0
        right_ptr = len(documents) - 1

        for i, doc in enumerate(documents):
            if i % 2 == 0:
                result[left_ptr] = doc
                left_ptr += 1
            else:
                result[right_ptr] = doc
                right_ptr -= 1

        return result