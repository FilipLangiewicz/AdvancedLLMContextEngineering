from abc import ABC, abstractmethod
from enum import Enum
from langchain_core.documents import Document


class RerankingStrategy(str, Enum):
    ORIGINAL_ORDER = "original_order"
    U_SHAPE_REORDER = "u_shape_reorder"


class BaseReranker(ABC):

    @abstractmethod
    def rerank(self, documents: list[Document]) -> list[Document]:
        """
        Accepts a list of documents sorted by similarity score (descending)
        and returns a reordered list according to the chosen strategy.
        """
        pass