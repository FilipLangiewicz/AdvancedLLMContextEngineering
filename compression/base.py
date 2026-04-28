from abc import ABC, abstractmethod
from enum import Enum
from langchain_core.documents import Document


class CompressionStrategy(str, Enum):
    EXTRACTIVE_FILTER = "extractive_filter"
    HIERARCHICAL_SUMMARY = "hierarchical_summary"
    BRIEF_CONTEXT = "brief_context"


class BaseContextCompressor(ABC):

    @abstractmethod
    def compress(self, query: str, docs: list[Document]) -> list[Document]:
        """
        Takes query and retrieved docs, returns compressed docs.
        """
        ...