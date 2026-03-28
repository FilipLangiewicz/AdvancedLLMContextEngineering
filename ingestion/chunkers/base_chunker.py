from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document


class BaseChunker(ABC):

    @abstractmethod
    def chunk(self, documents: List[Document]) -> List[Document]:
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    
    