from abc import ABC, abstractmethod
from enum import Enum


class QueryUnderstandingStrategy(str, Enum):
    REWRITING = "rewriting"
    HYDE = "hyde"
    MULTI_QUERY = "multi_query"
    STEP_BACK = "step_back"


class BaseQueryUnderstanding(ABC):

    @abstractmethod
    def transform(self, query: str) -> list[str]:
        """
        Takes original query, returns list of transformed queries.
        Rewriting returns 1 query, Multi-Query returns N queries.
        """
        ...