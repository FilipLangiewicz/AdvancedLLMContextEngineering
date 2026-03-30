import logging
import numpy as np
from dataclasses import dataclass, field
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    query: str
    answer: str
    sources: list[dict]
    strategy: str


@dataclass
class SemanticCache:
    embeddings: Embeddings
    threshold: float = 0.92
    _entries: list[CacheEntry] = field(default_factory=list)
    _vectors: list[np.ndarray] = field(default_factory=list)

    def get(self, query: str) -> CacheEntry | None:
        if not self._entries:
            return None

        query_vector = np.array(self.embeddings.embed_query(query))
        scores = [self._cosine_similarity(query_vector, v) for v in self._vectors]
        best_idx = int(np.argmax(scores))
        best_score = scores[best_idx]

        if best_score >= self.threshold:
            logger.info(f"Cache HIT | score={best_score:.4f} | query='{query[:60]}'")
            return self._entries[best_idx]

        logger.info(f"Cache MISS | best_score={best_score:.4f} | query='{query[:60]}'")
        return None

    def set(self, query: str, answer: str, sources: list[dict], strategy: str) -> None:
        vector = np.array(self.embeddings.embed_query(query))
        self._entries.append(CacheEntry(query=query, answer=answer, sources=sources, strategy=strategy))
        self._vectors.append(vector)
        logger.info(f"Cache SET | total_entries={len(self._entries)} | query='{query[:60]}'")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    @property
    def size(self) -> int:
        return len(self._entries)