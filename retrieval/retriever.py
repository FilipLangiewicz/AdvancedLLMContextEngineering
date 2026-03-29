import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

from config.settings import settings, ChunkingStrategy
from ingestion.vector_store import get_vector_store

logger = logging.getLogger(__name__)


class Retriever:

    def __init__(self, embeddings, strategy: ChunkingStrategy = settings.chunking_strategy):
        self.strategy = strategy
        collection = (
            settings.qdrant_collection_structure
            if strategy == ChunkingStrategy.STRUCTURE
            else settings.qdrant_collection_semantic
        )
        self._store: QdrantVectorStore = get_vector_store(collection, embeddings)
        logger.info(f"Retriever initialized | strategy={strategy} | collection={collection}")

    def retrieve(
        self,
        query: str,
        k: int = 5,
        source_filter: Optional[str] = None,
    ) -> List[Document]:
        if source_filter:
            results = self._store.similarity_search(
                query,
                k=k,
                filter={"must": [{"key": "metadata.source", "match": {"value": source_filter}}]},
            )
        else:
            results = self._store.similarity_search(query, k=k)

        logger.info(f"Retrieved {len(results)} chunks for query: '{query[:60]}...'")
        return results