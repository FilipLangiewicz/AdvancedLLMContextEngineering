import logging
from typing import List

from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document

from ingestion.chunkers.base_chunker import BaseChunker

logger = logging.getLogger(__name__)

MIN_CHUNK_LENGTH = 100


class SemanticDocChunker(BaseChunker):

    def __init__(self, embeddings, breakpoint_threshold_type: str = "percentile"):
        self._splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
        )

    def chunk(self, documents: List[Document]) -> List[Document]:
        all_chunks: List[Document] = []

        for doc in documents:
            try:
                sub_docs = self._splitter.create_documents(
                    texts=[doc.page_content],
                    metadatas=[doc.metadata],
                )
                sub_docs = self._filter_short_chunks(sub_docs)
                for idx, chunk in enumerate(sub_docs):
                    chunk.metadata["chunk_type"] = "semantic"
                    chunk.metadata["chunk_index"] = idx
                    chunk.metadata["chunking_strategy"] = "semantic"
                all_chunks.extend(sub_docs)
            except Exception as e:
                logger.warning(f"SemanticChunker failed on {doc.metadata.get('source')}, page {doc.metadata.get('page')}: {e}")
                doc.metadata["chunk_type"] = "semantic_fallback"
                doc.metadata["chunking_strategy"] = "semantic"
                all_chunks.append(doc)

        logger.info(f"[SemanticChunker] {len(documents)} pages → {len(all_chunks)} chunks")
        return all_chunks
    
    @staticmethod
    def _filter_short_chunks(chunks: List[Document]) -> List[Document]:
        filtered = []
        for chunk in chunks:
            text = chunk.page_content.strip()
            if len(text) < MIN_CHUNK_LENGTH and filtered:
                filtered[-1].page_content += " " + text
            else:
                filtered.append(chunk)
        return filtered