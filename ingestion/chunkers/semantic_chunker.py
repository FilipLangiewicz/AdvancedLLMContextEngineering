import logging
from typing import List, Dict

from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document

from ingestion.chunkers.base_chunker import BaseChunker

logger = logging.getLogger(__name__)

MIN_CHUNK_LENGTH = 200


class SemanticDocChunker(BaseChunker):

    def __init__(self, embeddings, breakpoint_threshold_type: str = "percentile"):
        self._splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
        )

    def chunk(self, documents: List[Document]) -> List[Document]:
        grouped = self._group_by_source(documents)  
        all_chunks: List[Document] = []

        for source, docs in grouped.items():
            full_text, base_metadata = self._merge_pages(docs)
            try:
                sub_docs = self._splitter.create_documents(
                    texts=[full_text],
                    metadatas=[base_metadata],
                )
                sub_docs = self._filter_short_chunks(sub_docs)
                for idx, chunk in enumerate(sub_docs):
                    chunk.metadata.update(base_metadata)
                    chunk.metadata["chunk_type"] = "semantic"
                    chunk.metadata["chunk_index"] = idx
                    chunk.metadata["chunking_strategy"] = "semantic"
                all_chunks.extend(sub_docs)
                logger.info(f"[SemanticChunker] {source}: {len(sub_docs)} chunks")
            except Exception as e:
                logger.warning(f"SemanticChunker failed on {source}: {e}")
                for doc in docs:
                    doc.metadata["chunk_type"] = "semantic_fallback"
                    doc.metadata["chunking_strategy"] = "semantic"
                    all_chunks.append(doc)

        logger.info(f"[SemanticChunker] {len(documents)} pages → {len(all_chunks)} chunks")
        return all_chunks
    
    @staticmethod
    def _group_by_source(documents: List[Document]) -> Dict[str, List[Document]]:
        grouped: Dict[str, List[Document]] = {}
        for doc in documents:
            src = doc.metadata.get("source", "unknown")
            grouped.setdefault(src, []).append(doc)
        return grouped

    @staticmethod
    def _merge_pages(docs: List[Document]):
        """Merges all pages of a source into one string, keeps base metadata."""
        sorted_docs = sorted(docs, key=lambda d: d.metadata.get("page", 0))
        full_text = "\n".join(doc.page_content for doc in sorted_docs)
        base_metadata = {
            k: v for k, v in sorted_docs[0].metadata.items()
            if k not in ("page_label",)
        }
        return full_text, base_metadata
    
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