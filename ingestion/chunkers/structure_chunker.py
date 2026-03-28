import re
import logging
from typing import List, Tuple, Dict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from ingestion.chunkers.base_chunker import BaseChunker

logger = logging.getLogger(__name__)

_CHAPTER_PATTERN = re.compile(
    r'(Rozdzia[łl]\s+\d+[a-z]?\s*\n[^\n]+)',
    re.IGNORECASE
)
_ARTICLE_PATTERN = re.compile(
    r'(Art\.\s*\d+[a-z]?\s*[a-z]?\.|§\s*\d+[a-z]?\.)',
    re.IGNORECASE
)


class StructureChunker(BaseChunker):

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, documents: List[Document]) -> List[Document]:
        grouped = self._group_by_source(documents)
        all_chunks: List[Document] = []

        for source, docs in grouped.items():
            full_text, page_map = self._merge_pages(docs)
            chunks = self._split_by_structure(full_text, page_map, source, docs[0].metadata)
            all_chunks.extend(chunks)
            logger.info(f"[StructureChunker] {source}: {len(chunks)} chunks")

        return all_chunks

    @staticmethod
    def _group_by_source(documents: List[Document]) -> Dict[str, List[Document]]:
        grouped: Dict[str, List[Document]] = {}
        for doc in documents:
            src = doc.metadata.get("source", "unknown")
            grouped.setdefault(src, []).append(doc)
        return grouped

    @staticmethod
    def _merge_pages(docs: List[Document]) -> Tuple[str, List[int]]:
        parts = []
        page_map: List[int] = []
        for doc in sorted(docs, key=lambda d: d.metadata.get("page", 0)):
            page_num = doc.metadata.get("page", 0)
            parts.append(doc.page_content)
            page_map.extend([page_num] * len(doc.page_content))
            parts.append("\n")
            page_map.append(page_num)
        return "".join(parts), page_map

    def _split_by_structure(
        self,
        text: str,
        page_map: List[int],
        source: str,
        base_metadata: dict,
    ) -> List[Document]:
        chunks: List[Document] = []
        current_chapter: str = "General provisions"

        article_spans = [
            (m.start(), m.group(0)) for m in _ARTICLE_PATTERN.finditer(text)
        ]

        if not article_spans:
            logger.warning(f"No articles found in {source}, using fallback splitter.")
            return self._fallback_chunk(text, page_map, source, base_metadata)

        for idx, (start, article_id) in enumerate(article_spans):
            search_from = article_spans[idx - 1][0] if idx > 0 else 0
            chapter_match = _CHAPTER_PATTERN.search(text, search_from, start)
            if chapter_match:
                current_chapter = chapter_match.group(0).strip()

            end = article_spans[idx + 1][0] if idx + 1 < len(article_spans) else len(text)
            article_text = text[start:end].strip()
            
            chunk_unit = "paragraph" if article_id.startswith("§") else "article"

            if len(article_text) > self.chunk_size * 2:
                sub_chunks = self._fallback_splitter.split_text(article_text)
                for sub_idx, sub_text in enumerate(sub_chunks):
                    page = self._get_page(page_map, start)
                    chunks.append(Document(
                        page_content=sub_text,
                        metadata={
                            **base_metadata,
                            "source": source,
                            "page": page,
                            "chunk_type": "fragment",
                            "article_id": article_id,
                            "sub_chunk": sub_idx,
                            "chapter": current_chapter,
                            "chunking_strategy": "structure",
                        }
                    ))
            else:
                page = self._get_page(page_map, start)
                chunks.append(Document(
                    page_content=article_text,
                    metadata={
                        **base_metadata,
                        "source": source,
                        "page": page,
                        "chunk_type": chunk_unit,
                        "article_id": article_id,
                        "chapter": current_chapter,
                        "chunking_strategy": "structure",
                    }
                ))

        return chunks

    def _fallback_chunk(
        self,
        text: str,
        page_map: List[int],
        source: str,
        base_metadata: dict,
    ) -> List[Document]:
        raw_chunks = self._fallback_splitter.split_text(text)
        docs = []
        offset = 0
        for raw in raw_chunks:
            page = self._get_page(page_map, offset)
            docs.append(Document(
                page_content=raw,
                metadata={
                    **base_metadata,
                    "source": source,
                    "page": page,
                    "chunk_type": "fragment",
                    "chunking_strategy": "structure",
                }
            ))
            offset += len(raw)
        return docs

    @staticmethod
    def _get_page(page_map: List[int], pos: int) -> int:
        if pos < len(page_map):
            return page_map[pos]
        return page_map[-1] if page_map else 0