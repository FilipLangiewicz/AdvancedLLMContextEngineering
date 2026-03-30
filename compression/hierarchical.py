import logging
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from compression.base import BaseContextCompressor
from config.prompts import HIERARCHICAL_CHUNK_SUMMARY_PROMPT, HIERARCHICAL_FINAL_SUMMARY_PROMPT

logger = logging.getLogger(__name__)


class HierarchicalSummaryCompressor(BaseContextCompressor):

    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def compress(self, query: str, docs: list[Document]) -> list[Document]:
        # Level 1 — summarize each chunk individually
        chunk_summaries = []
        for i, doc in enumerate(docs):
            messages = [
                SystemMessage(content=HIERARCHICAL_CHUNK_SUMMARY_PROMPT),
                HumanMessage(content=f"Zapytanie: {query}\n\nFragment:\n{doc.page_content}"),
            ]
            summary = self._llm.invoke(messages).content.strip()
            chunk_summaries.append(summary)
            logger.info(f"Chunk {i+1}/{len(docs)} summarized | source={doc.metadata.get('source')}")

        # Level 2 — merge all summaries into one final summary
        merged = "\n\n".join(f"[{i+1}] {s}" for i, s in enumerate(chunk_summaries))
        messages = [
            SystemMessage(content=HIERARCHICAL_FINAL_SUMMARY_PROMPT),
            HumanMessage(content=f"Zapytanie: {query}\n\nPodsumowania fragmentów:\n{merged}"),
        ]
        final_summary = self._llm.invoke(messages).content.strip()

        original_tokens = sum(len(d.page_content.split()) for d in docs)
        final_tokens = len(final_summary.split())
        logger.info(f"Hierarchical compression | chunks={len(docs)} | tokens: ~{original_tokens}→~{final_tokens}")

        return [Document(page_content=final_summary, metadata={"source": "hierarchical_summary", "chunks_count": len(docs)})]