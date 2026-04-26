import logging
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from compression.base import BaseContextCompressor
from config.prompts import (
    BRIEF_CONTEXT_PARTITION_PROMPT,
    BRIEF_CONTEXT_AGGREGATE_PROMPT,
)

logger = logging.getLogger(__name__)


class BriefContextCompressor(BaseContextCompressor):
    """
    Map-reduce inspirowany BriefContext (Zhang et al. 2025).
    
    Dzieli pobrane fragmenty na N partycji, generuje odpowiedź per partycja,
    następnie agreguje wszystkie odpowiedzi w jedną finalną.
    
    Mityguje lost-in-the-middle eliminując długi kontekst - każda partycja
    jest na tyle krótka, że nic nie ginie w środku.
    """

    def __init__(self, llm: BaseChatModel, n_partitions: int = 3):
        self._llm = llm
        self._n = n_partitions

    def compress(self, query: str, docs: list[Document]) -> list[Document]:
        if not docs:
            return docs
        
        partitions = self._partition(docs, self._n)
        actual_n = len(partitions)
        logger.info(f"BriefContext | {len(docs)} docs → {actual_n} partycji")
        
        partial_answers = []
        for i, partition in enumerate(partitions):
            partition_text = "\n\n".join(
                f"[{j+1}] {d.metadata.get('source', '?')}, "
                f"strona {d.metadata.get('page', '?')}: {d.page_content}"
                for j, d in enumerate(partition)
            )
            messages = [
                SystemMessage(content=BRIEF_CONTEXT_PARTITION_PROMPT),
                HumanMessage(content=f"PYTANIE: {query}\n\nFRAGMENTY:\n{partition_text}"),
            ]
            partial = self._llm.invoke(messages).content.strip()
            partial_answers.append(partial)
            logger.info(f"  Partycja {i+1}/{actual_n} | {len(partition)} docs | answer={len(partial.split())}t")
        merged = "\n\n".join(f"--- Odpowiedź cząstkowa {i+1} ---\n{a}"
                             for i, a in enumerate(partial_answers))
        messages = [
            SystemMessage(content=BRIEF_CONTEXT_AGGREGATE_PROMPT),
            HumanMessage(content=f"PYTANIE: {query}\n\nODPOWIEDZI CZĄSTKOWE:\n{merged}"),
        ]
        final = self._llm.invoke(messages).content.strip()
        
        original_tokens = sum(len(d.page_content.split()) for d in docs)
        final_tokens = len(final.split())
        logger.info(f"BriefContext compression | docs={len(docs)} | tokens: ~{original_tokens}→~{final_tokens}")
        
        return [Document(
            page_content=final,
            metadata={
                "source": "brief_context",
                "n_partitions": actual_n,
                "n_original_docs": len(docs),
            }
        )]
    
    @staticmethod
    def _partition(docs: list[Document], n: int) -> list[list[Document]]:
        if len(docs) <= n:
            return [[d] for d in docs]
        
        size = len(docs) // n
        remainder = len(docs) % n
        partitions = []
        idx = 0
        for i in range(n):
            extra = 1 if i < remainder else 0
            chunk_size = size + extra
            partitions.append(docs[idx:idx + chunk_size])
            idx += chunk_size
        return partitions