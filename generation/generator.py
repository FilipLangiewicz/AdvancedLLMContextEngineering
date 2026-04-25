import logging
from dataclasses import dataclass, field
from typing import List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import LLMProvider, settings
from config.prompts import SYSTEM_PROMPT

from generation.llm_factory import build_llm


logger = logging.getLogger(__name__)


@dataclass
class GeneratorResponse:
    answer: str
    sources: List[dict] = field(default_factory=list)
    strategy: str = ""
    context_text: str = ""

def _build_context(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "?")
        page = doc.metadata.get("page", "?")
        article = doc.metadata.get("article_id", "")
        header = f"[{i}] {source}, strona {page}" + (f", {article}" if article else "")
        parts.append(f"{header}:\n{doc.page_content}")
    return "\n\n".join(parts)


def _extract_sources(docs: List[Document]) -> List[dict]:
    return [
        {
            "source": doc.metadata.get("source", "?"),
            "page": doc.metadata.get("page", "?"),
            "article_id": doc.metadata.get("article_id", ""),
            "chapter": doc.metadata.get("chapter", ""),
            "chunking_strategy": doc.metadata.get("chunking_strategy", ""),
        }
        for doc in docs
    ]


class Generator:
    
    def __init__(
        self,
        provider: LLMProvider | None = None,
        model: str | None = None,
    ):
        self._llm = build_llm(provider=provider, model=model)

    def generate(self, query: str, docs: List[Document], strategy: str = "") -> GeneratorResponse:
        context = _build_context(docs)
        sources = _extract_sources(docs)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"KONTEKST:\n{context}\n\nPYTANIE: {query}"),
        ]

        logger.info(f"Generating answer | query='{query[:60]}' | docs={len(docs)}")
        response = self._llm.invoke(messages)

        return GeneratorResponse(
            answer=response.content,
            sources=sources,
            strategy=strategy,
            context_text=context,
        )