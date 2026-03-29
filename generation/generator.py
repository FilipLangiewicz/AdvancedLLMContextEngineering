import logging
from dataclasses import dataclass, field
from typing import List

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import settings, LLMProvider
from config.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class GeneratorResponse:
    answer: str
    sources: List[dict] = field(default_factory=list)
    strategy: str = ""


def build_llm() -> BaseChatModel:
    if settings.llm_provider == LLMProvider.GOOGLE:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.google_api_key,
            temperature=settings.llm_temperature,
        )
    elif settings.llm_provider == LLMProvider.GROQ:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=settings.llm_model,
            api_key=settings.groq_api_key,
            temperature=settings.llm_temperature,
        )
    elif settings.llm_provider == LLMProvider.OPENAI:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=settings.llm_temperature,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


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

    def __init__(self):
        self._llm = build_llm()
        logger.info(f"Generator initialized | provider={settings.llm_provider} | model={settings.llm_model}")

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
        )