import logging
from typing import Callable
from langchain_core.language_models import BaseChatModel
from config.settings import settings, LLMProvider
from itertools import cycle
from threading import Lock

def _parse_groq_keys() -> list[str]:
    if settings.groq_api_keys:
        keys = [k.strip() for k in settings.groq_api_keys.split(",") if k.strip()]
        if keys:
            return keys
    if settings.groq_api_key:
        return [settings.groq_api_key]
    return []

_GROQ_KEYS = _parse_groq_keys()
_GROQ_KEY_CYCLE = cycle(_GROQ_KEYS) if _GROQ_KEYS else None
_GROQ_KEY_LOCK = Lock()

def _next_groq_key() -> str:
    if _GROQ_KEY_CYCLE is None:
        raise ValueError("No Groq API key configured (set GROQ_API_KEY or GROQ_API_KEYS in .env)")
    with _GROQ_KEY_LOCK:
        return next(_GROQ_KEY_CYCLE)

logger = logging.getLogger(__name__)


def _make_google(model: str) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=settings.google_api_key,
        temperature=settings.llm_temperature,
    )

def _make_groq(model: str) -> BaseChatModel:
    from langchain_groq import ChatGroq
    key = _next_groq_key()
    logger.info(f"Groq LLM | model={model} | key=...{key[-6:]}")
    return ChatGroq(
        model=model,
        api_key=key,
        temperature=settings.llm_temperature,
    )

def _make_openai(model: str) -> BaseChatModel:
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model,
        api_key=settings.openai_api_key,
        temperature=settings.llm_temperature,
    )

def _make_anthropic(model: str) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(
        model=model,
        api_key=settings.anthropic_api_key,
        temperature=settings.llm_temperature,
    )


_LLM_REGISTRY: dict[LLMProvider, Callable[[str], BaseChatModel]] = {
    LLMProvider.GOOGLE:    _make_google,
    LLMProvider.GROQ:      _make_groq,
    LLMProvider.OPENAI:    _make_openai,
    LLMProvider.ANTHROPIC: _make_anthropic,
}

_DEFAULT_MODELS: dict[LLMProvider, str] = {
    LLMProvider.GOOGLE:    settings.google_model,
    LLMProvider.GROQ:      settings.groq_model,
    LLMProvider.OPENAI:    settings.openai_model,
    LLMProvider.ANTHROPIC: settings.anthropic_model,
}


def build_llm(
    provider: LLMProvider | None = None,
    model: str | None = None,
) -> BaseChatModel:
    resolved_provider = provider or settings.llm_provider
    resolved_model = model or _DEFAULT_MODELS[resolved_provider]

    factory = _LLM_REGISTRY.get(resolved_provider)
    if factory is None:
        raise ValueError(f"Unsupported LLM provider: {resolved_provider}")
    
    llm = factory(resolved_model)
    logger.info(f"LLM built | provider={resolved_provider} | model={resolved_model}")
    return llm