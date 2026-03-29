import logging
from typing import Callable

from langchain_core.language_models import BaseChatModel

from query.base import BaseQueryUnderstanding, QueryUnderstandingStrategy
from query.rewriter import QueryRewriter

logger = logging.getLogger(__name__)

_QUERY_REGISTRY: dict[QueryUnderstandingStrategy, Callable[[BaseChatModel], BaseQueryUnderstanding]] = {
    QueryUnderstandingStrategy.REWRITING: lambda llm: QueryRewriter(llm),
    # QueryUnderstandingStrategy.HYDE:        lambda llm: HyDETransformer(llm),
    # QueryUnderstandingStrategy.MULTI_QUERY: lambda llm: MultiQueryTransformer(llm),
}


def build_query_understanding(
    strategy: QueryUnderstandingStrategy,
    llm: BaseChatModel,
) -> BaseQueryUnderstanding:
    factory = _QUERY_REGISTRY.get(strategy)
    if factory is None:
        raise ValueError(f"Unsupported query understanding strategy: {strategy}")
    logger.info(f"Query understanding built | strategy={strategy.value}")
    return factory(llm)