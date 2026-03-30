import logging
from typing import Callable
from langchain_core.language_models import BaseChatModel

from compression.base import BaseContextCompressor, CompressionStrategy
from compression.extractive import ExtractiveFilterCompressor
from compression.hierarchical import HierarchicalSummaryCompressor

logger = logging.getLogger(__name__)

_COMPRESSION_REGISTRY: dict[CompressionStrategy, Callable[[BaseChatModel], BaseContextCompressor]] = {
    CompressionStrategy.EXTRACTIVE_FILTER:    lambda llm: ExtractiveFilterCompressor(llm),
    CompressionStrategy.HIERARCHICAL_SUMMARY: lambda llm: HierarchicalSummaryCompressor(llm),
}


def build_compressor(
    strategy: CompressionStrategy,
    llm: BaseChatModel,
) -> BaseContextCompressor:
    factory = _COMPRESSION_REGISTRY.get(strategy)
    if factory is None:
        raise ValueError(f"Unsupported compression strategy: {strategy}")
    logger.info(f"Compressor built | strategy={strategy.value}")
    return factory(llm)