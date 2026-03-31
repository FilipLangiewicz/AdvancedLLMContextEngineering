import logging

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from cache.semantic_cache import SemanticCache
from compression.base import BaseContextCompressor, CompressionStrategy
from compression.compression_factory import build_compressor
from config.settings import settings, ChunkingStrategy, LLMProvider
from generation.generator import Generator, GeneratorResponse
from generation.llm_factory import build_llm
from query.base import BaseQueryUnderstanding, QueryUnderstandingStrategy
from query.query_factory import build_query_understanding
from reranking.base import BaseReranker, RerankingStrategy
from reranking.reranking_factory import create_reranker
from retrieval.retriever import Retriever

logger = logging.getLogger(__name__)

_UNSET = object()


class RAGPipeline:
    """
    Orchestrates the full RAG pipeline:
    query → cache → query transform → retrieve → rerank → compress → generate → cache
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        reranker: BaseReranker,
        compressor: BaseContextCompressor | None = None,
        query_transformer: BaseQueryUnderstanding | None = None,
        cache: SemanticCache | None = None,
        retrieval_k: int = 5,
    ):
        self._retriever = retriever
        self._generator = generator
        self._reranker = reranker
        self._compressor = compressor
        self._query_transformer = query_transformer
        self._cache = cache
        self._retrieval_k = retrieval_k

    def run(self, query: str, source_filter: str | None = None) -> GeneratorResponse:
        logger.info(f"Pipeline run | query='{query[:80]}'")

        # 1. Semantic cache check
        if self._cache:
            cached = self._cache.get(query)
            if cached:
                return GeneratorResponse(
                    answer=cached.answer,
                    sources=cached.sources,
                    strategy=cached.strategy + " [CACHED]",
                )

        # 2. Query transformation (optional)
        queries = [query]
        if self._query_transformer:
            queries = self._query_transformer.transform(query)

        # 3. Retrieval — supports multi-query (e.g. future MultiQuery strategy)
        docs = self._retrieve(queries, source_filter)

        # 4. Reranking
        docs = self._reranker.rerank(docs)

        # 5. Compression (optional)
        if self._compressor:
            docs = self._compressor.compress(query, docs)

        # 6. Generation
        strategy_label = self._strategy_label()
        response = self._generator.generate(query, docs, strategy=strategy_label)

        # 7. Cache store
        if self._cache:
            self._cache.set(query, response.answer, response.sources, response.strategy)

        return response

    def _retrieve(self, queries: list[str], source_filter: str | None) -> list[Document]:
        if len(queries) == 1:
            return self._retriever.retrieve(
                queries[0], k=self._retrieval_k, source_filter=source_filter
            )

        # Multi-query: deduplicate by first 100 chars of content
        seen: set[str] = set()
        merged: list[Document] = []
        for q in queries:
            for doc in self._retriever.retrieve(q, k=self._retrieval_k, source_filter=source_filter):
                fingerprint = doc.page_content[:100]
                if fingerprint not in seen:
                    seen.add(fingerprint)
                    merged.append(doc)

        logger.info(f"Multi-query retrieval | queries={len(queries)} | unique_docs={len(merged)}")
        return merged

    def _strategy_label(self) -> str:
        return " | ".join([
            f"query_transform={self._query_transformer.__class__.__name__ if self._query_transformer else 'none'}",
            f"reranking={self._reranker.__class__.__name__}",
            f"compression={self._compressor.__class__.__name__ if self._compressor else 'none'}",
        ])


def build_pipeline(
    embeddings: Embeddings,
    chunking_strategy: ChunkingStrategy | object = _UNSET,
    query_understanding_strategy: QueryUnderstandingStrategy | None | object = _UNSET,
    compression_strategy: CompressionStrategy | None | object = _UNSET,
    reranking_strategy: RerankingStrategy | object = _UNSET,
    llm_provider: LLMProvider | object = _UNSET,
    retrieval_k: int | object = _UNSET,
    cache_threshold: float | object = _UNSET,
    with_cache: bool = True,
) -> RAGPipeline:
    """
    Builds a RAGPipeline from explicit parameters, falling back to settings for any unset value.

    Args:
        embeddings:                   Required — used by Retriever and SemanticCache.
        chunking_strategy:            Which vector store collection to query.
        query_understanding_strategy: None disables query transformation.
        compression_strategy:         None disables compression.
        reranking_strategy:           Defaults to ORIGINAL_ORDER.
        llm_provider:                 Which LLM backend to use.
        retrieval_k:                  Number of documents to retrieve.
        cache_threshold:              Cosine similarity threshold for cache hits.
        with_cache:                   Set to False to disable cache entirely (useful in tests).
    """
    resolved_chunking = settings.chunking_strategy if chunking_strategy is _UNSET else chunking_strategy
    resolved_query = settings.query_understanding_strategy if query_understanding_strategy is _UNSET else query_understanding_strategy
    resolved_compression = settings.compression_strategy if compression_strategy is _UNSET else compression_strategy
    resolved_reranking = settings.reranking_strategy if reranking_strategy is _UNSET else reranking_strategy
    resolved_provider = settings.llm_provider if llm_provider is _UNSET else llm_provider
    resolved_k = settings.retrieval_k if retrieval_k is _UNSET else retrieval_k
    resolved_threshold = settings.cache_threshold if cache_threshold is _UNSET else cache_threshold

    llm = build_llm(provider=resolved_provider)

    retriever = Retriever(embeddings, strategy=resolved_chunking)
    generator = Generator(provider=resolved_provider)
    reranker = create_reranker(resolved_reranking)
    compressor = build_compressor(resolved_compression, llm) if resolved_compression else None
    query_transformer = build_query_understanding(resolved_query, llm) if resolved_query else None
    cache = SemanticCache(embeddings=embeddings, threshold=resolved_threshold) if with_cache else None

    logger.info(
        f"Pipeline built | chunking={resolved_chunking} | query={resolved_query} | "
        f"reranking={resolved_reranking} | compression={resolved_compression} | "
        f"provider={resolved_provider} | k={resolved_k} | cache={'on' if cache else 'off'}"
    )

    return RAGPipeline(
        retriever=retriever,
        generator=generator,
        reranker=reranker,
        compressor=compressor,
        query_transformer=query_transformer,
        cache=cache,
        retrieval_k=resolved_k,
    )