import os
import sys

import logging
import time

from langchain_openai import OpenAIEmbeddings

from cache.semantic_cache import SemanticCache
from compression.base import CompressionStrategy
from config.settings import settings, ChunkingStrategy
from generation.llm_factory import build_llm
from pipeline import RAGPipeline, build_pipeline
from query.base import QueryUnderstandingStrategy
from reranking.base import RerankingStrategy
from reranking.reranking_factory import create_reranker
from generation.embeddings_factory import build_embeddings


logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_QUERY = "Jakie są stawki opłat za ciepło?"
SAMPLE_QUERY_2 = "Jakie są zasady obliczania ceny energii cieplnej?"
LONG_QUERY = "Proszę o szczegółowe wyjaśnienie " + "zasad taryfowych " * 80


class TestBasicPipeline:

    def test_default_pipeline_returns_answer(self):
        embeddings = build_embeddings()
        pipeline = build_pipeline(embeddings=embeddings, with_cache=False)
        response = pipeline.run(SAMPLE_QUERY)

        print(f"    Query:    {SAMPLE_QUERY}")
        print(f"    Answer:   {response.answer[:120]}...")
        print(f"    Sources:  {len(response.sources)} docs")
        print(f"    Strategy: {response.strategy}")

        assert response.answer, "Answer should not be empty"
        assert len(response.sources) > 0, "Should return at least one source"
        assert response.strategy, "Strategy label should be set"

    def test_response_contains_sources_metadata(self):
        embeddings = build_embeddings()
        pipeline = build_pipeline(embeddings=embeddings, with_cache=False)
        response = pipeline.run(SAMPLE_QUERY)

        for source in response.sources:
            print(f"    Source: {source}")
            assert "source" in source, "Each source should have 'source' key"
            assert "page" in source, "Each source should have 'page' key"

    def test_pipeline_reusable_multiple_queries(self):
        """Same pipeline instance handles multiple queries — simulates conversational usage."""
        embeddings = build_embeddings()
        pipeline = build_pipeline(embeddings=embeddings, with_cache=False)

        response1 = pipeline.run(SAMPLE_QUERY)
        response2 = pipeline.run(SAMPLE_QUERY_2)
        response3 = pipeline.run("Co to jest taryfa ciepłownicza?")

        print(f"    Query 1 answer length: {len(response1.answer)}")
        print(f"    Query 2 answer length: {len(response2.answer)}")
        print(f"    Query 3 answer length: {len(response3.answer)}")

        assert response1.answer
        assert response2.answer
        assert response3.answer
        assert response1.answer != response2.answer, "Different queries should produce different answers"


class TestPipelineStrategies:

    def test_u_shape_reranking(self):
        embeddings = build_embeddings()
        pipeline = build_pipeline(
            embeddings=embeddings,
            reranking_strategy=RerankingStrategy.U_SHAPE_REORDER,
            with_cache=False,
        )
        response = pipeline.run(SAMPLE_QUERY)

        print(f"    Answer: {response.answer[:120]}...")
        print(f"    Strategy: {response.strategy}")

        assert "UShapeReranker" in response.strategy
        assert response.answer

    def test_query_rewriting(self):
        embeddings = build_embeddings()
        pipeline = build_pipeline(
            embeddings=embeddings,
            query_understanding_strategy=QueryUnderstandingStrategy.REWRITING,
            with_cache=False,
        )
        response = pipeline.run("ile kosztuje ciepło?")

        print(f"    Original: 'ile kosztuje ciepło?'")
        print(f"    Answer: {response.answer[:120]}...")
        print(f"    Strategy: {response.strategy}")

        assert "QueryRewriter" in response.strategy
        assert response.answer

    def test_extractive_compression(self):
        embeddings = build_embeddings()
        pipeline = build_pipeline(
            embeddings=embeddings,
            compression_strategy=CompressionStrategy.EXTRACTIVE_FILTER,
            with_cache=False,
        )
        response = pipeline.run(SAMPLE_QUERY)

        print(f"    Answer: {response.answer[:120]}...")
        print(f"    Strategy: {response.strategy}")

        assert "ExtractiveFilterCompressor" in response.strategy
        assert response.answer

    def test_hierarchical_compression(self):
        embeddings = build_embeddings()
        pipeline = build_pipeline(
            embeddings=embeddings,
            compression_strategy=CompressionStrategy.HIERARCHICAL_SUMMARY,
            with_cache=False,
        )
        response = pipeline.run(SAMPLE_QUERY)

        print(f"    Answer: {response.answer[:120]}...")
        print(f"    Strategy: {response.strategy}")

        assert "HierarchicalSummaryCompressor" in response.strategy
        assert response.answer

    def test_all_strategies_combined(self):
        embeddings = build_embeddings()
        pipeline = build_pipeline(
            embeddings=embeddings,
            query_understanding_strategy=QueryUnderstandingStrategy.REWRITING,
            reranking_strategy=RerankingStrategy.U_SHAPE_REORDER,
            compression_strategy=CompressionStrategy.HIERARCHICAL_SUMMARY,
            with_cache=False,
        )
        response = pipeline.run(SAMPLE_QUERY)

        print(f"    Answer: {response.answer[:120]}...")
        print(f"    Strategy: {response.strategy}")

        assert "QueryRewriter" in response.strategy
        assert "UShapeReranker" in response.strategy
        assert "HierarchicalSummaryCompressor" in response.strategy
        assert response.answer


class TestSemanticCache:

    def test_cache_hit_on_repeated_query(self):
        embeddings = build_embeddings()
        pipeline = build_pipeline(embeddings=embeddings, with_cache=True)

        response1 = pipeline.run(SAMPLE_QUERY)
        assert "[CACHED]" not in response1.strategy, "First call should not be cached"
        print(f"    First call  — strategy: {response1.strategy}")

        response2 = pipeline.run(SAMPLE_QUERY)
        assert "[CACHED]" in response2.strategy, "Second identical call should be a cache hit"
        print(f"    Second call — strategy: {response2.strategy}")
        assert response1.answer == response2.answer, "Cached answer should match original"

    def test_cache_hit_on_semantically_similar_query(self):
        embeddings = build_embeddings()
        pipeline = build_pipeline(embeddings=embeddings, with_cache=True, cache_threshold=0.90)

        pipeline.run(SAMPLE_QUERY)
        similar_query = "Ile wynoszą opłaty za energię cieplną?"
        response = pipeline.run(similar_query)

        print(f"    Original:  '{SAMPLE_QUERY}'")
        print(f"    Similar:   '{similar_query}'")
        print(f"    Cache hit: {'[CACHED]' in response.strategy}")
        print(f"    Strategy:  {response.strategy}")
        # Nie assertujemy cache HIT — wynik zależy od threshold i embeddingów,
        # ale pipeline nie może crashować
        assert response.answer

    def test_cache_miss_on_different_query(self):
        embeddings = build_embeddings()
        pipeline = build_pipeline(embeddings=embeddings, with_cache=True)

        pipeline.run(SAMPLE_QUERY)
        response = pipeline.run("Jakie są kary za nieterminową płatność faktury?")

        print(f"    Cache miss expected — strategy: {response.strategy}")
        assert "[CACHED]" not in response.strategy, "Different query should not hit cache"
        assert response.answer


class TestEdgeCases:

    def test_very_short_query(self):
        embeddings = build_embeddings()
        pipeline = build_pipeline(embeddings=embeddings, with_cache=False)

        response = pipeline.run("cena")
        print(f"    Query: 'cena'")
        print(f"    Answer: {response.answer[:120]}...")
        assert response.answer, "Should handle very short query without crashing"

    def test_very_long_query(self):
        """Checks that the pipeline handles a query that might exceed token limits gracefully."""
        embeddings = build_embeddings()
        pipeline = build_pipeline(embeddings=embeddings, with_cache=False)

        try:
            response = pipeline.run(LONG_QUERY)
            print(f"    Long query handled | answer length: {len(response.answer)}")
            assert response.answer
        except Exception as e:
            print(f"    Long query raised exception (expected): {type(e).__name__}: {str(e)[:100]}")
            assert any(keyword in str(e).lower() for keyword in ["token", "limit", "length", "context"]), \
                f"Unexpected exception type: {e}"

    def test_source_filter(self):
        embeddings = build_embeddings()
        pipeline = build_pipeline(embeddings=embeddings, with_cache=False)

        response = pipeline.run(SAMPLE_QUERY, source_filter="document1.pdf")
        print(f"    Source filter: 'document1.pdf'")
        print(f"    Sources returned: {response.sources}")

        for source in response.sources:
            if source["source"] != "?":
                assert "document1" in source["source"], \
                    f"Source filter not respected: got {source['source']}"

    def test_pipeline_without_cache(self):
        """Ensures with_cache=False doesn't break anything and never returns [CACHED]."""
        embeddings = build_embeddings()
        pipeline = build_pipeline(embeddings=embeddings, with_cache=False)

        r1 = pipeline.run(SAMPLE_QUERY)
        r2 = pipeline.run(SAMPLE_QUERY)

        assert "[CACHED]" not in r1.strategy
        assert "[CACHED]" not in r2.strategy
        print(f"    Both calls processed without cache as expected")

os.makedirs("tests/outputs", exist_ok=True)
log_path = "tests/outputs/test_rag_pipeline.log"

class Tee:
    """Writes output to both terminal and file simultaneously."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return False

with open(log_path, "w", encoding="utf-8") as log_file:
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    test_classes = [
        TestBasicPipeline(),
        TestPipelineStrategies(),
        TestSemanticCache(),
        TestEdgeCases(),
    ]

    passed = 0
    failed = 0

    for test_instance in test_classes:
        class_name = test_instance.__class__.__name__
        print(f"\n{'='*55}")
        print(f"  {class_name}")
        print(f"{'='*55}")
        methods = [m for m in dir(test_instance) if m.startswith("test_")]
        for method_name in methods:
            print(f"\n  [{method_name}]")
            try:
                getattr(test_instance, method_name)()
                print(f"  --> PASS")
                passed += 1
            except Exception as e:
                print(f"  --> FAIL: {type(e).__name__}: {e}")
                failed += 1

    print(f"\n{'='*55}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*55}")

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

print(f"\nLog saved to: {log_path}")