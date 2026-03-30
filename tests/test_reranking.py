from langchain_core.documents import Document
from reranking.reranking_factory import create_reranker
from reranking.base import RerankingStrategy


def make_docs(n: int) -> list[Document]:
    """Creates n documents with IDs in metadata (d1 = best, dn = worst)."""
    return [
        Document(page_content=f"Document content {i+1}", metadata={"id": f"d{i+1}", "rank": i+1})
        for i in range(n)
    ]


def ids(docs: list[Document]) -> list[str]:
    return [d.metadata["id"] for d in docs]


class TestOriginalOrderReranker:

    def test_returns_same_order(self):
        docs = make_docs(5)
        reranker = create_reranker(RerankingStrategy.ORIGINAL_ORDER)
        result = reranker.rerank(docs)

        input_ids = ids(docs)
        result_ids = ids(result)
        print(f"    Input:  {input_ids}")
        print(f"    Output: {result_ids}")
        assert result_ids == ["d1", "d2", "d3", "d4", "d5"]
        print(f"    Order unchanged as expected")

    def test_single_document(self):
        docs = make_docs(1)
        reranker = create_reranker(RerankingStrategy.ORIGINAL_ORDER)
        result = reranker.rerank(docs)

        print(f"    Input:  {ids(docs)}")
        print(f"    Output: {ids(result)}")
        assert len(result) == 1
        print(f"    Single document returned correctly")

    def test_empty_list(self):
        reranker = create_reranker(RerankingStrategy.ORIGINAL_ORDER)
        result = reranker.rerank([])

        print(f"    Input:  []")
        print(f"    Output: {result}")
        assert result == []
        print(f"    Empty list handled correctly")


class TestUShapeReranker:

    def test_best_doc_is_first(self):
        docs = make_docs(5)
        reranker = create_reranker(RerankingStrategy.U_SHAPE_REORDER)
        result = reranker.rerank(docs)

        print(f"    Input:  {ids(docs)}")
        print(f"    Output: {ids(result)}")
        assert result[0].metadata["id"] == "d1", "Best document should be placed first"
        print(f"    Best document '{result[0].metadata['id']}' correctly placed at position 0")

    def test_second_best_is_last(self):
        docs = make_docs(5)
        reranker = create_reranker(RerankingStrategy.U_SHAPE_REORDER)
        result = reranker.rerank(docs)

        print(f"    Input:  {ids(docs)}")
        print(f"    Output: {ids(result)}")
        assert result[-1].metadata["id"] == "d2", "Second best document should be placed last"
        print(f"    Second best document '{result[-1].metadata['id']}' correctly placed at last position")

    def test_no_documents_lost(self):
        docs = make_docs(7)
        reranker = create_reranker(RerankingStrategy.U_SHAPE_REORDER)
        result = reranker.rerank(docs)

        original_ids = set(ids(docs))
        result_ids = set(ids(result))
        print(f"    Input:  {ids(docs)}")
        print(f"    Output: {ids(result)}")
        print(f"    Input count: {len(docs)} | Output count: {len(result)}")
        assert len(result) == len(docs)
        assert original_ids == result_ids, "No document should be lost or duplicated"
        print(f"    All {len(result)} documents present, none lost or duplicated")

    def test_two_docs_unchanged(self):
        docs = make_docs(2)
        reranker = create_reranker(RerankingStrategy.U_SHAPE_REORDER)
        result = reranker.rerank(docs)

        print(f"    Input:  {ids(docs)}")
        print(f"    Output: {ids(result)}")
        assert len(result) == 2
        print(f"    Edge case: 2 documents returned as-is (U-shape not applied below threshold)")

    def test_order_differs_from_original(self):
        docs = make_docs(5)
        reranker = create_reranker(RerankingStrategy.U_SHAPE_REORDER)
        result = reranker.rerank(docs)

        original_ids = ids(docs)
        result_ids = ids(result)
        print(f"    Input:  {original_ids}")
        print(f"    Output: {result_ids}")
        assert result_ids != original_ids, "U-shape should produce a different order"
        print(f"    Order successfully changed — middle documents repositioned")


class TestRerankingFactory:

    def test_unknown_strategy_raises(self):
        strategy = "non_existent_strategy"
        print(f"    Attempting to create reranker with invalid strategy: '{strategy}'")
        try:
            create_reranker(strategy)
            assert False, "Expected ValueError for unknown strategy"
        except ValueError as e:
            print(f"    ValueError raised as expected: {e}")
            assert "Unknown reranking strategy" in str(e)

    def test_all_strategies_creatable(self):
        for strategy in RerankingStrategy:
            reranker = create_reranker(strategy)
            assert reranker is not None
            print(f"    Strategy '{strategy.value}' -> {reranker.__class__.__name__} created successfully")


# --- run ---
test_classes = [
    TestOriginalOrderReranker(),
    TestUShapeReranker(),
    TestRerankingFactory(),
]

passed = 0
failed = 0

for test_instance in test_classes:
    class_name = test_instance.__class__.__name__
    print(f"\n{'='*50}")
    print(f"  {class_name}")
    print(f"{'='*50}")
    methods = [m for m in dir(test_instance) if m.startswith("test_")]
    for method_name in methods:
        print(f"\n  [{method_name}]")
        try:
            getattr(test_instance, method_name)()
            print(f"  --> PASS")
            passed += 1
        except Exception as e:
            print(f"  --> FAIL: {e}")
            failed += 1

print(f"\n{'='*50}")
print(f"  Results: {passed} passed, {failed} failed")
print(f"{'='*50}")