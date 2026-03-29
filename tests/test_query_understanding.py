import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
logging.basicConfig(level=logging.INFO)

from generation.llm_factory import build_llm
from query.base import QueryUnderstandingStrategy
from query.query_factory import build_query_understanding

llm = build_llm()
rewriter = build_query_understanding(QueryUnderstandingStrategy.REWRITING, llm)

QUERIES = [
    "co z tą mocą cieplną?",
    "ile kosztuje ciepło?",
    "jak liczyć opłaty?",
    "kto ustala taryfy?",
    "jakie są przepisy o węzłach?",
]

print("\n=== QUERY REWRITING TEST ===\n")
for query in QUERIES:
    transformed = rewriter.transform(query)
    print(f"  ORG : {query}")
    print(f"  REW : {transformed[0]}")
    print()