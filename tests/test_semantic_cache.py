import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
logging.basicConfig(level=logging.INFO)

from langchain_huggingface import HuggingFaceEmbeddings
from cache.semantic_cache import SemanticCache

embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-mpnet-base-v2")
cache = SemanticCache(embeddings=embeddings, threshold=0.85)

print("\n=== SEMANTIC CACHE TEST ===\n")

# Empty cache
result = cache.get("ile kosztuje ciepło?")
print(f"Empty cache lookup:     {'HIT' if result else 'MISS'} (expected: MISS)")

# Add entry
cache.set(
    query="ile kosztuje ciepło?",
    answer="Cena ciepła wynosi X zł/GJ.",
    sources=[{"source": "document2.pdf", "page": 3}],
    strategy="structure",
)

QUERIES = [
    ("ile kosztuje ciepło?",                            "HIT"),
    ("jaka jest cena energii cieplnej?",                "HIT?"),
    ("ile płacimy za ogrzewanie?",                      "HIT?"),
    ("jakie są przepisy dotyczące węzłów ciepłowniczych?", "MISS"),
    ("co to jest zamówiona moc cieplna?",               "MISS"),
]

print()
for query, expected in QUERIES:
    result = cache.get(query)
    status = "HIT" if result else "MISS"
    print(f"  [{status}] (expected: {expected}) — '{query}'")

print(f"\nCache size: {cache.size}")