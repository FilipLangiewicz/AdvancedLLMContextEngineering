import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
logging.basicConfig(level=logging.INFO)

from langchain_huggingface import HuggingFaceEmbeddings
from retrieval.retriever import Retriever
from generation.generator import Generator
from generation.llm_factory import _LLM_REGISTRY, _DEFAULT_MODELS
from config.settings import ChunkingStrategy, LLMProvider

# ── FACTORY VERIFICATION ─────────────────────────────────────
print("\n=== REGISTRY CHECK ===")
for provider in LLMProvider:
    status = "OK" if provider in _LLM_REGISTRY else "MISSING IN REGISTRY"
    model = _DEFAULT_MODELS.get(provider, "?")
    print(f"  {provider.value:<12} -> {model:<35} [{status}]")

print("\n=== PROVIDER OVERRIDE TEST ===")
gen_groq = Generator(provider=LLMProvider.GROQ)
gen_groq_small = Generator(provider=LLMProvider.GROQ, model="llama-3.1-8b-instant")
print("  OK - Generator(provider=GROQ)")
print("  OK - Generator(provider=GROQ, model='llama-3.1-8b-instant')")
# ─────────────────────────────────────────────────────────────

embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-mpnet-base-v2")
generator = Generator()

QUERIES = [
    "Jakie ceny zawiera taryfa wytwórcy ciepła?",
    "Co to jest zamówiona moc cieplna?",
    "Jakie są zasady obliczania opłat za ciepło?",
]

for strategy in [ChunkingStrategy.STRUCTURE, ChunkingStrategy.SEMANTIC]:
    retriever = Retriever(embeddings=embeddings, strategy=strategy)
    print(f"\n{'='*60}")
    print(f"STRATEGIA: {strategy.value.upper()}")
    print('='*60)

    for query in QUERIES:
        docs = retriever.retrieve(query, k=5)
        response = generator.generate(query, docs, strategy=strategy.value)

        print(f"\nQ: {query}")
        print(f"A: {response.answer}")
        print("Źródła:")
        for s in response.sources:
            print(f"  - {s['source']}, s.{s['page']}, {s['article_id']}")
        print()