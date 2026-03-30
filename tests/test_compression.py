import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
logging.basicConfig(level=logging.INFO)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from generation.llm_factory import build_llm
from retrieval.retriever import Retriever
from compression.base import CompressionStrategy
from compression.compression_factory import build_compressor
from config.settings import ChunkingStrategy

# ── SETUP ────────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-mpnet-base-v2")
llm = build_llm()

QUERIES = [
    "Jakie ceny zawiera taryfa wytwórcy ciepła?",
    "Co to jest zamówiona moc cieplna?",
]

# ── HELPER ───────────────────────────────────────────────────
def token_count(docs: list[Document]) -> int:
    return sum(len(d.page_content.split()) for d in docs)

def run_compression_test(strategy: CompressionStrategy):
    print(f"\n{'='*60}")
    print(f"COMPRESSION: {strategy.value.upper()}")
    print('='*60)

    retriever = Retriever(embeddings=embeddings, strategy=ChunkingStrategy.STRUCTURE)
    compressor = build_compressor(strategy=strategy, llm=llm)

    for query in QUERIES:
        docs = retriever.retrieve(query, k=5)
        tokens_before = token_count(docs)

        compressed = compressor.compress(query, docs)
        tokens_after = token_count(compressed)

        reduction = (1 - tokens_after / tokens_before) * 100 if tokens_before > 0 else 0

        print(f"\nQ: {query}")
        print(f"  Docs:   {len(docs)} → {len(compressed)}")
        print(f"  Tokens: ~{tokens_before} → ~{tokens_after} ({reduction:.1f}% reduction)")
        print(f"  --- Compressed content preview ---")
        for i, doc in enumerate(compressed):
            preview = doc.page_content[:200].replace('\n', ' ')
            print(f"  [{i+1}] {preview}...")

# ── BASELINE (no compression) ────────────────────────────────
print(f"\n{'='*60}")
print("COMPRESSION: NONE (baseline)")
print('='*60)

retriever = Retriever(embeddings=embeddings, strategy=ChunkingStrategy.STRUCTURE)
for query in QUERIES:
    docs = retriever.retrieve(query, k=5)
    print(f"\nQ: {query}")
    print(f"  Docs:   {len(docs)}")
    print(f"  Tokens: ~{token_count(docs)}")

# ── EXTRACTIVE ───────────────────────────────────────────────
run_compression_test(CompressionStrategy.EXTRACTIVE_FILTER)

# ── HIERARCHICAL ─────────────────────────────────────────────
run_compression_test(CompressionStrategy.HIERARCHICAL_SUMMARY)