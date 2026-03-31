import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
logging.basicConfig(level=logging.INFO)

from ingestion.pipeline import run_pipeline
from ingestion.vector_store import get_vector_store
from config.settings import settings, ChunkingStrategy
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-mpnet-base-v2")

# --- test structure ---
run_pipeline(strategy=ChunkingStrategy.STRUCTURE, recreate=True, collection=settings.qdrant_collection_structure_test, test=True, embeddings=embeddings)

store = get_vector_store(settings.qdrant_collection_structure_test, embeddings)
results = store.similarity_search("obliczanie taryfy dla ciepła", k=3)

print(f"\n--- Structure pipeline: Top 3 ---")
for i, r in enumerate(results):
    print(f"[{i+1}] page={r.metadata.get('page')} article={r.metadata.get('article_id')} chapter={r.metadata.get('chapter','')[:40]}")
    print(f"     {r.page_content[:200]}\n")

print("Pipeline test passed!")