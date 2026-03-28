import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
logging.basicConfig(level=logging.INFO)

from langchain_huggingface import HuggingFaceEmbeddings
from ingestion.loader import load_pdfs
from ingestion.cleaner import clean_documents
from ingestion.chunkers.structure_chunker import StructureChunker
from ingestion.vector_store import upsert_chunks, get_vector_store
from config.settings import settings

# --- embeddings ---
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-mpnet-base-v2")

docs = load_pdfs()
docs = clean_documents(docs)
docs = [d for d in docs if d.metadata["source"] == "document2.pdf"]
print(f"Loaded: {len(docs)} pages")

# --- chunk ---
chunker = StructureChunker()
chunks = chunker.chunk(docs)
print(f"Chunks: {len(chunks)}")

# --- upsert ---
n = upsert_chunks(chunks, settings.qdrant_collection_structure_test, embeddings, recreate=True)
print(f"Upserted: {n} chunks")

# --- test similarity search ---
store = get_vector_store(settings.qdrant_collection_structure_test, embeddings)
results = store.similarity_search("taryfa dla ciepła", k=3)

print(f"\n--- Top 3 results ---")
for i, r in enumerate(results):
    print(f"\n[{i+1}] source={r.metadata['source']} page={r.metadata.get('page')} article={r.metadata.get('article_id')}")
    print(r.page_content[:300])