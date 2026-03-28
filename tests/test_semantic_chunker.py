import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import logging
logging.basicConfig(level=logging.INFO)
from tqdm import tqdm

from langchain_huggingface import HuggingFaceEmbeddings
from ingestion.loader import load_pdfs
from ingestion.cleaner import clean_documents
from ingestion.chunkers.semantic_chunker import SemanticDocChunker

logger = logging.getLogger(__name__)

logger.info("Loading embedding model (first run will download ~1.2GB)...")
embeddings = HuggingFaceEmbeddings(
    # model_name="intfloat/multilingual-e5-large", 
    model_name="paraphrase-multilingual-mpnet-base-v2", 
    show_progress=True)

docs = load_pdfs()
docs = clean_documents(docs)

docs = [d for d in docs if d.metadata['source'] == 'document2.pdf']
print(f"Testing on {len(docs)} pages")

chunker = SemanticDocChunker(embeddings=embeddings)
chunks = chunker.chunk(docs)

# --- Summary ---
print(f"\nTotal chunks: {len(chunks)}")

from collections import Counter
per_source = Counter(c.metadata['source'] for c in chunks)
per_type = Counter(c.metadata['chunk_type'] for c in chunks)

print("\nChunks per source:")
for src, count in per_source.items():
    print(f"  {src}: {count}")

print("\nChunks per type:")
for t, count in per_type.items():
    print(f"  {t}: {count}")

# --- Save first 50 chunks per source ---
output_dir = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(output_dir, exist_ok=True)

from collections import defaultdict
chunks_by_source = defaultdict(list)
for c in chunks:
    chunks_by_source[c.metadata['source']].append(c)

for source, source_chunks in chunks_by_source.items():
    sample = source_chunks[:50]
    data = [
        {
            "chunk_index": i,
            "text": c.page_content,
            "metadata": c.metadata
        }
        for i, c in enumerate(sample)
    ]
    filename = source.replace(".pdf", "_semantic_chunks.json")
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(sample)} chunks → tests/outputs/{filename}")

# --- Preview first chunk ---
print(f"\n--- First chunk preview ---")
c = chunks[0]
# print(f"Metadata keys: {c.metadata}")  
print(f"Source: {c.metadata['source']}")
print(f"Page: {c.metadata['page']}")
print(f"Type: {c.metadata['chunk_type']}")
print(f"Text (first 400 chars):\n{c.page_content[:400]}")