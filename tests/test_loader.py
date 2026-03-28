import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
logging.basicConfig(level=logging.INFO)

from ingestion.loader import load_pdfs

docs = load_pdfs()
print(f"\nTotal pages: {len(docs)}")
print(f"\n--- First page ---")
print(f"Source: {docs[0].metadata['source']}")
print(f"Page: {docs[0].metadata['page']}")
print(f"Text (first 300 chars):\n{docs[0].page_content[:300]}")