import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
logging.basicConfig(level=logging.INFO)

from langchain_huggingface import HuggingFaceEmbeddings
from retrieval.retriever import Retriever
from config.settings import ChunkingStrategy

embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-mpnet-base-v2")

QUERY = "Jakie ceny zawiera taryfa wytwórcy ciepła?"

print("\n=== STRUCTURE ===")
retriever_structure = Retriever(embeddings=embeddings, strategy=ChunkingStrategy.STRUCTURE)
results = retriever_structure.retrieve(QUERY, k=3)
for i, r in enumerate(results):
    print(f"\n[{i+1}] source={r.metadata.get('source')} page={r.metadata.get('page')} article={r.metadata.get('article_id','–')}")
    print(r.page_content[:300])

print("\n=== SEMANTIC ===")
retriever_semantic = Retriever(embeddings=embeddings, strategy=ChunkingStrategy.SEMANTIC)
results = retriever_semantic.retrieve(QUERY, k=3)
for i, r in enumerate(results):
    print(f"\n[{i+1}] source={r.metadata.get('source')} page={r.metadata.get('page')} article={r.metadata.get('article_id','–')}")
    print(r.page_content[:300])