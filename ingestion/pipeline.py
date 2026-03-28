import logging
import argparse
from langchain_huggingface import HuggingFaceEmbeddings

from ingestion.loader import load_pdfs
from ingestion.cleaner import clean_documents
from ingestion.chunkers.structure_chunker import StructureChunker
from ingestion.chunkers.semantic_chunker import SemanticDocChunker
from ingestion.vector_store import upsert_chunks
from config.settings import settings, ChunkingStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_pipeline(strategy: ChunkingStrategy = settings.chunking_strategy, recreate: bool = False, collection: str = None, test: bool = False, embeddings: HuggingFaceEmbeddings = None):
    
    if collection is None:
        collection = (
            settings.qdrant_collection_structure
            if strategy == ChunkingStrategy.STRUCTURE
            else settings.qdrant_collection_semantic
        )

    logger.info(f"Starting pipeline | strategy={strategy} | collection={collection} | recreate={recreate}")

    # 1. Load & clean
    docs = load_pdfs()
    if test:
        docs = [d for d in docs if d.metadata["source"] == "document2.pdf"]  # --- test with one document ---
    docs = clean_documents(docs)

    # 2. Embeddings
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-mpnet-base-v2")

    # 3. Chunk + upsert
    if strategy == ChunkingStrategy.STRUCTURE:
        chunker = StructureChunker()
    else:
        chunker = SemanticDocChunker(embeddings=embeddings)

    chunks = chunker.chunk(docs)
    logger.info(f"Total chunks: {len(chunks)}")

    upsert_chunks(chunks, collection, embeddings, recreate=recreate)
    logger.info(f"Pipeline done → {collection}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["structure", "semantic"], default=settings.chunking_strategy.value)
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate collection before upsert")
    parser.add_argument("--collection", type=str, default=None, help="Specify collection name (overrides default from settings)")
    parser.add_argument("--test", action="store_true", help="Run in test mode with a single document")
    args = parser.parse_args()

    run_pipeline(
        strategy=ChunkingStrategy(args.strategy),
        recreate=args.recreate,
        test=args.test,
        collection=args.collection
    )