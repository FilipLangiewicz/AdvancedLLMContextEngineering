import logging
from typing import List

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from config.settings import settings

logger = logging.getLogger(__name__)


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int):
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info(f"Created collection: {collection_name}")
    else:
        logger.info(f"Collection already exists: {collection_name}")


def get_vector_store(collection_name: str, embeddings) -> QdrantVectorStore:
    client = get_qdrant_client()
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )


def upsert_chunks(chunks: List[Document], collection_name: str, embeddings, recreate: bool = False) -> int:
    client = get_qdrant_client()
    sample_vector = embeddings.embed_query("test")

    if recreate:
        existing = [c.name for c in client.get_collections().collections]
        if collection_name in existing:
            client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")

    ensure_collection(client, collection_name, vector_size=len(sample_vector))
    store = QdrantVectorStore(client=client, collection_name=collection_name, embedding=embeddings)
    store.add_documents(chunks)
    logger.info(f"Upserted {len(chunks)} chunks → {collection_name}")
    return len(chunks)