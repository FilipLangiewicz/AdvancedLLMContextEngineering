from pydantic_settings import BaseSettings
from enum import Enum


class ChunkingStrategy(str, Enum):
    SEMANTIC = "semantic"
    STRUCTURE = "structure"


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class Settings(BaseSettings):
    # LLM
    llm_provider: LLMProvider = LLMProvider.OPENAI
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.0

    # Embeddings
    embedding_model: str = "text-embedding-3-small"

    # Ingestion
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.STRUCTURE
    chunk_size: int = 1000
    chunk_overlap: int = 150
    pdf_dir: str = "data/pdf"

    # Qdrant
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection_structure: str = "legal_docs_structure"
    qdrant_collection_semantic: str = "legal_docs_semantic"
    qdrant_collection_structure_test: str = "legal_docs_structure_test"
    qdrant_collection_semantic_test: str = "legal_docs_semantic_test"

    # API Keys
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""

    class Config:
        env_file = ".env"


settings = Settings()