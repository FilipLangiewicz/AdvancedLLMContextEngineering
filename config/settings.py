from pydantic_settings import BaseSettings
from enum import Enum
from compression.base import CompressionStrategy
from query.base import QueryUnderstandingStrategy
from reranking.base import RerankingStrategy


class ChunkingStrategy(str, Enum):
    SEMANTIC = "semantic"
    STRUCTURE = "structure"


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROQ = "groq"


class Settings(BaseSettings):
    # LLM
    llm_provider: LLMProvider = LLMProvider.GROQ
    llm_temperature: float = 0.0
    
    # Models per provider
    groq_model: str
    google_model: str
    openai_model: str
    anthropic_model: str

    # Embeddings
    embedding_model: str = "paraphrase-multilingual-mpnet-base-v2"

    # Ingestion
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.STRUCTURE
    chunk_size: int = 1000
    chunk_overlap: int = 150
    pdf_dir: str = "data/pdf"
    
    # Retrieval
    retrieval_k: int = 5
    
    # Compression
    compression_strategy: CompressionStrategy | None = None
    
    # Query Understanding
    query_understanding_strategy: QueryUnderstandingStrategy | None = QueryUnderstandingStrategy.REWRITING
    
    # Reranking
    reranking_strategy: RerankingStrategy = RerankingStrategy.ORIGINAL_ORDER

    # Semantic cache
    cache_threshold: float = 0.92
    
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
    groq_api_key: str = ""

    class Config:
        env_file = ".env"


settings = Settings()