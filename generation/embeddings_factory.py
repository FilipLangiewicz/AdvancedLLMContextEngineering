from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from config.settings import settings


def build_embeddings() -> Embeddings:
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)