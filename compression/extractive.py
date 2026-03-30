import logging
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

from compression.base import BaseContextCompressor

logger = logging.getLogger(__name__)


class ExtractiveFilterCompressor(BaseContextCompressor):

    def __init__(self, llm: BaseChatModel):
        self._compressor = LLMChainExtractor.from_llm(llm)

    def compress(self, query: str, docs: list[Document]) -> list[Document]:
        compressed = []
        for doc in docs:
            result = self._compressor.compress_documents([doc], query)
            if result:
                compressed.extend(result)
            else:
                logger.info(f"Chunk fully filtered out | source={doc.metadata.get('source')} | page={doc.metadata.get('page')}")

        original_tokens = sum(len(d.page_content.split()) for d in docs)
        compressed_tokens = sum(len(d.page_content.split()) for d in compressed)
        logger.info(f"Extractive compression | docs: {len(docs)}→{len(compressed)} | tokens: ~{original_tokens}→~{compressed_tokens}")
        return compressed