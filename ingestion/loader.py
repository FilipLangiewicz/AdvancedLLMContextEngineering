import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from config.settings import settings

logger = logging.getLogger(__name__)


def load_pdfs(pdf_dir: str = settings.pdf_dir) -> List[Document]:
    pdf_path = Path(pdf_dir)
    pdf_files = list(pdf_path.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in directory: {pdf_dir}")

    all_docs: List[Document] = []

    for pdf_file in sorted(pdf_files):
        logger.info(f"Loading: {pdf_file.name}")
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = pdf_file.name
            doc.metadata["file_path"] = str(pdf_file)
            doc.metadata["page"] = doc.metadata.get("page", 0) + 1

        all_docs.extend(docs)
        logger.info(f"  → {len(docs)} pages from {pdf_file.name}")

    logger.info(f"Total: {len(all_docs)} pages from {len(pdf_files)} files.")
    return all_docs