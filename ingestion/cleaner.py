import re
import logging
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Patterns to remove
_KANCELARIA_HEADER = re.compile(r'©Kancelaria Sejmu\s+s\.\s*\d+/\d+', re.IGNORECASE)
_DATE_STAMP = re.compile(r'\d{2}\.\d{2}\.\d{4}\s*\n')
_DZ_U_HEADER = re.compile(r'Dziennik Ustaw\s*–?\s*\d*\s*–?\s*Poz\.\s*\d+')
_POZ_HEADER = re.compile(r'Poz\.\s*\d+\s*\n')
_FORM_FEED = re.compile(r'\x0c')
_MULTIPLE_NEWLINES = re.compile(r'\n{3,}')
_MULTIPLE_SPACES = re.compile(r'[ \t]{2,}')
_HYPHEN_NEWLINE = re.compile(r'-\n')


def clean_text(text: str) -> str:
    text = _KANCELARIA_HEADER.sub('', text)
    text = _DATE_STAMP.sub('', text)
    text = _DZ_U_HEADER.sub('', text)
    text = _POZ_HEADER.sub('', text)
    text = _FORM_FEED.sub('\n', text)
    text = _HYPHEN_NEWLINE.sub('', text)
    text = _MULTIPLE_SPACES.sub(' ', text)
    text = _MULTIPLE_NEWLINES.sub('\n\n', text)
    return text.strip()


def clean_documents(documents: List[Document]) -> List[Document]:
    cleaned = []
    for doc in documents:
        cleaned_text = clean_text(doc.page_content)
        if not cleaned_text:
            logger.debug(f"Skipping empty document after cleaning: {doc.metadata}")
            continue
        cleaned.append(Document(
            page_content=cleaned_text,
            metadata=doc.metadata
        ))
    logger.info(f"Cleaned {len(documents)} documents → {len(cleaned)} non-empty documents.")
    return cleaned