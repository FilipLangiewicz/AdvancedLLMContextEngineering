import logging
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from query.base import BaseQueryUnderstanding

from config.prompts import REWRITER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class QueryRewriter(BaseQueryUnderstanding):

    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def transform(self, query: str) -> list[str]:
        messages = [
            SystemMessage(content=REWRITER_SYSTEM_PROMPT),
            HumanMessage(content=f"Oryginalne zapytanie: {query}\n\nPrzepisane zapytanie:"),        
        ]
        response = self._llm.invoke(messages)
        rewritten = response.content.strip()
        logger.info(f"Query rewritten | original='{query}' | rewritten='{rewritten}'")
        return [rewritten]