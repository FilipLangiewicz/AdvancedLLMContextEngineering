import json
import logging
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


FAITHFULNESS_PROMPT = """Jesteś ekspertem oceniającym jakość odpowiedzi systemu RAG w domenie polskiego prawa energetycznego.

Twoim zadaniem jest ocena FAITHFULNESS — czy każde stwierdzenie w odpowiedzi wynika z dostarczonego kontekstu.

Skala oceny:
- 5: Wszystkie stwierdzenia w odpowiedzi w pełni wynikają z kontekstu. Brak halucynacji.
- 4: Większość stwierdzeń wynika z kontekstu, drobne nieścisłości lub interpretacje.
- 3: Część stwierdzeń wynika z kontekstu, są też informacje spoza kontekstu.
- 2: Niewiele stwierdzeń wynika z kontekstu, znaczące halucynacje.
- 1: Odpowiedź nie wynika z kontekstu lub jest całkowicie zmyślona.

WAŻNE: Jeśli odpowiedź mówi wprost "nie znaleziono informacji w dokumentach" lub równoważnie, a kontekst rzeczywiście nie zawiera odpowiedzi — oceń jako 5 (system poprawnie odmówił halucynacji).

Zwróć WYŁĄCZNIE poprawny JSON, bez żadnych dodatkowych komentarzy ani markdown:
{"score": <liczba 1-5>, "explanation": "<krótkie uzasadnienie po polsku, max 2 zdania>"}"""


RELEVANCE_PROMPT = """Jesteś ekspertem oceniającym jakość odpowiedzi systemu RAG w domenie polskiego prawa energetycznego.

Twoim zadaniem jest ocena ANSWER RELEVANCE — czy odpowiedź faktycznie odnosi się do zadanego pytania.

Skala oceny:
- 5: Odpowiedź w pełni i bezpośrednio adresuje pytanie. Brak zbędnych dygresji.
- 4: Odpowiedź adresuje pytanie, drobne odchylenia od tematu.
- 3: Odpowiedź częściowo adresuje pytanie — brakuje istotnych elementów lub jest dużo zbędnej treści.
- 2: Odpowiedź marginalnie odnosi się do pytania.
- 1: Odpowiedź nie odnosi się do pytania.

WAŻNE: Jeśli pytanie nie ma odpowiedzi w kontekście, a odpowiedź mówi to wprost — oceń jako 5 (system poprawnie zaadresował brak danych).

Zwróć WYŁĄCZNIE poprawny JSON, bez żadnych dodatkowych komentarzy ani markdown:
{"score": <liczba 1-5>, "explanation": "<krótkie uzasadnienie po polsku, max 2 zdania>"}"""

COMPLETENESS_PROMPT = """Jesteś ekspertem oceniającym jakość odpowiedzi systemu RAG w domenie polskiego prawa energetycznego.

Twoim zadaniem jest ocena COMPLETENESS — czy odpowiedź zawiera konkretną, merytoryczną treść odpowiadającą na pytanie.

Najpierw sam oceń: czy DOSTARCZONY KONTEKST zawiera informacje pozwalające odpowiedzieć na pytanie?

PRZYPADEK A — kontekst ZAWIERA informacje istotne dla pytania:
- 5: Odpowiedź pełna — wykorzystuje wszystkie istotne informacje z kontekstu, podaje konkretne fakty (numery artykułów, definicje, procedury, wartości).
- 4: Odpowiedź w większości pełna, drobne braki.
- 3: Odpowiedź częściowa — pomija istotne elementy obecne w kontekście.
- 2: Odpowiedź powierzchowna lub bardzo krótka mimo dostępnych informacji.
- 1: Model odmówił odpowiedzi ("nie znalazłem informacji") MIMO że kontekst zawiera odpowiedź — to poważny błąd retrievala lub generacji.

PRZYPADEK B — kontekst NIE zawiera informacji istotnych dla pytania (lub pytanie wykracza poza dziedzinę dokumentów):
- 5: Model poprawnie odmówił odpowiedzi i wskazał brak informacji w dokumentach.
- 3: Model próbował odpowiedzieć z niepełnych przesłanek, ale nie halucynował.
- 1: Model halucynował konkretne fakty, których nie ma w kontekście.

Zwróć WYŁĄCZNIE poprawny JSON, bez markdown:
{"score": <liczba 1-5>, "context_has_answer": <true|false>, "explanation": "<krótkie uzasadnienie po polsku, max 2 zdania>"}"""


@dataclass
class JudgeResult:
    faithfulness: int
    faithfulness_explanation: str
    relevance: int
    relevance_explanation: str
    completeness: int                  
    completeness_explanation: str      
    context_has_answer: bool | None = None

class LLMJudge:
    """LLM-as-a-Judge dla ewaluacji RAG bez ground truth.
    Oceniane są dwie metryki: faithfulness i answer relevance."""

    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def evaluate(self, question: str, context: str, answer: str) -> JudgeResult:
        faith = self._evaluate_one(FAITHFULNESS_PROMPT, question, context, answer)
        rel = self._evaluate_one(RELEVANCE_PROMPT, question, context, answer)
        comp = self._evaluate_one(COMPLETENESS_PROMPT, question, context, answer)
        return JudgeResult(
            faithfulness=faith["score"],
            faithfulness_explanation=faith["explanation"],
            relevance=rel["score"],
            relevance_explanation=rel["explanation"],
            completeness=comp["score"],
            completeness_explanation=comp["explanation"],
            context_has_answer=comp.get("context_has_answer"),
        )

    def _evaluate_one(self, system_prompt: str, question: str, context: str, answer: str) -> dict:
        user_msg = (
            f"PYTANIE:\n{question}\n\n"
            f"KONTEKST (fragmenty dokumentów dostarczone do modelu):\n{context}\n\n"
            f"ODPOWIEDŹ MODELU:\n{answer}\n\n"
            f"Oceń odpowiedź zgodnie z instrukcją. Zwróć tylko JSON."
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_msg),
        ]
        response = self._llm.invoke(messages)
        return self._parse_json(response.content)

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]
        try:
            parsed = json.loads(text)
            return {
                "score": int(parsed.get("score", 0)),
                "explanation": str(parsed.get("explanation", "")),
                "context_has_answer": parsed.get("context_has_answer"),  # może być None
            }
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse judge response: {text[:200]}")
            return {"score": 0, "explanation": f"PARSE_ERROR: {str(e)[:100]}", "context_has_answer": None}
        
