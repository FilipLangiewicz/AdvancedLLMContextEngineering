from dataclasses import dataclass, field
from enum import Enum


class QuestionCategory(str, Enum):
    SHORT = "short"
    LONG = "long"
    NEGATIVE = "negative"


@dataclass
class TestQuestion:
    id: str
    question: str
    category: QuestionCategory
    notes: str = ""


TEST_QUESTIONS: list[TestQuestion] = [
    TestQuestion(
        id="S1",
        question="Jakie koszty przyjąć do kalkulacji taryfy dla ciepła?",
        category=QuestionCategory.SHORT,
    ),
    TestQuestion(
        id="S2",
        question="Jak ocenia się koszty uzasadnione przyjęte do taryfy dla ciepła?",
        category=QuestionCategory.SHORT,
    ),
    TestQuestion(
        id="S3",
        question="Jak ustala się koszt zakupu uprawnień do emisji CO2 przyjmowanych do taryfy dla ciepła?",
        category=QuestionCategory.SHORT,
    ),
    TestQuestion(
        id="S4",
        question="Jak określić współczynnik udziału opłat stałych w opłatach łącznych?",
        category=QuestionCategory.SHORT,
    ),

    TestQuestion(
        id="L1",
        question="Jak wyliczyć kwotę zwrotu z kapitału w taryfie dla ciepła przy wytwarzaniu ciepła przy użyciu paliwa węglowego?",
        category=QuestionCategory.LONG,
    ),
    TestQuestion(
        id="L2",
        question="Co należy przedstawić we wniosku o zatwierdzenie taryfy dla ciepła?",
        category=QuestionCategory.LONG,
        notes="Skrócone z oryginału — usunięto część o tworzeniu przykładowych kalkulacji, ponieważ wymagałaby kreatywnego wykroczenia poza dokument i zaniżałaby faithfulness.",
    ),
    TestQuestion(
        id="L3",
        question="Czy spółka dostarczająca ciepło odbiorcom musi stosować bonifikatę gdy występuje awaria sieci?",
        category=QuestionCategory.LONG,
    ),

    TestQuestion(
        id="N1",
        question="Jaka jest stawka VAT na energię cieplną w 2024 roku?",
        category=QuestionCategory.NEGATIVE,
        notes="Stawki VAT są w odrębnej ustawie podatkowej, nie w prawie energetycznym.",
    ),
    TestQuestion(
        id="N2",
        question="Jaka jest średnia cena ciepła systemowego w Polsce w 2024 roku?",
        category=QuestionCategory.NEGATIVE,
        notes="Konkretne ceny rynkowe nie znajdują się w aktach prawnych.",
    ),
    TestQuestion(
        id="N3",
        question="Jak rozliczyć podatek dochodowy od dochodów uzyskanych ze sprzedaży nadwyżki energii z mikroinstalacji fotowoltaicznej?",
        category=QuestionCategory.NEGATIVE,
        notes="Kwestie podatkowe dotyczące prosumentów PV są poza zakresem dostarczonych dokumentów.",
    ),
]


def get_questions(category: QuestionCategory | None = None) -> list[TestQuestion]:
    if category is None:
        return TEST_QUESTIONS
    return [q for q in TEST_QUESTIONS if q.category == category]