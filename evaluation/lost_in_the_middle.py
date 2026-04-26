import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import json
import logging
import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from compression.base import CompressionStrategy
from config.settings import LLMProvider
from generation.embeddings_factory import build_embeddings
from generation.generator import Generator
from generation.llm_factory import build_llm
from query.base import QueryUnderstandingStrategy
from reranking.base import RerankingStrategy
from reranking.reranking_factory import create_reranker
from retrieval.retriever import Retriever
from compression.compression_factory import build_compressor

from evaluation.test_questions import TEST_QUESTIONS, QuestionCategory
from evaluation.judge import LLMJudge

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

INTER_QUERY_SLEEP = 0.5

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "lost_in_middle_results.csv"

LITM_QUESTIONS = [q for q in TEST_QUESTIONS if q.category == QuestionCategory.SHORT][:3] + \
                 [q for q in TEST_QUESTIONS if q.category == QuestionCategory.LONG][:1]
# = S1, S2, S3, L1 — 4 pytania

K = 10  
POSITIONS = [0, 2, 4, 6, 9]  # pozycje, na których umieszczamy kluczowy chunk

MITIGATION_CONFIGS = [
    {
        "label": "no_mitigation",
        "reranking": RerankingStrategy.ORIGINAL_ORDER,
        "compression": None,
    },
    {
        "label": "u_shape",
        "reranking": RerankingStrategy.U_SHAPE_REORDER,
        "compression": None,
    },
    {
        "label": "brief_context",
        "reranking": RerankingStrategy.ORIGINAL_ORDER,
        "compression": CompressionStrategy.BRIEF_CONTEXT,
    },
]


KEY_CHUNK_SELECTOR_PROMPT = """Jesteś ekspertem prawa energetycznego.
Otrzymujesz pytanie i listę ponumerowanych fragmentów dokumentów prawnych.

Twoim zadaniem jest wskazać, KTÓRY fragment najlepiej odpowiada na pytanie — zawiera kluczową informację.

Zwróć WYŁĄCZNIE numer fragmentu (liczbę całkowitą od 1 do N), bez żadnego dodatkowego tekstu, znaków interpunkcyjnych ani wyjaśnień.

Jeśli żaden fragment nie zawiera odpowiedzi, zwróć 0."""


def select_key_chunk(llm, query: str, docs: list[Document]) -> int | None:
    """LLM wybiera index (0-based) chunka z odpowiedzią. Zwraca None jeśli żaden nie pasuje."""
    chunks_text = "\n\n".join(
        f"[{i+1}] {d.page_content[:600]}"  
        for i, d in enumerate(docs)
    )
    messages = [
        SystemMessage(content=KEY_CHUNK_SELECTOR_PROMPT),
        HumanMessage(content=f"PYTANIE: {query}\n\nFRAGMENTY:\n{chunks_text}"),
    ]
    response = llm.invoke(messages).content.strip()
    
    for token in response.replace(".", " ").replace(",", " ").split():
        try:
            n = int(token)
            if n == 0:
                return None
            if 1 <= n <= len(docs):
                return n - 1  # konwersja na 0-based
        except ValueError:
            continue
    
    logger.warning(f"Nie udało się sparsować numeru z: '{response[:100]}'")
    return None


def reposition(docs: list[Document], key_idx: int, target_position: int) -> list[Document]:
    """Przenosi docs[key_idx] na pozycję target_position. Reszta zachowuje względną kolejność."""
    others = [d for i, d in enumerate(docs) if i != key_idx]
    key_doc = docs[key_idx]
    others.insert(target_position, key_doc)
    return others


def main():
    logger.info("Inicjalizacja")
    embeddings = build_embeddings()
    main_llm = build_llm()  # Llama 3.3 70b — do generacji
    judge_llm = build_llm(provider=LLMProvider.GOOGLE, model="gemini-2.5-flash")
    judge = LLMJudge(judge_llm)
    
    retriever = Retriever(embeddings)
    generator = Generator()
    
    fieldnames = [
        "question_id", "question", "mitigation",
        "key_position", "actual_key_idx_in_retrieval",
        "answer", "answer_tokens",
        "faithfulness", "relevance", "completeness",
        "comp_explanation", "context_has_answer",
        "latency_s",
    ]
    
    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    
        question_setups = {}
        for q in LITM_QUESTIONS:
            logger.info(f"\n[{q.id}] {q.question}")
            docs = retriever.retrieve(q.question, k=K)
            
            if len(docs) < K:
                logger.warning(f"  Tylko {len(docs)} chunków — pomijam")
                continue
            
            key_idx = select_key_chunk(judge_llm, q.question, docs)
            if key_idx is None:
                logger.warning(f"  Judge nie wskazał key chunk — pomijam")
                continue
            
            logger.info(f"  Key chunk: pozycja {key_idx} w retrievalu (0-indexed)")
            logger.info(f"  Preview: {docs[key_idx].page_content[:120]}...")
            question_setups[q.id] = (q, docs, key_idx)
        
        if not question_setups:
            logger.error("Brak pytań z poprawnie wybranym key chunkiem.")
            return
        
        total = len(question_setups) * len(MITIGATION_CONFIGS) * len(POSITIONS)
        logger.info(f"\nUruchamiam {total} eksperymentów ({len(question_setups)} pytań x {len(MITIGATION_CONFIGS)} mitigations × {len(POSITIONS)} pozycji)")
        
        for q_id, (q, docs, key_idx) in question_setups.items():
            for mitigation in MITIGATION_CONFIGS:
                reranker = create_reranker(mitigation["reranking"])
                compressor = build_compressor(mitigation["compression"], main_llm) if mitigation["compression"] else None
                
                for target_pos in POSITIONS:
                    logger.info(f"\n  [{q_id}] mit={mitigation['label']} pos={target_pos}")
                    try:
                        repositioned = reposition(docs, key_idx, target_pos)
                        
                        ranked = reranker.rerank(repositioned)
                        
                        if compressor:
                            ranked = compressor.compress(q.question, ranked)
                        
                        start = time.time()
                        response = generator.generate(q.question, ranked, strategy=mitigation["label"])
                        latency = time.time() - start
                        
                        judge_result = judge.evaluate(
                            question=q.question,
                            context=response.context_text,
                            answer=response.answer,
                        )
                        
                        writer.writerow({
                            "question_id": q_id,
                            "question": q.question,
                            "mitigation": mitigation["label"],
                            "key_position": target_pos,
                            "actual_key_idx_in_retrieval": key_idx,
                            "answer": response.answer,
                            "answer_tokens": len(response.answer.split()),
                            "faithfulness": judge_result.faithfulness,
                            "relevance": judge_result.relevance,
                            "completeness": judge_result.completeness,
                            "comp_explanation": judge_result.completeness_explanation,
                            "context_has_answer": judge_result.context_has_answer,
                            "latency_s": round(latency, 2),
                        })
                        f.flush()
                        logger.info(f"    F={judge_result.faithfulness} R={judge_result.relevance} C={judge_result.completeness}")
                    
                    except Exception as e:
                        logger.error(f"    BŁĄD: {type(e).__name__}: {e}")
                        writer.writerow({
                            "question_id": q_id,
                            "question": q.question,
                            "mitigation": mitigation["label"],
                            "key_position": target_pos,
                            "actual_key_idx_in_retrieval": key_idx,
                            "answer": f"ERROR: {e}",
                            "answer_tokens": 0,
                            "faithfulness": 0,
                            "relevance": 0,
                            "completeness": 0,
                            "comp_explanation": f"ERROR: {e}",
                            "context_has_answer": None,
                            "latency_s": 0,
                        })
                        f.flush()
                    
                    time.sleep(INTER_QUERY_SLEEP)
    
    logger.info(f"\nGotowe. Wyniki: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()