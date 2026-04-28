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
KEY_CHUNKS_PATH = OUTPUT_DIR / "lost_in_middle_key_chunks.json"

LITM_QUESTION_IDS = ["S1", "S2", "L3"]
LITM_QUESTIONS = [q for q in TEST_QUESTIONS if q.id in LITM_QUESTION_IDS]

K = 10
POSITIONS = [0, 2, 4, 6, 9]

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

FIELDNAMES = [
    "question_id", "question", "mitigation",
    "key_position", "actual_key_idx_in_retrieval",
    "answer", "answer_tokens",
    "faithfulness", "relevance", "completeness",
    "comp_explanation", "context_has_answer",
    "latency_s",
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
                return n - 1
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


def load_completed_runs() -> set[tuple[str, str, int]]:
    """(question_id, mitigation, key_position) trójki które są już ukończone i validne."""
    if not OUTPUT_PATH.exists():
        return set()
    completed = set()
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                faith = int(row.get("faithfulness", 0))
            except ValueError:
                faith = 0
            answer = row.get("answer", "")
            if faith == 0 or answer.startswith("ERROR"):
                continue
            completed.add((
                row["question_id"],
                row["mitigation"],
                int(row["key_position"]),
            ))
    return completed


def remove_failed_rows() -> int:
    """Usuwa błędne wiersze z CSV. Zwraca liczbę usuniętych."""
    if not OUTPUT_PATH.exists():
        return 0
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    
    valid = []
    removed = 0
    for r in rows:
        try:
            faith = int(r.get("faithfulness", 0))
        except ValueError:
            faith = 0
        if faith == 0 or r.get("answer", "").startswith("ERROR"):
            removed += 1
            continue
        valid.append(r)
    
    if removed > 0:
        with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES)
            w.writeheader()
            w.writerows(valid)
    
    return removed


def load_key_chunks_cache() -> dict[str, dict]:
    """Ładuje wyniki selekcji key chunk: question_id -> {key_idx, retrieved_docs (jako listy contentów)}."""
    if not KEY_CHUNKS_PATH.exists():
        return {}
    with open(KEY_CHUNKS_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_key_chunks_cache(cache: dict[str, dict]) -> None:
    with open(KEY_CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def docs_to_serializable(docs: list[Document]) -> list[dict]:
    return [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]


def docs_from_serializable(data: list[dict]) -> list[Document]:
    return [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data]


def main():
    removed = remove_failed_rows()
    if removed > 0:
        logger.info(f"Usunięto {removed} błędnych wierszy z poprzednich uruchomień")
    
    completed_runs = load_completed_runs()
    if completed_runs:
        logger.info(f"Znaleziono {len(completed_runs)} ukończonych eksperymentów — pomijam je")
    
    file_exists = OUTPUT_PATH.exists() and OUTPUT_PATH.stat().st_size > 0
    
    logger.info("Inicjalizacja")
    embeddings = build_embeddings()
    main_llm = build_llm()
    judge_llm = build_llm(provider=LLMProvider.GOOGLE, model="gemini-2.5-flash")
    judge = LLMJudge(judge_llm)
    
    retriever = Retriever(embeddings)
    generator = Generator()
    
    key_chunks_cache = load_key_chunks_cache()
    question_setups = {}
    
    for q in LITM_QUESTIONS:
        if q.id in key_chunks_cache:
            cached = key_chunks_cache[q.id]
            docs = docs_from_serializable(cached["docs"])
            key_idx = cached["key_idx"]
            logger.info(f"\n[{q.id}] cache hit — key chunk pos {key_idx}")
            question_setups[q.id] = (q, docs, key_idx)
            continue
        
        logger.info(f"\n[{q.id}] {q.question}")
        docs = retriever.retrieve(q.question, k=K)
        
        if len(docs) < K:
            logger.warning(f"  Tylko {len(docs)} chunków — pomijam")
            continue
        
        try:
            key_idx = select_key_chunk(judge_llm, q.question, docs)
        except Exception as e:
            logger.error(f"  Selekcja key chunk padła ({type(e).__name__}: {str(e)[:80]}) — pomijam")
            continue
        
        if key_idx is None:
            logger.warning(f"  Judge nie wskazał key chunk — pomijam")
            continue
        
        logger.info(f"  Key chunk: pozycja {key_idx} w retrievalu (0-indexed)")
        logger.info(f"  Preview: {docs[key_idx].page_content[:120]}...")
        
        key_chunks_cache[q.id] = {
            "key_idx": key_idx,
            "docs": docs_to_serializable(docs),
        }
        save_key_chunks_cache(key_chunks_cache)
        
        question_setups[q.id] = (q, docs, key_idx)
    
    if not question_setups:
        logger.error("Brak pytań z poprawnie wybranym key chunkiem.")
        return
    
    pending = []
    for q_id, (q, docs, key_idx) in question_setups.items():
        for mit in MITIGATION_CONFIGS:
            for pos in POSITIONS:
                if (q_id, mit["label"], pos) in completed_runs:
                    continue
                pending.append((q_id, q, docs, key_idx, mit, pos))
    
    total_planned = len(question_setups) * len(MITIGATION_CONFIGS) * len(POSITIONS)
    logger.info(f"\nZaplanowane: {total_planned} | Pozostało: {len(pending)}")
    
    if not pending:
        logger.info("Wszystko już ukończone.")
        print_summary()
        return
    
    mode = "a" if file_exists else "w"
    with open(OUTPUT_PATH, mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        
        # Cache rerankerów i kompresorów per mitigation
        mitigation_cache = {}
        for mit in MITIGATION_CONFIGS:
            reranker = create_reranker(mit["reranking"])
            compressor = build_compressor(mit["compression"], main_llm) if mit["compression"] else None
            mitigation_cache[mit["label"]] = (reranker, compressor)
        
        for i, (q_id, q, docs, key_idx, mit, pos) in enumerate(pending, 1):
            logger.info(f"\n[{i}/{len(pending)}] {q_id} | mit={mit['label']} | pos={pos}")
            reranker, compressor = mitigation_cache[mit["label"]]
            
            try:
                repositioned = reposition(docs, key_idx, pos)
                ranked = reranker.rerank(repositioned)
                
                if compressor:
                    ranked = compressor.compress(q.question, ranked)
                
                start = time.time()
                response = generator.generate(q.question, ranked, strategy=mit["label"])
                latency = time.time() - start
                
                judge_result = judge.evaluate(
                    question=q.question,
                    context=response.context_text,
                    answer=response.answer,
                )
                
                writer.writerow({
                    "question_id": q_id,
                    "question": q.question,
                    "mitigation": mit["label"],
                    "key_position": pos,
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
                logger.error(f"    BŁĄD: {type(e).__name__}: {str(e)[:120]}")
                writer.writerow({
                    "question_id": q_id,
                    "question": q.question,
                    "mitigation": mit["label"],
                    "key_position": pos,
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
    print_summary()


def print_summary():
    """Średnie completeness per mitigation × position."""
    if not OUTPUT_PATH.exists():
        return
    
    from collections import defaultdict
    cells = defaultdict(list)
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                c = int(row["completeness"])
            except ValueError:
                continue
            if c == 0:
                continue
            cells[(row["mitigation"], int(row["key_position"]))].append(c)
    
    if not cells:
        return
    
    mitigations = sorted({k[0] for k in cells.keys()})
    positions = sorted({k[1] for k in cells.keys()})
    
    print(f"\n{'='*72}")
    print(f"Completeness — średnia per mitigation × pozycja")
    print('-' * 72)
    header = f"{'Mitigation':<18}" + " ".join(f"{f'pos={p}':>9}" for p in positions) + f"{'avg':>8}"
    print(header)
    print('-' * 72)
    for mit in mitigations:
        row_vals = []
        all_vals = []
        for p in positions:
            vals = cells.get((mit, p), [])
            if vals:
                row_vals.append(f"{sum(vals)/len(vals):>9.2f}")
                all_vals.extend(vals)
            else:
                row_vals.append(f"{'—':>9}")
        avg = sum(all_vals)/len(all_vals) if all_vals else 0
        print(f"{mit:<18}" + " ".join(row_vals) + f"{avg:>8.2f}")
    print('=' * 72)


if __name__ == "__main__":
    main()