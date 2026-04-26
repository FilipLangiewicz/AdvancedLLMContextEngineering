import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
from collections import defaultdict
import time
import logging
from pathlib import Path

from compression.base import CompressionStrategy
from config.settings import ChunkingStrategy, LLMProvider
from generation.embeddings_factory import build_embeddings
from generation.llm_factory import build_llm
from pipeline import build_pipeline
from query.base import QueryUnderstandingStrategy
from reranking.base import RerankingStrategy

from evaluation.test_questions import TEST_QUESTIONS
from evaluation.judge import LLMJudge

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

JUDGE_PROVIDER = LLMProvider.GOOGLE
JUDGE_MODEL = "gemini-2.5-flash"

INTER_QUERY_SLEEP = 0.5

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "experiment_results.csv"

CONFIGS = [
    {
        "label": "baseline",
        "chunking": ChunkingStrategy.STRUCTURE,
        "reranking": RerankingStrategy.ORIGINAL_ORDER,
        "compression": None,
        "query_understanding": None,
    },
    {
        "label": "semantic_chunk",
        "chunking": ChunkingStrategy.SEMANTIC,
        "reranking": RerankingStrategy.ORIGINAL_ORDER,
        "compression": None,
        "query_understanding": None,
    },
    {
        "label": "u_shape",
        "chunking": ChunkingStrategy.STRUCTURE,
        "reranking": RerankingStrategy.U_SHAPE_REORDER,
        "compression": None,
        "query_understanding": None,
    },
    {
        "label": "rewrite",
        "chunking": ChunkingStrategy.STRUCTURE,
        "reranking": RerankingStrategy.ORIGINAL_ORDER,
        "compression": None,
        "query_understanding": QueryUnderstandingStrategy.REWRITING,
    },
    {
        "label": "extractive",
        "chunking": ChunkingStrategy.STRUCTURE,
        "reranking": RerankingStrategy.ORIGINAL_ORDER,
        "compression": CompressionStrategy.EXTRACTIVE_FILTER,
        "query_understanding": None,
    },
    {
        "label": "hierarchical",
        "chunking": ChunkingStrategy.STRUCTURE,
        "reranking": RerankingStrategy.ORIGINAL_ORDER,
        "compression": CompressionStrategy.HIERARCHICAL_SUMMARY,
        "query_understanding": None,
    },
    {
        "label": "full_stack",
        "chunking": ChunkingStrategy.STRUCTURE,
        "reranking": RerankingStrategy.U_SHAPE_REORDER,
        "compression": CompressionStrategy.HIERARCHICAL_SUMMARY,
        "query_understanding": QueryUnderstandingStrategy.REWRITING,
    },
]


FIELDNAMES = [
    "config", "question_id", "category", "question",
    "answer", "strategy", "latency_s",
    "n_sources", "context_tokens", "answer_tokens",
    "faithfulness", "faith_explanation",
    "relevance", "rel_explanation",
    "completeness", "comp_explanation",
    "context_has_answer",
]


def load_completed() -> set[tuple[str, str]]:
    """Returns set of (config, question_id) pairs that already have valid results in CSV.
    Failed rows (faithfulness=0 or answer starts with ERROR) are NOT counted as completed,
    so they will be retried."""
    if not OUTPUT_PATH.exists():
        return set()
    completed = set()
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            answer = row.get("answer", "")
            try:
                faith = int(row.get("faithfulness", 0))
            except ValueError:
                faith = 0
            if faith == 0 or answer.startswith("ERROR"):
                continue
            completed.add((row["config"], row["question_id"]))
    return completed


def remove_failed_rows() -> int:
    """Rewrites CSV without failed rows. Returns number of rows removed."""
    if not OUTPUT_PATH.exists():
        return 0
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    
    valid_rows = []
    removed = 0
    for row in rows:
        answer = row.get("answer", "")
        try:
            faith = int(row.get("faithfulness", 0))
        except ValueError:
            faith = 0
        if faith == 0 or answer.startswith("ERROR"):
            removed += 1
            continue
        valid_rows.append(row)
    
    if removed > 0:
        with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(valid_rows)
    
    return removed


def main():
    removed = remove_failed_rows()
    if removed > 0:
        logger.info(f"Usunięto {removed} błędnych wierszy z poprzednich uruchomień")
    
    completed = load_completed()
    if completed:
        logger.info(f"Znaleziono {len(completed)} ukończonych par (config, question) — pomijam je")
    
    file_exists = OUTPUT_PATH.exists() and OUTPUT_PATH.stat().st_size > 0
    
    logger.info("Inicjalizacja embeddings i judge'a")
    embeddings = build_embeddings()
    judge_llm = build_llm(provider=JUDGE_PROVIDER, model=JUDGE_MODEL)
    judge = LLMJudge(judge_llm)

    mode = "a" if file_exists else "w"
    with open(OUTPUT_PATH, mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()

        for config in CONFIGS:
            label = config["label"]
            
            pending = [q for q in TEST_QUESTIONS if (label, q.id) not in completed]
            if not pending:
                logger.info(f"Konfiguracja {label} już ukończona — pomijam")
                continue
            
            logger.info(f"\n{'='*70}\nKonfiguracja: {label} | pozostało: {len(pending)}/{len(TEST_QUESTIONS)} pytań\n{'='*70}")

            pipeline = build_pipeline(
                embeddings=embeddings,
                chunking_strategy=config["chunking"],
                reranking_strategy=config["reranking"],
                compression_strategy=config["compression"],
                query_understanding_strategy=config["query_understanding"],
                with_cache=False,
            )

            for q in pending:
                logger.info(f"  [{q.id}] {q.question[:70]}")
                try:
                    start = time.time()
                    response = pipeline.run(q.question)
                    latency = time.time() - start

                    judge_result = judge.evaluate(
                        question=q.question,
                        context=response.context_text,
                        answer=response.answer,
                    )

                    row = {
                        "config": label,
                        "question_id": q.id,
                        "category": q.category.value,
                        "question": q.question,
                        "answer": response.answer,
                        "strategy": response.strategy,
                        "latency_s": round(latency, 2),
                        "n_sources": len(response.sources),
                        "context_tokens": len(response.context_text.split()),
                        "answer_tokens": len(response.answer.split()),
                        "faithfulness": judge_result.faithfulness,
                        "faith_explanation": judge_result.faithfulness_explanation,
                        "relevance": judge_result.relevance,
                        "rel_explanation": judge_result.relevance_explanation,
                        "completeness": judge_result.completeness,
                        "comp_explanation": judge_result.completeness_explanation,
                        "context_has_answer": judge_result.context_has_answer,
                    }
                    writer.writerow(row)
                    f.flush()
                    logger.info(
                        f"    F={judge_result.faithfulness} "
                        f"R={judge_result.relevance} "
                        f"C={judge_result.completeness} "
                        f"| ctx_has_ans={judge_result.context_has_answer} "
                        f"| {latency:.1f}s | ctx={row['context_tokens']}t ans={row['answer_tokens']}t"
                    )
                except Exception as e:
                    logger.error(f"    BŁĄD: {type(e).__name__}: {e}")
                    writer.writerow({
                        "config": label,
                        "question_id": q.id,
                        "category": q.category.value,
                        "question": q.question,
                        "answer": f"ERROR: {e}",
                        "strategy": "",
                        "latency_s": 0,
                        "n_sources": 0,
                        "context_tokens": 0,
                        "answer_tokens": 0,
                        "faithfulness": 0,
                        "faith_explanation": f"ERROR: {e}",
                        "relevance": 0,
                        "rel_explanation": f"ERROR: {e}",
                        "completeness": 0,
                        "comp_explanation": f"ERROR: {e}",
                        "context_has_answer": None,
                    })
                    f.flush()

                time.sleep(INTER_QUERY_SLEEP)

    logger.info(f"\nGotowe. Wyniki: {OUTPUT_PATH}")
    print_summary()


def print_summary():
    sums = defaultdict(lambda: {"f": 0, "r": 0, "c": 0, "lat": 0.0, "ctx": 0, "ans": 0, "n": 0})
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                faith = int(row["faithfulness"])
            except ValueError:
                faith = 0
            if faith == 0 or row["answer"].startswith("ERROR"):
                continue  # skip failed rows in summary
            cfg = row["config"]
            sums[cfg]["f"] += faith
            sums[cfg]["r"] += int(row["relevance"])
            sums[cfg]["c"] += int(row["completeness"])
            sums[cfg]["lat"] += float(row["latency_s"])
            sums[cfg]["ctx"] += int(row["context_tokens"])
            sums[cfg]["ans"] += int(row["answer_tokens"])
            sums[cfg]["n"] += 1

    print(f"\n{'='*88}")
    print(f"{'Config':<18} {'Faith':>7} {'Rel':>7} {'Comp':>7} {'Lat[s]':>9} {'CtxTok':>9} {'AnsTok':>9}  N")
    print('-' * 88)
    for cfg, s in sums.items():
        n = s["n"]
        if n == 0:
            continue
        print(
            f"{cfg:<18} "
            f"{s['f']/n:>7.2f} {s['r']/n:>7.2f} {s['c']/n:>7.2f} "
            f"{s['lat']/n:>9.2f} {s['ctx']/n:>9.0f} {s['ans']/n:>9.0f}  {n}"
        )
    print('=' * 88)


if __name__ == "__main__":
    main()