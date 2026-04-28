import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import logging
import time
from collections import defaultdict
from pathlib import Path

from datasets import Dataset
from ragas import evaluate, EvaluationDataset
from ragas.metrics import LLMContextPrecisionWithoutReference, ContextRelevance
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from config.settings import settings
from generation.embeddings_factory import build_embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

INPUT_PATH = Path(__file__).parent / "outputs" / "experiment_results.csv"
OUTPUT_PATH = Path(__file__).parent / "outputs" / "ragas_results.csv"


def parse_context_to_chunks(context_text: str) -> list[str]:
    """Pipeline buduje kontekst jako '[1] header:\n content\n\n[2] header:\n content...'.
    Dzielimy po pustej linii — to są nasze fragmenty."""
    if not context_text or not context_text.strip():
        return []
    chunks = [c.strip() for c in context_text.split("\n\n") if c.strip()]
    return chunks


def load_rows() -> list[dict]:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Brak pliku: {INPUT_PATH}")
    
    with open(INPUT_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    
    valid = []
    skipped_error = 0
    skipped_empty = 0
    for r in rows:
        try:
            faith = int(r.get("faithfulness", 0))
        except ValueError:
            faith = 0
        if faith == 0 or r.get("answer", "").startswith("ERROR"):
            skipped_error += 1
            continue
        # Wymaga niepustego context_text
        if not r.get("context_text", "").strip():
            skipped_empty += 1
            continue
        valid.append(r)
    
    logger.info(f"Wczytano {len(rows)} wierszy")
    logger.info(f"  Validnych: {len(valid)}")
    logger.info(f"  Pominiętych (error): {skipped_error}")
    logger.info(f"  Pominiętych (pusty context_text): {skipped_empty}")
    return valid


def load_completed() -> set[tuple[str, str]]:
    """(config, question_id) pary które są już ocenione w ragas_results.csv."""
    if not OUTPUT_PATH.exists():
        return set()
    completed = set()
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                # Sprawdź czy obie metryki są poprawnymi liczbami
                float(row.get("llm_context_precision_without_reference", "nan"))
                float(row.get("nv_context_relevance", "nan"))
            except ValueError:
                continue
            completed.add((row["config"], row["question_id"]))
    return completed


def evaluate_one_at_a_time(rows: list[dict], ragas_llm, ragas_embeddings) -> list[dict]:
    """Ewaluacja per wiersz z idempotencją.
    Powolniejsze niż batch, ale nie traci postępu jak coś padnie."""
    
    completed = load_completed()
    if completed:
        logger.info(f"Pomijam {len(completed)} już ocenionych wierszy")
    
    pending = [r for r in rows if (r["config"], r["question_id"]) not in completed]
    logger.info(f"Do oceny: {len(pending)} wierszy")
    
    if not pending:
        logger.info("Wszystko już ocenione.")
        return []
    
    metrics = [
        LLMContextPrecisionWithoutReference(llm=ragas_llm),
        ContextRelevance(llm=ragas_llm),
    ]
    
    file_exists = OUTPUT_PATH.exists() and OUTPUT_PATH.stat().st_size > 0
    fieldnames = [
        "config", "question_id", "category", "question",
        "n_contexts",
        "llm_context_precision_without_reference",
        "nv_context_relevance",
    ]
    
    mode = "a" if file_exists else "w"
    results = []
    
    with open(OUTPUT_PATH, mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        
        for i, row in enumerate(pending, 1):
            chunks = parse_context_to_chunks(row["context_text"])
            
            sample = {
                "user_input": row["question"],
                "retrieved_contexts": chunks,
                "response": row["answer"],
            }
            
            logger.info(f"[{i}/{len(pending)}] {row['config']} | {row['question_id']} | n_chunks={len(chunks)}")
            
            try:
                ds = Dataset.from_list([sample])
                eval_result = evaluate(
                    dataset=ds,
                    metrics=metrics,
                    llm=ragas_llm,
                    embeddings=ragas_embeddings,
                    show_progress=False,
                    raise_exceptions=False,
                )
                df = eval_result.to_pandas()
                
                # Wyciągnij wartości metryk
                precision = df["llm_context_precision_without_reference"].iloc[0] if "llm_context_precision_without_reference" in df.columns else None
                relevance = df["nv_context_relevance"].iloc[0] if "nv_context_relevance" in df.columns else None
                
                writer.writerow({
                    "config": row["config"],
                    "question_id": row["question_id"],
                    "category": row.get("category", ""),
                    "question": row["question"],
                    "n_contexts": len(chunks),
                    "llm_context_precision_without_reference": precision,
                    "nv_context_relevance": relevance,
                })
                f.flush()
                
                p_str = f"{precision:.3f}" if precision is not None else "nan"
                r_str = f"{relevance:.3f}" if relevance is not None else "nan"
                logger.info(f"    precision={p_str}  relevance={r_str}")
                
                results.append({
                    "config": row["config"],
                    "question_id": row["question_id"],
                    "precision": precision,
                    "relevance": relevance,
                })
            
            except Exception as e:
                logger.error(f"    BŁĄD: {type(e).__name__}: {str(e)[:150]}")
                # Nie zapisujemy do CSV — chcemy retry przy następnym uruchomieniu
            
            time.sleep(0.3)
    
    return results


def print_summary():
    if not OUTPUT_PATH.exists():
        return
    
    sums = defaultdict(lambda: {"precision": [], "relevance": []})
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cfg = row["config"]
            try:
                p = float(row["llm_context_precision_without_reference"])
                r = float(row["nv_context_relevance"])
            except ValueError:
                continue
            if p == p:  # not NaN
                sums[cfg]["precision"].append(p)
            if r == r:
                sums[cfg]["relevance"].append(r)
    
    if not sums:
        return
    
    ORDER = ["baseline", "semantic_chunk", "u_shape", "rewrite",
             "extractive", "hierarchical", "full_stack", "brief_context"]
    
    print(f"\n{'='*72}")
    print(f"{'Config':<18} {'Precision':>12} {'Relevance':>12}  N")
    print('-' * 72)
    for cfg in ORDER:
        if cfg not in sums:
            continue
        s = sums[cfg]
        n_p = len(s["precision"])
        n_r = len(s["relevance"])
        p_avg = sum(s["precision"]) / n_p if n_p else 0
        r_avg = sum(s["relevance"]) / n_r if n_r else 0
        n = max(n_p, n_r)
        print(f"{cfg:<18} {p_avg:>12.3f} {r_avg:>12.3f}  {n}")
    print('=' * 72)


def main():
    rows = load_rows()
    if not rows:
        logger.error("Brak wierszy z context_text — czy CSV ma kolumnę?")
        return
    
    logger.info("Inicjalizacja LLM (Gemini) i embeddings dla ragas")
    judge_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=settings.google_api_key,
        temperature=0.0,
    )
    embeddings = build_embeddings()
    
    ragas_llm = LangchainLLMWrapper(judge_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
    
    evaluate_one_at_a_time(rows, ragas_llm, ragas_embeddings)
    
    logger.info(f"\nGotowe. Wyniki: {OUTPUT_PATH}")
    print_summary()


if __name__ == "__main__":
    main()