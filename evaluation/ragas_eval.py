import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import logging
from collections import defaultdict
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
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
    if not context_text or not context_text.strip():
        return []
    chunks = [c.strip() for c in context_text.split("\n\n") if c.strip()]
    return chunks


def load_rows() -> list[dict]:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Brak pliku: {INPUT_PATH}")
    with open(INPUT_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    valid = [
        r for r in rows
        if int(r.get("faithfulness", 0)) > 0
        and not r.get("answer", "").startswith("ERROR")
        and r.get("context_tokens", "0") != "0" 
    ]
    logger.info(f"Wczytano {len(rows)} wierszy ({len(valid)} validnych z niepustym kontekstem)")
    return valid


def evaluate_ragas(rows: list[dict]):
    
    judge_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=settings.google_api_key,
        temperature=0.0,
    )
    embeddings = build_embeddings()
    
    ragas_llm = LangchainLLMWrapper(judge_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
    
    data = {
        "user_input": [],
        "retrieved_contexts": [],
        "response": [],
        "_config": [],
        "_question_id": [],
    }
    
    for row in rows:
        chunks = parse_context_to_chunks(row["context_text"] if "context_text" in row else "")
        if not chunks:
            continue
        
        data["user_input"].append(row["question"])
        data["retrieved_contexts"].append(chunks)
        data["response"].append(row["answer"])
        data["_config"].append(row["config"])
        data["_question_id"].append(row["question_id"])
    
    if not data["user_input"]:
        logger.error("Brak wierszy do oceny - czy CSV zawiera kolumnę 'context_text'?")
        return None
    
    logger.info(f"Ewaluacja ragas | {len(data['user_input'])} przykładów")
    
    ragas_data = {k: v for k, v in data.items() if not k.startswith("_")}
    dataset = Dataset.from_dict(ragas_data)
    
    metrics = [
        LLMContextPrecisionWithoutReference(),
        ContextRelevance(),
    ]
    
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        show_progress=True,
    )
    
    df = result.to_pandas()
    df["config"] = data["_config"]
    df["question_id"] = data["_question_id"]
    
    return df


def save_and_summarize(df):
    if df is None:
        return
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    logger.info(f"Wyniki zapisane: {OUTPUT_PATH}")
    
    metric_cols = [c for c in df.columns if c not in ("user_input", "retrieved_contexts",
                                                        "response", "config", "question_id")]
    print(f"\n{'='*70}")
    print(f"{'Config':<18} | " + " | ".join(f"{m[:18]:>18}" for m in metric_cols))
    print('-' * 70)
    for cfg, group in df.groupby("config"):
        means = [group[m].mean() for m in metric_cols]
        print(f"{cfg:<18} | " + " | ".join(f"{v:>18.3f}" for v in means))
    print('=' * 70)


if __name__ == "__main__":
    rows = load_rows()
    df = evaluate_ragas(rows)
    save_and_summarize(df)