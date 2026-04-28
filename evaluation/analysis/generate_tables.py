import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # evaluation/
EXPERIMENT_PATH = ROOT / "outputs" / "experiment_results.csv"
RAGAS_PATH = ROOT / "outputs" / "ragas_results.csv"
LITM_PATH = ROOT / "outputs" / "lost_in_middle_results.csv"

CONFIG_ORDER = [
    "baseline", "semantic_chunk", "u_shape", "rewrite",
    "extractive", "hierarchical", "full_stack", "brief_context",
]


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def safe_int(v, default=0):
    try:
        return int(v)
    except (ValueError, TypeError):
        return default


def safe_float(v, default=None):
    try:
        x = float(v)
        if x != x:  # NaN check
            return default
        return x
    except (ValueError, TypeError):
        return default


def table_main_experiment():
    print("=" * 100)
    print("TABELA 1: Macierz konfiguracji (experiment_results.csv)")
    print("=" * 100)

    if not EXPERIMENT_PATH.exists():
        print(f"BŁĄD: brak pliku {EXPERIMENT_PATH}")
        return

    with open(EXPERIMENT_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    sums = defaultdict(lambda: {"f": [], "r": [], "c": [], "lat": [], "ctx": [], "ans": []})
    for r in rows:
        f_val = safe_int(r.get("faithfulness"))
        if f_val == 0 or r.get("answer", "").startswith("ERROR"):
            continue
        cfg = r["config"]
        sums[cfg]["f"].append(f_val)
        sums[cfg]["r"].append(safe_int(r["relevance"]))
        sums[cfg]["c"].append(safe_int(r["completeness"]))
        sums[cfg]["lat"].append(safe_float(r["latency_s"], 0.0))
        sums[cfg]["ctx"].append(safe_int(r["context_tokens"]))
        sums[cfg]["ans"].append(safe_int(r["answer_tokens"]))

    print("\n| Konfiguracja      | Faithfulness | Relevance | Completeness | Latencja [s] | Kontekst [tok.] | Odpowiedź [tok.] |")
    print("|-------------------|--------------|-----------|--------------|--------------|-----------------|------------------|")
    for cfg in CONFIG_ORDER:
        if cfg not in sums:
            continue
        s = sums[cfg]
        n = len(s["f"])
        if n == 0:
            continue
        print(
            f"| {cfg:<17} | "
            f"{mean(s['f']):>12.2f} | "
            f"{mean(s['r']):>9.2f} | "
            f"{mean(s['c']):>12.2f} | "
            f"{mean(s['lat']):>12.2f} | "
            f"{mean(s['ctx']):>15.0f} | "
            f"{mean(s['ans']):>16.0f} |"
        )

    # Plain format (czytelniejszy w terminalu)
    print(f"\n{'Konfiguracja':<18} {'F':>6} {'R':>6} {'C':>6} {'Lat':>8} {'Ctx':>8} {'Ans':>6}  N")
    print("-" * 70)
    for cfg in CONFIG_ORDER:
        if cfg not in sums:
            continue
        s = sums[cfg]
        n = len(s["f"])
        if n == 0:
            continue
        print(
            f"{cfg:<18} "
            f"{mean(s['f']):>6.2f} {mean(s['r']):>6.2f} {mean(s['c']):>6.2f} "
            f"{mean(s['lat']):>8.2f} {mean(s['ctx']):>8.0f} {mean(s['ans']):>6.0f}  {n}"
        )


def table_ragas():
    print("\n" + "=" * 100)
    print("TABELA 2: Ragas (ragas_results.csv)")
    print("=" * 100)

    if not RAGAS_PATH.exists():
        print(f"BŁĄD: brak pliku {RAGAS_PATH}")
        return

    with open(RAGAS_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    sums = defaultdict(lambda: {"p": [], "r": []})
    for row in rows:
        cfg = row["config"]
        p = safe_float(row.get("llm_context_precision_without_reference"))
        if p is not None:
            sums[cfg]["p"].append(p)
        rel = safe_float(row.get("nv_context_relevance"))
        if rel is not None:
            sums[cfg]["r"].append(rel)

    print("\n| Konfiguracja      | Context Precision | Context Relevance | N  |")
    print("|-------------------|-------------------|-------------------|----|")
    for cfg in CONFIG_ORDER:
        if cfg not in sums:
            continue
        s = sums[cfg]
        n = max(len(s["p"]), len(s["r"]))
        if n == 0:
            continue
        p_avg = mean(s["p"])
        r_avg = mean(s["r"])
        print(f"| {cfg:<17} | {p_avg:>17.3f} | {r_avg:>17.3f} | {n:>2} |")

    print(f"\n{'Konfiguracja':<18} {'Precision':>10} {'Relevance':>10}  N")
    print("-" * 50)
    for cfg in CONFIG_ORDER:
        if cfg not in sums:
            continue
        s = sums[cfg]
        n = max(len(s["p"]), len(s["r"]))
        if n == 0:
            continue
        print(f"{cfg:<18} {mean(s['p']):>10.3f} {mean(s['r']):>10.3f}  {n}")


def table_litm():
    print("\n" + "=" * 100)
    print("TABELA 3: Lost-in-the-middle — completeness (lost_in_middle_results.csv)")
    print("=" * 100)

    if not LITM_PATH.exists():
        print(f"BŁĄD: brak pliku {LITM_PATH}")
        return

    with open(LITM_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    cells_c = defaultdict(list)
    cells_f = defaultdict(list)
    for r in rows:
        c = safe_int(r["completeness"])
        if c == 0:
            continue  # pomijamy błędne
        f_val = safe_int(r["faithfulness"])
        mit = r["mitigation"]
        pos = safe_int(r["key_position"])
        cells_c[(mit, pos)].append(c)
        if f_val > 0:
            cells_f[(mit, pos)].append(f_val)

    if not cells_c:
        print("Brak danych do wyświetlenia.")
        return

    mitigations_present = sorted({k[0] for k in cells_c.keys()})
    positions = sorted({k[1] for k in cells_c.keys()})

    pos_label = {p: p + 1 for p in positions}

    MIT_ORDER = ["no_mitigation", "u_shape", "brief_context"]
    MIT_LABELS = {"no_mitigation": "brak", "u_shape": "u_shape", "brief_context": "brief_context"}
    mitigations = [m for m in MIT_ORDER if m in mitigations_present]

    print("\n### Completeness")
    header = "| Mitigation     | " + " | ".join(f"pos={pos_label[p]:<3}" for p in positions) + " | średnia |"
    sep = "|----------------|" + "|".join(["-------"] * len(positions)) + "|---------|"
    print(header)
    print(sep)
    for mit in mitigations:
        row = f"| {MIT_LABELS[mit]:<14} | "
        all_vals = []
        cells_str = []
        for p in positions:
            vals = cells_c.get((mit, p), [])
            if vals:
                avg = mean(vals)
                cells_str.append(f"{avg:>5.2f} ")
                all_vals.extend(vals)
            else:
                cells_str.append(f"{'—':>6}")
        row += " | ".join(cells_str)
        overall = mean(all_vals)
        row += f" | {overall:>7.2f} |"
        print(row)

    # Plain
    print(f"\n{'Mitigation':<16}", end="")
    for p in positions:
        print(f"{f'pos={pos_label[p]}':>10}", end="")
    print(f"{'avg':>10}")
    print("-" * (16 + 10 * (len(positions) + 1)))
    for mit in mitigations:
        print(f"{MIT_LABELS[mit]:<16}", end="")
        all_vals = []
        for p in positions:
            vals = cells_c.get((mit, p), [])
            if vals:
                print(f"{mean(vals):>10.2f}", end="")
                all_vals.extend(vals)
            else:
                print(f"{'—':>10}", end="")
        print(f"{mean(all_vals):>10.2f}")

    print("\n### Faithfulness")
    print(f"{'Mitigation':<16}", end="")
    for p in positions:
        print(f"{f'pos={pos_label[p]}':>10}", end="")
    print(f"{'avg':>10}")
    print("-" * (16 + 10 * (len(positions) + 1)))
    for mit in mitigations:
        print(f"{MIT_LABELS[mit]:<16}", end="")
        all_vals = []
        for p in positions:
            vals = cells_f.get((mit, p), [])
            if vals:
                print(f"{mean(vals):>10.2f}", end="")
                all_vals.extend(vals)
            else:
                print(f"{'—':>10}", end="")
        print(f"{mean(all_vals):>10.2f}")


def table_litm_per_question():
    print("\n" + "=" * 100)
    print("TABELA 4 (dodatkowa): LITM per pytanie — completeness")
    print("=" * 100)

    if not LITM_PATH.exists():
        return

    with open(LITM_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_cell = {}
    for r in rows:
        c = safe_int(r["completeness"])
        if c == 0:
            continue
        by_cell[(r["question_id"], r["mitigation"], safe_int(r["key_position"]))] = c

    questions = sorted({k[0] for k in by_cell.keys()})
    mitigations = ["no_mitigation", "u_shape", "brief_context"]
    positions = sorted({k[2] for k in by_cell.keys()})

    print(f"\n{'qid':<5} {'mitigation':<16}", end="")
    for p in positions:
        print(f"{f'pos={p+1}':>8}", end="")
    print()
    print("-" * 70)

    for q in questions:
        for m in mitigations:
            print(f"{q:<5} {m:<16}", end="")
            for p in positions:
                v = by_cell.get((q, m, p))
                print(f"{(str(v) if v else '—'):>8}", end="")
            print()
        print()


if __name__ == "__main__":
    table_main_experiment()
    table_ragas()
    table_litm()
    table_litm_per_question()