import csv
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "outputs" / "experiment_results.csv"
LITM_PATH = ROOT / "outputs" / "lost_in_middle_results.csv"
RAGAS_PATH = ROOT / "outputs" / "ragas_results.csv"
PLOTS_DIR = ROOT / "analysis" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_ORDER = [
    "baseline",
    "semantic_chunk",
    "u_shape",
    "rewrite",
    "extractive",
    "hierarchical",
    "full_stack",
    "brief_context",
]

CONFIG_LABELS = {
    "baseline": "Baseline\n(structure)",
    "semantic_chunk": "Semantic\nchunking",
    "u_shape": "U-shape\nreranking",
    "rewrite": "Query\nrewriting",
    "extractive": "Extractive\ncompression",
    "hierarchical": "Hierarchical\ncompression",
    "full_stack": "Full stack",
    "brief_context": "Brief\nContext",
}

COLORS = {
    "faithfulness": "#2E7D32",
    "relevance":    "#1565C0",
    "completeness": "#E65100",
    "ctx_tokens":   "#6A1B9A",
    "latency":      "#C62828",
    "precision":    "#00695C",
    "ragas_relevance": "#283593",
}

MITIGATION_ORDER = ["no_mitigation", "u_shape", "brief_context"]
MITIGATION_LABELS = {
    "no_mitigation": "Brak mitygacji",
    "u_shape": "U-shape",
    "brief_context": "BriefContext",
}
MITIGATION_COLORS = {
    "no_mitigation": "#C62828",
    "u_shape": "#1565C0",
    "brief_context": "#2E7D32",
}


def load_data(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    valid = [
        r for r in rows
        if int(r.get("faithfulness", 0)) > 0
        and not r.get("answer", "").startswith("ERROR")
    ]
    logger.info(f"Loaded {len(rows)} rows ({len(valid)} valid) from {path.name}")
    return valid


def aggregate_by_config(rows: list[dict]) -> dict[str, dict]:
    sums = defaultdict(lambda: {"f": [], "r": [], "c": [], "lat": [], "ctx": [], "ans": []})
    for r in rows:
        cfg = r["config"]
        sums[cfg]["f"].append(int(r["faithfulness"]))
        sums[cfg]["r"].append(int(r["relevance"]))
        sums[cfg]["c"].append(int(r["completeness"]))
        sums[cfg]["lat"].append(float(r["latency_s"]))
        sums[cfg]["ctx"].append(int(r["context_tokens"]))
        sums[cfg]["ans"].append(int(r["answer_tokens"]))
    return dict(sums)


def configs_present(agg: dict) -> list[str]:
    return [c for c in CONFIG_ORDER if c in agg]


def mean(xs: list) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def safe_float(v, default=None):
    try:
        x = float(v)
        if x != x:
            return default
        return x
    except (ValueError, TypeError):
        return default


def save(fig, name: str) -> None:
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path.relative_to(ROOT.parent)}")

def plot_metrics_by_config(agg: dict) -> None:
    cfgs = configs_present(agg)
    f_means = [mean(agg[c]["f"]) for c in cfgs]
    r_means = [mean(agg[c]["r"]) for c in cfgs]
    c_means = [mean(agg[c]["c"]) for c in cfgs]

    x = np.arange(len(cfgs))
    width = 0.27

    fig, ax = plt.subplots(figsize=(12, 5.5))
    b1 = ax.bar(x - width, f_means, width, label="Faithfulness",
                color=COLORS["faithfulness"], edgecolor="white")
    b2 = ax.bar(x,         r_means, width, label="Relevance",
                color=COLORS["relevance"], edgecolor="white")
    b3 = ax.bar(x + width, c_means, width, label="Completeness",
                color=COLORS["completeness"], edgecolor="white")

    ax.set_ylabel("Średnia ocena (skala 1–5)")
    ax.set_title("Jakość odpowiedzi wg konfiguracji")
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS[c] for c in cfgs])
    ax.set_ylim(0, 5.4)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    save(fig, "01_metrics_by_config.png")


def plot_completeness_vs_context(agg: dict) -> None:
    cfgs = configs_present(agg)
    fig, ax = plt.subplots(figsize=(9, 6))

    for cfg in cfgs:
        ctx = mean(agg[cfg]["ctx"])
        comp = mean(agg[cfg]["c"])
        ax.scatter(ctx, comp, s=180, alpha=0.85, edgecolor="white", linewidth=1.5)
        ax.annotate(
            CONFIG_LABELS[cfg].replace("\n", " "),
            xy=(ctx, comp),
            xytext=(8, 8), textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel("Średnia liczba tokenów kontekstu (przekazanych do LLM)")
    ax.set_ylabel("Średnia ocena completeness (1-5)")
    ax.set_title("Trade-off: jakość odpowiedzi vs koszt tokenów")
    ax.set_ylim(0, 5.4)
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    save(fig, "02_completeness_vs_context.png")


def plot_latency(agg: dict) -> None:
    cfgs = configs_present(agg)
    lat_means = [mean(agg[c]["lat"]) for c in cfgs]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(
        [CONFIG_LABELS[c] for c in cfgs], lat_means,
        color=COLORS["latency"], edgecolor="white"
    )
    ax.set_ylabel("Średni czas odpowiedzi (s)")
    ax.set_title("Latencja per konfiguracja (bez czasu LLM-as-a-Judge)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    for bar, val in zip(bars, lat_means):
        ax.annotate(f"{val:.2f}s", xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    save(fig, "03_latency_by_config.png")


def plot_token_usage(agg: dict) -> None:
    cfgs = configs_present(agg)
    ctx_means = [mean(agg[c]["ctx"]) for c in cfgs]
    ans_means = [mean(agg[c]["ans"]) for c in cfgs]

    x = np.arange(len(cfgs))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    b1 = ax.bar(x, ctx_means, color=COLORS["ctx_tokens"],
                edgecolor="white", label="Kontekst (input)")
    b2 = ax.bar(x, ans_means, bottom=ctx_means, color="#FFB300",
                edgecolor="white", label="Odpowiedź (output)")

    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS[c] for c in cfgs])
    ax.set_ylabel("Średnia liczba tokenów")
    ax.set_title("Zużycie tokenów per konfiguracja (input + output)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    for i, (c_val, a_val) in enumerate(zip(ctx_means, ans_means)):
        ax.annotate(f"{c_val + a_val:.0f}", xy=(i, c_val + a_val),
                    xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    save(fig, "04_token_usage.png")


def plot_completeness_per_question(rows: list[dict], agg: dict) -> None:
    cfgs = configs_present(agg)
    qids = sorted({r["question_id"] for r in rows},
                  key=lambda q: (q[0], int(q[1:])))

    matrix = np.full((len(cfgs), len(qids)), np.nan)
    for r in rows:
        cfg = r["config"]
        qid = r["question_id"]
        if cfg in cfgs and qid in qids:
            i = cfgs.index(cfg)
            j = qids.index(qid)
            matrix[i, j] = int(r["completeness"])

    fig, ax = plt.subplots(figsize=(max(8, len(qids) * 0.7), 0.7 * len(cfgs) + 1.5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=1, vmax=5, aspect="auto")

    ax.set_xticks(np.arange(len(qids)))
    ax.set_xticklabels(qids)
    ax.set_yticks(np.arange(len(cfgs)))
    ax.set_yticklabels([CONFIG_LABELS[c].replace("\n", " ") for c in cfgs])
    ax.set_title("Completeness per pytanie i konfiguracja")
    ax.set_xlabel("Pytanie testowe (S=krótkie, L=długie, N=negatywne)")

    for i in range(len(cfgs)):
        for j in range(len(qids)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{int(v)}", ha="center", va="center",
                        color="black", fontsize=10, fontweight="bold")

    fig.colorbar(im, ax=ax, label="Completeness (1–5)", shrink=0.8)
    save(fig, "05_completeness_heatmap.png")


def plot_metrics_by_category(rows: list[dict], agg: dict) -> None:
    cfgs = configs_present(agg)
    categories = ["short", "long", "negative"]
    cat_labels = {"short": "Krótkie", "long": "Długie", "negative": "Negatywne"}

    by_cat = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_cat[r["config"]][r["category"]].append(int(r["completeness"]))

    x = np.arange(len(cfgs))
    width = 0.27
    cat_colors = {"short": "#43A047", "long": "#1E88E5", "negative": "#FB8C00"}

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, cat in enumerate(categories):
        means = [mean(by_cat[cfg].get(cat, [])) for cfg in cfgs]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, means, width, label=cat_labels[cat],
                      color=cat_colors[cat], edgecolor="white")
        for bar, val in zip(bars, means):
            if val > 0:
                ax.annotate(f"{val:.1f}", xy=(bar.get_x() + bar.get_width()/2, val),
                            xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS[c] for c in cfgs])
    ax.set_ylabel("Średnia completeness (1–5)")
    ax.set_title("Completeness wg kategorii pytań")
    ax.set_ylim(0, 5.4)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    save(fig, "06_completeness_by_category.png")


def plot_summary_table(agg: dict) -> None:
    cfgs = configs_present(agg)
    headers = ["Config", "F", "R", "C", "Lat[s]", "CtxTok", "AnsTok", "N"]
    data = []
    for cfg in cfgs:
        s = agg[cfg]
        n = len(s["f"])
        data.append([
            CONFIG_LABELS[cfg].replace("\n", " "),
            f"{mean(s['f']):.2f}",
            f"{mean(s['r']):.2f}",
            f"{mean(s['c']):.2f}",
            f"{mean(s['lat']):.2f}",
            f"{mean(s['ctx']):.0f}",
            f"{mean(s['ans']):.0f}",
            str(n),
        ])

    fig, ax = plt.subplots(figsize=(11, 0.55 * len(data) + 1.2))
    ax.axis("off")
    tbl = ax.table(cellText=data, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)

    tbl.auto_set_column_width(list(range(len(headers))))

    for j in range(len(headers)):
        tbl[(0, j)].set_facecolor("#1a2038")
        tbl[(0, j)].set_text_props(color="white", weight="bold")

    ax.set_title("Podsumowanie wyników eksperymentu", pad=14, fontsize=13, weight="bold")
    save(fig, "07_summary_table.png")


def plot_litm_curves() -> None:
    if not LITM_PATH.exists():
        logger.warning(f"Brak {LITM_PATH} — pomijam wykres LITM curves")
        return

    with open(LITM_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    cells = defaultdict(list)
    for r in rows:
        try:
            c = int(r["completeness"])
        except ValueError:
            continue
        if c == 0:
            continue
        mit = r["mitigation"]
        pos = int(r["key_position"])
        cells[(mit, pos)].append(c)

    if not cells:
        logger.warning("Brak danych LITM")
        return

    positions = sorted({k[1] for k in cells.keys()})
    pos_labels = [p + 1 for p in positions]  # 0-indexed → 1-indexed dla raportu

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for mit in MITIGATION_ORDER:
        if not any((mit, p) in cells for p in positions):
            continue
        means = []
        for p in positions:
            vals = cells.get((mit, p), [])
            means.append(mean(vals) if vals else np.nan)
        ax.plot(pos_labels, means, marker="o", markersize=10, linewidth=2.5,
                label=MITIGATION_LABELS[mit], color=MITIGATION_COLORS[mit])

        for x, y in zip(pos_labels, means):
            if not np.isnan(y):
                ax.annotate(f"{y:.2f}", xy=(x, y),
                            xytext=(0, 8), textcoords="offset points",
                            ha="center", fontsize=8, color=MITIGATION_COLORS[mit])

    ax.set_xlabel("Pozycja kluczowego fragmentu w kontekście (1 = początek, 10 = koniec)")
    ax.set_ylabel("Średnia completeness (1–5)")
    ax.set_title("Lost-in-the-middle: completeness vs pozycja klucza")
    ax.set_ylim(0.5, 5.4)
    ax.set_xticks(pos_labels)
    ax.legend(loc="lower right", title="Metoda mitygacji")
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    save(fig, "08_litm_curves.png")


def plot_litm_per_question() -> None:
    """Heatmap LITM: pytanie × pozycja, jeden subplot per metoda mitygacji."""
    if not LITM_PATH.exists():
        logger.warning(f"Brak {LITM_PATH} — pomijam wykres LITM per question")
        return

    with open(LITM_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    valid_rows = [r for r in rows if int(r.get("completeness", 0)) > 0]
    if not valid_rows:
        logger.warning("Brak danych LITM")
        return

    questions = sorted({r["question_id"] for r in valid_rows},
                       key=lambda q: (q[0], int(q[1:])))
    positions = sorted({int(r["key_position"]) for r in valid_rows})
    pos_labels = [p + 1 for p in positions]
    mitigations = [m for m in MITIGATION_ORDER if any(r["mitigation"] == m for r in valid_rows)]

    fig, axes = plt.subplots(1, len(mitigations),
                              figsize=(4 * len(mitigations), 0.6 * len(questions) + 2),
                              sharey=True)
    if len(mitigations) == 1:
        axes = [axes]

    for ax, mit in zip(axes, mitigations):
        matrix = np.full((len(questions), len(positions)), np.nan)
        for r in valid_rows:
            if r["mitigation"] != mit:
                continue
            qi = questions.index(r["question_id"])
            pi = positions.index(int(r["key_position"]))
            matrix[qi, pi] = int(r["completeness"])

        im = ax.imshow(matrix, cmap="RdYlGn", vmin=1, vmax=5, aspect="auto")
        ax.set_title(MITIGATION_LABELS[mit], fontsize=11)
        ax.set_xticks(np.arange(len(positions)))
        ax.set_xticklabels(pos_labels)
        ax.set_xlabel("Pozycja klucza")
        if ax is axes[0]:
            ax.set_yticks(np.arange(len(questions)))
            ax.set_yticklabels(questions)
            ax.set_ylabel("Pytanie")

        for i in range(len(questions)):
            for j in range(len(positions)):
                v = matrix[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{int(v)}", ha="center", va="center",
                            color="black", fontsize=10, fontweight="bold")

    fig.suptitle("Lost-in-the-middle: completeness per pytanie × pozycja × metoda", y=1.02)
    fig.colorbar(im, ax=axes, label="Completeness (1–5)", shrink=0.7)
    save(fig, "09_litm_per_question_heatmap.png")


def plot_ragas_metrics() -> None:
    """Bar chart: ragas precision / relevance per config."""
    if not RAGAS_PATH.exists():
        logger.warning(f"Brak {RAGAS_PATH} — pomijam wykres ragas")
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

    cfgs = [c for c in CONFIG_ORDER if c in sums]
    if not cfgs:
        logger.warning("Brak danych ragas")
        return

    p_means = [mean(sums[c]["p"]) for c in cfgs]
    r_means = [mean(sums[c]["r"]) for c in cfgs]

    x = np.arange(len(cfgs))
    width = 0.4

    fig, ax = plt.subplots(figsize=(12, 5.5))
    b1 = ax.bar(x - width/2, p_means, width, label="Context Precision",
                color=COLORS["precision"], edgecolor="white")
    b2 = ax.bar(x + width/2, r_means, width, label="Context Relevance",
                color=COLORS["ragas_relevance"], edgecolor="white")

    ax.set_ylabel("Średnia wartość metryki (0-1)")
    ax.set_title("Ragas: jakość retrievala wg konfiguracji")
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS[c] for c in cfgs])
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    # fig.text(0.5, -0.02,
    #          "Uwaga: precision = 1.00 dla configów z kompresją (hierarchical, brief_context, full_stack) "
    #          "to artefakt - Ragas dostaje już zsyntetyzowany pojedynczy chunk, a nie oryginalne fragmenty.",
    #          ha="center", fontsize=9, style="italic",
    #          bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF9C4", edgecolor="#F9A825", alpha=0.9))

    save(fig, "10_ragas_metrics.png")

def main():
    rows = load_data(INPUT_PATH)
    if not rows:
        logger.error("Brak prawidłowych wierszy w experiment_results.csv")
        return
    agg = aggregate_by_config(rows)

    plot_metrics_by_config(agg)
    plot_completeness_vs_context(agg)
    plot_latency(agg)
    plot_token_usage(agg)
    plot_completeness_per_question(rows, agg)
    plot_metrics_by_category(rows, agg)
    plot_summary_table(agg)

    # Faza 2 — LITM i Ragas
    plot_litm_curves()
    plot_litm_per_question()
    plot_ragas_metrics()

    logger.info(f"\nWszystkie wykresy zapisane do: {PLOTS_DIR}")


if __name__ == "__main__":
    main()