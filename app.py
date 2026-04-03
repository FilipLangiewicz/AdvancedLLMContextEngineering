import streamlit as st
from config.settings import ChunkingStrategy, LLMProvider
from compression.base import CompressionStrategy
from query.base import QueryUnderstandingStrategy
from reranking.base import RerankingStrategy
from generation.embeddings_factory import build_embeddings
from pipeline import build_pipeline
from pathlib import Path


def load_css(path: Path) -> None:
    with open(path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(
    page_title="Doradca Energetyczny",
    page_icon="⚖",
    layout="wide",
)

load_css(Path(__file__).parent / "style.css")


@st.cache_resource(show_spinner="Ładowanie modelu…")
def get_embeddings():
    return build_embeddings()


with st.sidebar:
    st.markdown("### Konfiguracja systemu")

    st.markdown('<span class="sidebar-section">Segmentacja dokumentów</span>', unsafe_allow_html=True)
    chunking = st.radio(
        "Strategia segmentacji",
        options=[ChunkingStrategy.STRUCTURE, ChunkingStrategy.SEMANTIC],
        format_func=lambda x: {
            ChunkingStrategy.STRUCTURE: "Strukturalna",
            ChunkingStrategy.SEMANTIC: "Semantyczna",
        }[x],
        label_visibility="collapsed",
    )

    st.markdown('<span class="sidebar-section">Przetwarzanie zapytań</span>', unsafe_allow_html=True)
    query_label = st.radio(
        "Transformacja zapytania",
        options=["rewriting", "none"],
        format_func=lambda x: "Przepisywanie zapytania" if x == "rewriting" else "Bez transformacji",
        label_visibility="collapsed",
    )
    query_understanding = (
        QueryUnderstandingStrategy.REWRITING if query_label == "rewriting" else None
    )

    st.markdown('<span class="sidebar-section">Reranking</span>', unsafe_allow_html=True)
    reranking = st.radio(
        "Strategia rerankingu",
        options=[RerankingStrategy.ORIGINAL_ORDER, RerankingStrategy.U_SHAPE_REORDER],
        format_func=lambda x: {
            RerankingStrategy.ORIGINAL_ORDER: "Kolejność oryginalna",
            RerankingStrategy.U_SHAPE_REORDER: "Zmiana kolejności U-Shape",
        }[x],
        label_visibility="collapsed",
    )

    st.markdown('<span class="sidebar-section">Kompresja kontekstu</span>', unsafe_allow_html=True)
    compression_label = st.radio(
        "Strategia kompresji",
        options=["none", "extractive", "hierarchical"],
        format_func=lambda x: {
            "none": "Bez kompresji",
            "extractive": "Filtr ekstrakcyjny",
            "hierarchical": "Streszczenie hierarchiczne",
        }[x],
        label_visibility="collapsed",
    )
    compression = {
        "none": None,
        "extractive": CompressionStrategy.EXTRACTIVE_FILTER,
        "hierarchical": CompressionStrategy.HIERARCHICAL_SUMMARY,
    }[compression_label]

    st.markdown('<span class="sidebar-section">Model językowy</span>', unsafe_allow_html=True)
    provider = st.selectbox(
        "Dostawca",
        options=[LLMProvider.GROQ, LLMProvider.OPENAI, LLMProvider.GOOGLE, LLMProvider.ANTHROPIC],
        format_func=lambda x: x.value.capitalize(),
        label_visibility="collapsed",
    )

    st.markdown('<span class="sidebar-section">Wyszukiwanie</span>', unsafe_allow_html=True)
    retrieval_k = st.slider("Liczba fragmentów (k)", min_value=1, max_value=10, value=5)

    st.markdown('<span class="sidebar-section">Pamięć podręczna</span>', unsafe_allow_html=True)
    with_cache = st.toggle("Włącz pamięć podręczną", value=True)
    cache_threshold = st.slider(
        "Próg podobieństwa",
        min_value=0.70,
        max_value=1.00,
        value=0.92,
        step=0.01,
        disabled=not with_cache,
    )

    st.divider()
    if st.button("Wyczyść historię rozmowy", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Rebuild pipeline only when settings change
pipeline_key = (chunking, query_understanding, compression, reranking, provider, retrieval_k, with_cache, cache_threshold)
if "pipeline_key" not in st.session_state or st.session_state.pipeline_key != pipeline_key:
    with st.spinner("Rozgrzewanie aplikacji…"):
        embeddings = get_embeddings()
        st.session_state.pipeline = build_pipeline(
            embeddings=embeddings,
            chunking_strategy=chunking,
            query_understanding_strategy=query_understanding,
            compression_strategy=compression,
            reranking_strategy=reranking,
            llm_provider=provider,
            retrieval_k=retrieval_k,
            cache_threshold=cache_threshold,
            with_cache=with_cache,
        )
    st.session_state.pipeline_key = pipeline_key


# Header
st.markdown("""
<div class="legal-header">
    <span class="legal-header-title">Doradca Energetyczny</span>
    <span class="legal-header-subtitle">Analiza przepisów z zakresu energetyki</span>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "meta" in msg:
            meta = msg["meta"]
            cached = "[CACHED]" in meta.get("strategy", "")
            label = "Szczegóły odpowiedzi" + (" — odpowiedź z pamięci podręcznej" if cached else "")
            with st.expander(label):
                st.caption(f"**Konfiguracja:** `{meta['strategy']}`")
                if meta.get("sources"):
                    st.caption("**Źródła:**")
                    for src in meta["sources"]:
                        st.caption(
                            f"- `{src['source']}` | strona {src['page']} "
                            f"| {src.get('article_id', '')} | {src.get('chapter', '')[:60]}…"
                        )

if query := st.chat_input("Zadaj pytanie dotyczące dokumentów prawnych…"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Analizowanie dokumentów…"):
            response = st.session_state.pipeline.run(query)

        st.markdown(response.answer)

        cached = "[CACHED]" in response.strategy
        label = "Szczegóły odpowiedzi" + (" — odpowiedź z pamięci podręcznej" if cached else "")
        with st.expander(label):
            st.caption(f"**Konfiguracja:** `{response.strategy}`")
            if response.sources:
                st.caption("**Źródła:**")
                for src in response.sources:
                    st.caption(
                        f"- `{src['source']}` | strona {src['page']} "
                        f"| {src.get('article_id', '')} | {src.get('chapter', '')[:60]}…"
                    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": response.answer,
        "meta": {"strategy": response.strategy, "sources": response.sources},
    })