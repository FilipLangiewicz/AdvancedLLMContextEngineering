import streamlit as st
from config.settings import ChunkingStrategy, LLMProvider
from compression.base import CompressionStrategy
from query.base import QueryUnderstandingStrategy
from reranking.base import RerankingStrategy
from generation.embeddings_factory import build_embeddings
from pipeline import build_pipeline

st.set_page_config(
    page_title="Asystent Prawny",
    page_icon="⚖",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif !important;
}

.stApp {
    background-color: #f5f4f0;
}

/* ====== SIDEBAR ====== */
[data-testid="stSidebar"] {
    background-color: #1a2038 !important;
}
[data-testid="stSidebar"] > div {
    background-color: #1a2038 !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #c8cdd9 !important;
    font-size: 0.85rem !important;
}
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-family: 'Lora', serif !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    padding-bottom: 0.7rem;
    border-bottom: 1px solid #2d3561;
    margin-bottom: 1.2rem;
}
.sidebar-section {
    display: block;
    color: #6e7898 !important;
    font-size: 0.67rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    margin: 1.5rem 0 0.3rem 0 !important;
}
[data-testid="stSidebar"] hr {
    border-color: #2d3561 !important;
    margin: 1.5rem 0 !important;
}
[data-testid="stSidebar"] .stButton > button {
    background-color: transparent !important;
    border: 1px solid #2d3561 !important;
    color: #6e7898 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em !important;
    transition: all 0.2s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    border-color: #5a6490 !important;
    color: #c8cdd9 !important;
    background-color: #222c4a !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background-color: #222c4a !important;
    border-color: #2d3561 !important;
    color: #c8cdd9 !important;
}

/* ====== HEADER ====== */
.legal-header {
    display: flex;
    align-items: baseline;
    gap: 1.2rem;
    padding: 1.4rem 0 1rem 0;
    border-bottom: 2px solid #1a2038;
    margin-bottom: 2rem;
}
.legal-header-title {
    font-family: 'Lora', Georgia, serif;
    font-size: 1.45rem;
    font-weight: 600;
    color: #1a2038;
    letter-spacing: 0.01em;
    margin: 0;
    line-height: 1;
}
.legal-header-subtitle {
    font-size: 0.7rem;
    font-weight: 500;
    color: #9ca3af;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 0;
    line-height: 1;
}

/* ====== CHAT ====== */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse !important;
    margin-left: 18% !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
    background-color: #1a2038 !important;
    border: none !important;
    border-radius: 4px 4px 0 4px !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] p {
    color: #eef0f5 !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    margin-right: 18% !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {
    background-color: #ffffff !important;
    border: 1px solid #e2dfd6 !important;
    border-radius: 4px 4px 4px 0 !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] p {
    color: #1a2038 !important;
}
[data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"] {
    display: none !important;
}

/* Chat input */
[data-testid="stChatInputTextArea"] {
    background-color: #ffffff !important;
    border: 1px solid #d4d0c8 !important;
    border-radius: 2px !important;
    color: #1a2038 !important;
    font-size: 0.9rem !important;
}

/* Expander */
[data-testid="stExpander"] {
    background-color: #faf9f6 !important;
    border: 1px solid #e2dfd6 !important;
    border-radius: 2px !important;
}
[data-testid="stExpander"] summary p {
    font-size: 0.75rem !important;
    color: #9ca3af !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
}

footer { display: none !important; }
#MainMenu { visibility: hidden !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #f5f4f0; }
::-webkit-scrollbar-thumb { background: #c8cdd9; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


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
            RerankingStrategy.U_SHAPE_REORDER: "Reorder U-Shape",
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
    with st.spinner("Budowanie potoku przetwarzania…"):
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
    <span class="legal-header-title">Asystent Prawny</span>
    <span class="legal-header-subtitle">System analizy dokumentów prawnych</span>
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