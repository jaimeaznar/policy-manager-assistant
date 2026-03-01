import streamlit as st
import os
import time

import config
from ingest import run_ingestion
from retriever import PolicyRetriever

# Page Configuration
st.set_page_config(
    page_title="ServiZurich Policy Assistant",
    layout="wide",
)

# Session State Initialization

if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested" not in st.session_state:
    st.session_state.ingested = False


def initialize_system():
    """Run ingestion and initialize the retriever."""
    with st.spinner("Ingesting policy documents..."):
        run_ingestion()
    with st.spinner("Initializing retrieval engine..."):
        st.session_state.retriever = PolicyRetriever()
    st.session_state.ingested = True


# Sidebar
with st.sidebar:
    st.title("Configuration")

    # Mode selection
    mode = st.radio(
        "Mode",
        options=["local", "openai"],
        index=0,
        help="**Local**: Free, no API key (requires Ollama). **OpenAI**: Best quality, requires API key.",
        horizontal=True,
    )
    config.MODE = mode

    if mode == "openai":
        api_key = st.text_input(
            "OpenAI API Key",
            value=config.OPENAI_API_KEY,
            type="password",
        )
        if api_key:
            config.OPENAI_API_KEY = api_key
            os.environ["OPENAI_API_KEY"] = api_key
    else:
        st.info("Local mode uses Ollama + sentence-transformers. No API key needed.")
        st.caption(f"LLM: `{config.OLLAMA_MODEL}` | Embeddings: `{config.LOCAL_EMBEDDING_MODEL}`")

    st.divider()

    # Document management
    st.subheader("Documents")

    # Show loaded documents
    docs_dir = config.SAMPLE_DOCS_DIR
    if os.path.exists(docs_dir):
        doc_files = [f for f in os.listdir(docs_dir) if not f.startswith(".")]
        for f in doc_files:
            st.text(f"{f}")
    else:
        st.warning("No documents directory found")

    # File upload for additional documents
    uploaded_files = st.file_uploader(
        "Upload additional documents",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        os.makedirs(docs_dir, exist_ok=True)
        for uploaded_file in uploaded_files:
            file_path = os.path.join(docs_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Uploaded: {uploaded_file.name}")
        # Force re-ingestion
        st.session_state.ingested = False

    st.divider()

    # Ingestion control
    if st.button("(Re)Index Documents", use_container_width=True):
        if config.MODE == "openai" and not config.OPENAI_API_KEY:
            st.error("Please enter your OpenAI API Key first")
        else:
            initialize_system()
            st.success("Documents indexed successfully!")

    # Clear chat
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.retriever:
            st.session_state.retriever.clear_memory()
        st.rerun()

    st.divider()

    # Architecture info (for demo talking points)
    with st.expander("Architecture"):
        if config.MODE == "openai":
            st.markdown(f"""
            - **Mode:** OpenAI (cloud)
            - **LLM:** `{config.OPENAI_LLM_MODEL}`
            - **Embeddings:** `{config.OPENAI_EMBEDDING_MODEL}`
            """)
        else:
            st.markdown(f"""
            - **Mode:** Local (Ollama + sentence-transformers)
            - **LLM:** `{config.OLLAMA_MODEL}` via Ollama
            - **Embeddings:** `{config.LOCAL_EMBEDDING_MODEL}`
            """)
        st.markdown(f"""
        - **Vector Store:** ChromaDB (local)
        - **Chunk Size:** {config.CHUNK_SIZE} chars
        - **Chunk Overlap:** {config.CHUNK_OVERLAP} chars
        - **Retrieval:** Top-{config.RETRIEVAL_K} with MMR
        - **Memory:** Last {config.MEMORY_MAX_EXCHANGES} exchanges
        """)


# Main Chat Interface 
st.title("ServiZurich Policy Assistant")
st.caption(
    "Ask questions about your insurance policies — coverage, exclusions, "
    "claims process, renewals, and more."
)

# Auto-initialize if not done yet
can_initialize = (config.MODE == "local") or (config.MODE == "openai" and config.OPENAI_API_KEY)
if not st.session_state.ingested and can_initialize:
    initialize_system()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 Sources"):
                for src in message["sources"]:
                    st.text(
                        f"• {src['file']} → {src['section'].replace('_', ' ').title()}"
                    )

# Suggested Questions (shown only when chat is empty)

if not st.session_state.messages and st.session_state.ingested:
    st.markdown("**Try one of these questions:**")

    SUGGESTED_QUESTIONS = [
        # Single-policy, specific section
        "What does my home policy cover for water damage?",
        # Exclusion awareness
        "Is flood damage covered under my home insurance?",
        # Cross-document comparison
        "Compare the deductibles across all my policies",
        # Guided workflow
        "How do I renew my auto policy?",
        # Health-specific
        "What dental coverage do I have?",
        # Tests graceful "I don't know"
        "Does my policy cover cryptocurrency theft?",
    ]

    cols = st.columns(2)
    for i, question in enumerate(SUGGESTED_QUESTIONS):
        col = cols[i % 2]
        if col.button(question, key=f"suggestion_{i}", use_container_width=True):
            st.session_state.pending_question = question
            st.rerun()

# Handle suggested question click (needs rerun cycle due to Streamlit flow)
prompt = None
if "pending_question" in st.session_state:
    prompt = st.session_state.pending_question
    del st.session_state.pending_question

# Chat input (manual typing)
if prompt is None:
    prompt = st.chat_input("Ask about your insurance policy...")

if prompt:
    # Check prerequisites
    if config.MODE == "openai" and not config.OPENAI_API_KEY:
        st.error("Please enter your OpenAI API Key in the sidebar.")
        st.stop()

    if not st.session_state.retriever:
        st.error("Please index documents first using the sidebar button.")
        st.stop()

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching policies..."):
            start_time = time.time()
            result = st.session_state.retriever.query(prompt)
            elapsed = time.time() - start_time

        st.markdown(result["answer"])

        # Show sources
        if result["sources"]:
            with st.expander("Sources"):
                for src in result["sources"]:
                    st.text(
                        f"• {src['file']} → {src['section'].replace('_', ' ').title()}"
                    )

        # Show debug metadata
        with st.expander("Debug"):
            st.json({
                "response_time_seconds": round(elapsed, 2),
                **result["metadata"],
            })

    # Save to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
