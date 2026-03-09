"""
Main Streamlit application for Literature Expert Agent (Streamlit Cloud).
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import streamlit as st
from loguru import logger

# Bump this on every deploy to force cache invalidation
_APP_VERSION = "1.0"

# Page configuration — must be the first Streamlit command
st.set_page_config(
    page_title="What Works Against Crime - Literature Expert",
    page_icon="\U0001F4DA",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Auto-clear stale cache after deploy
if st.session_state.get("_app_version") != _APP_VERSION:
    st.cache_resource.clear()
    st.session_state["_app_version"] = _APP_VERSION


from src.utils.auth import require_auth

# --- Password gate (blocks all content below if not authenticated) ---
require_auth()

# --- Main app (only runs after authentication) ---

# Inject Streamlit Cloud secrets into environment for pydantic-settings
try:
    for key, value in st.secrets.items():
        if isinstance(value, str) and key not in os.environ:
            os.environ[key] = value
except Exception:
    pass  # No secrets configured (local dev)

from src.config.settings import settings
from src.embeddings.vector_store import VectorStore
from src.rag.retriever import Retriever

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .stat-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f77b4;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_retriever():
    """Initialize and cache the retriever."""
    try:
        return Retriever()
    except Exception as e:
        st.error(f"Error initializing retriever: {e}")
        logger.error(f"Failed to initialize retriever: {e}", exc_info=True)
        return None


def main():
    """Main application page."""

    # Header
    st.markdown(
        '<div class="main-header">What Works Against Crime - Literature Expert</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">RAG-based research assistant for crime prevention, policing, and public safety interventions</div>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.markdown("### Navigation")
        st.markdown("""
        Use the pages in the sidebar to:
        - **Question Answering**: Traditional RAG Q&A
        - **Agentic Q&A**: Claude orchestrates its own research
        - **Literature Synthesis**: Generate literature reviews
        - **Research Review**: Get feedback on drafts
        """)

        st.markdown("---")

        if st.button("Clear cache & reload", use_container_width=True,
                      help="Force reload of all engines after a code update"):
            st.cache_resource.clear()
            st.rerun()

        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This agent helps you query a corpus of academic papers
        on crime prevention, policing, and public safety
        interventions through:
        - Semantic search across papers
        - Citation-backed answers
        - Literature synthesis
        - Research draft review
        """)

    # Main content
    tab1, tab2, tab3 = st.tabs(["Overview", "System Status", "Quick Start"])

    with tab1:
        st.markdown("## System Overview")

        col1, col2, col3 = st.columns(3)

        retriever = get_retriever()
        if retriever:
            try:
                stats = retriever.get_stats()
                vs_stats = stats.get("vector_store", {})

                with col1:
                    st.markdown(f"""
                    <div class="stat-box">
                        <div class="stat-value">{vs_stats.get('total_chunks', 0):,}</div>
                        <div class="stat-label">Total Chunks</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div class="stat-box">
                        <div class="stat-value">{vs_stats.get('sample_unique_items', 0):,}</div>
                        <div class="stat-label">Documents Indexed</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    year_range = vs_stats.get("year_range", (None, None))
                    year_text = f"{year_range[0]}-{year_range[1]}" if year_range[0] else "N/A"
                    st.markdown(f"""
                    <div class="stat-box">
                        <div class="stat-value">{year_text}</div>
                        <div class="stat-label">Year Range</div>
                    </div>
                    """, unsafe_allow_html=True)

                if vs_stats.get("sample_collections"):
                    st.markdown("### Collections in Database")
                    cols = st.columns(2)
                    for i, coll in enumerate(vs_stats["sample_collections"][:6]):
                        with cols[i % 2]:
                            st.markdown(f"- {coll}")

            except Exception as e:
                st.error(f"Error loading statistics: {e}")
                logger.error(f"Error in stats display: {e}", exc_info=True)
        else:
            st.warning("Retriever not initialized. Please check configuration.")

    with tab2:
        st.markdown("## System Status")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Configuration")
            st.code(f"""
Embedding Model: {settings.embedding_model}
LLM Model: {settings.llm_model}
Chunk Size: {settings.chunk_size} tokens
Chunk Overlap: {settings.chunk_overlap} tokens
Top-K Results: {settings.top_k}
            """)

        with col2:
            st.markdown("### Paths")
            st.code(f"""
ChromaDB: {settings.chromadb_path}
            """)

        if retriever:
            st.markdown("### Vector Database")
            try:
                count = retriever.vector_store.count()
                st.success(f"Connected - {count:,} chunks indexed")
            except Exception as e:
                st.error(f"Error: {e}")

        st.markdown("### API Configuration")
        if settings.anthropic_api_key and settings.anthropic_api_key != "YOUR_API_KEY_HERE":
            st.success("Anthropic API key configured")
        else:
            st.error("Anthropic API key not configured")

    with tab3:
        st.markdown("## Quick Start Guide")

        st.markdown("""
        ### How to Use

        1. **Question Answering** -- Ask questions about the literature on what works against crime.
           Filter by year range. Get citation-backed answers.

        2. **Agentic Q&A** -- Claude autonomously decides which searches to run,
           iterating until it has enough evidence. Best for complex questions.

        3. **Literature Synthesis** -- Enter a topic and generate a structured
           literature review with bibliography.

        4. **Research Review** -- Paste a research draft to get feedback
           with supporting/contradicting evidence from the corpus.

        ### Tips

        - Be specific in your questions for best results
        - Check sources to see which papers were used
        """)

        st.markdown("### Example Questions")
        examples = [
            "What policing strategies are most effective at reducing violent crime?",
            "Does community-based violence intervention work?",
            "What is the evidence on the effectiveness of incarceration for reducing crime?",
            "What are the most promising crime prevention programs for youth?",
            "How effective are hot spots policing strategies?",
        ]

        for example in examples:
            if st.button(f"{example}", key=example, use_container_width=True):
                st.session_state["example_question"] = example
                st.info("Go to the **Question Answering** page and paste this question!")


if __name__ == "__main__":
    main()
