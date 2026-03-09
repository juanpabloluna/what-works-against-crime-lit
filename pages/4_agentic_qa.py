"""
Agentic Q&A Page - Claude orchestrates its own retrieval strategy.

This demonstrates agentic RAG where Claude:
- Decides what to search for
- Iteratively refines searches
- Determines when it has enough information
- Synthesizes the final answer
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import json
from datetime import datetime

from src.utils.auth import require_auth
require_auth()

from src.agents.agentic_qa_engine import AgenticQAEngine
from src.rag.retriever import Retriever
from src.utils.usage_logger import log_usage


# Page styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .tool-call {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .search-query {
        color: #1f77b4;
        font-weight: bold;
    }
    .result-count {
        color: #2ca02c;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<p class="main-header">🤖 Agentic Q&A</p>', unsafe_allow_html=True)

st.markdown(
    """
    **What makes this "agentic"?**

    Unlike traditional RAG that retrieves once and answers, this system gives Claude **tools**
    to search your literature corpus. Claude then:

    - 🎯 **Decides** what to search for
    - 🔄 **Iteratively refines** searches based on what it finds
    - 🧠 **Reasons** about what information it needs
    - ✅ **Determines** when it has enough information to answer

    Watch below as Claude orchestrates its own research strategy!
    """
)

# Initialize engine
@st.cache_resource
def get_agentic_engine():
    """Get cached agentic Q&A engine."""
    return AgenticQAEngine()


try:
    engine = get_agentic_engine()
    st.success("✅ Agentic engine initialized")
except Exception as e:
    st.error(f"Failed to initialize agentic engine: {e}")
    st.stop()

# Sidebar configuration
st.sidebar.header("Configuration")

# Collection filter (optional)
use_collections = st.sidebar.checkbox("Filter by collections", value=False)
collections = None
if use_collections:
    collections_input = st.sidebar.text_input(
        "Collections (comma-separated)",
        help="Example: Hot spots policing, Focused deterrence",
    )
    if collections_input:
        collections = [c.strip() for c in collections_input.split(",")]

verbose_mode = st.sidebar.checkbox(
    "Verbose mode", value=True, help="Show detailed tool execution"
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    ### How it works

    1. You ask a question
    2. Claude analyzes the question
    3. Claude searches the literature (potentially multiple times)
    4. Claude synthesizes an answer
    5. You see every step Claude takes!
    """
)

# Main Q&A interface
st.header("Ask Your Question")

question = st.text_area(
    "Question",
    placeholder="What are the most effective crime prevention strategies according to the evidence?",
    height=100,
)

@st.cache_resource
def _get_retriever_for_detection():
    """Retriever used only for author name detection."""
    return Retriever()

if st.button("Ask Question", type="primary"):
    if not question:
        st.warning("Please enter a question")
    else:
        try:
            _retr = _get_retriever_for_detection()
            detected = _retr._detect_author_names(question)
            if detected:
                st.info(
                    f"Authors detected: **{', '.join(detected)}** "
                    f"(note: the agentic engine runs its own search strategy)"
                )
        except (AttributeError, Exception):
            pass  # Stale cached Retriever or other init error

        with st.spinner("Claude is researching your question..."):
            # Get answer
            result = engine.answer_question(
                question=question,
                collections=collections,
                verbose=verbose_mode,
            )

        log_usage(
            user=st.session_state.get("user_name", "unknown"),
            page="agentic_qa",
            query=question,
            extra={"iterations": result.get("iterations", 0), "sources": len(result.get("sources", []))},
        )

        # Display results
        st.markdown("---")
        st.header("Results")

        # Show agent's thought process
        if verbose_mode and result.get("tool_calls"):
            st.subheader("🔍 Agent's Research Process")

            st.markdown(
                f"*Claude made {len(result['tool_calls'])} tool call(s) across {result['iterations']} iteration(s)*"
            )

            for i, call in enumerate(result["tool_calls"], 1):
                with st.expander(
                    f"**Step {i}:** {call['tool']} (Iteration {call['iteration']})",
                    expanded=(i <= 3),
                ):
                    # Show input
                    st.markdown("**Input:**")
                    if call["tool"] == "search_literature":
                        query = call["input"].get("query", "")
                        n_results = call["input"].get("n_results", 5)
                        st.markdown(
                            f'<div class="tool-call"><span class="search-query">Query:</span> "{query}"<br><span class="result-count">Requested:</span> {n_results} results</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.json(call["input"])

                    # Show results
                    st.markdown("**Results:**")
                    tool_result = call["result"]

                    if "results" in tool_result:
                        results = tool_result["results"]
                        st.markdown(
                            f'<span class="result-count">Found {len(results)} results</span>',
                            unsafe_allow_html=True,
                        )

                        for j, res in enumerate(results[:3], 1):  # Show top 3
                            with st.container():
                                st.markdown(f"**Result {j}:**")
                                st.markdown(f"*{res.get('title', 'Unknown')}*")
                                st.markdown(
                                    f"**Authors:** {res.get('authors', 'Unknown')} ({res.get('year', 'N/A')})"
                                )
                                st.markdown(
                                    f"**Relevance:** {res.get('relevance_score', 0):.3f}"
                                )
                                with st.expander("View text excerpt"):
                                    st.text(res.get("text", "")[:500] + "...")
                                st.markdown("---")

                        if len(results) > 3:
                            st.markdown(f"*...and {len(results) - 3} more results*")
                    else:
                        st.json(tool_result)

        # Display answer
        st.subheader("✨ Claude's Answer")

        if result.get("warning"):
            st.warning(f"⚠️ {result['warning']}")

        st.markdown(result["answer"])

        # Display sources
        if result.get("sources"):
            st.subheader("📚 Sources Consulted")
            st.markdown(
                f"*Claude consulted {len(result['sources'])} unique paper(s) to answer your question*"
            )

            for source in result["sources"]:
                st.markdown(
                    f"- **{source.get('title', 'Unknown')}** - {source.get('authors', 'Unknown')} ({source.get('year', 'N/A')})"
                )

        # Export option
        st.markdown("---")
        col1, col2 = st.columns([3, 1])

        with col2:
            # Export as JSON
            export_data = {
                "question": question,
                "answer": result["answer"],
                "timestamp": datetime.now().isoformat(),
                "iterations": result["iterations"],
                "tool_calls": result.get("tool_calls", []),
                "sources": result.get("sources", []),
            }

            st.download_button(
                label="📥 Export Result",
                data=json.dumps(export_data, indent=2, ensure_ascii=False),
                file_name=f"agentic_qa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

# Example questions
st.markdown("---")
st.header("Example Questions")

examples = [
    "What policing strategies are most effective at reducing violent crime?",
    "What is the evidence on focused deterrence programs?",
    "What methodologies are used to evaluate crime prevention programs?",
    "How effective are hot spots policing strategies?",
    "What role does community engagement play in crime reduction?",
]

st.markdown("Try these questions to see the agentic system in action:")
for example in examples:
    st.markdown(f"- {example}")
