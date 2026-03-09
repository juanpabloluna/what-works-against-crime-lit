"""
Question Answering page for the Literature Expert Agent.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from loguru import logger

from src.agents.qa_engine import QAEngine
from src.rag.retriever import Retriever
from src.utils.usage_logger import log_usage


st.set_page_config(
    page_title="Q&A - Literature Expert",
    page_icon="❓",
    layout="wide",
)

from src.utils.auth import require_auth
require_auth()


@st.cache_resource
def get_qa_engine():
    """Initialize and cache the QA engine."""
    try:
        retriever = Retriever()
        return QAEngine(retriever=retriever)
    except Exception as e:
        st.error(f"Error initializing QA engine: {e}")
        logger.error(f"Failed to initialize QA engine: {e}", exc_info=True)
        return None


def display_source(source, index):
    """Display a source document in an expandable format."""
    with st.expander(f"📄 {source.title}"):
        col1, col2 = st.columns([3, 1])

        with col1:
            if source.authors:
                authors = ", ".join(source.authors[:3])
                if len(source.authors) > 3:
                    authors += " et al."
                st.markdown(f"**Authors:** {authors}")

            if source.year:
                st.markdown(f"**Year:** {source.year}")

            if source.publication:
                st.markdown(f"**Publication:** {source.publication}")

        with col2:
            if source.collections:
                st.markdown(f"**Collections:**")
                for coll in source.collections[:3]:
                    st.markdown(f"- {coll}")

        if source.abstract:
            st.markdown("**Abstract:**")
            st.markdown(source.abstract[:500] + ("..." if len(source.abstract) > 500 else ""))

        if source.doi:
            st.markdown(f"**DOI:** [{source.doi}](https://doi.org/{source.doi})")

        if source.pdf_path:
            st.markdown(f"**PDF:** `{source.pdf_path}`")


def main():
    """Main Q&A page."""

    st.title("❓ Question Answering")
    st.markdown("Ask questions about the crime prevention literature and get answers with citations.")

    # Initialize session state
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []
    if "current_answer" not in st.session_state:
        st.session_state.current_answer = None

    # Get QA engine
    qa_engine = get_qa_engine()
    if not qa_engine:
        st.error("⚠️ QA Engine failed to initialize. Please check your configuration and try again.")
        st.stop()

    # Sidebar filters
    with st.sidebar:
        st.markdown("### Query Settings")

        # Author index diagnostic
        try:
            idx_size = len(qa_engine.retriever._author_lookup)
            if idx_size > 0:
                st.caption(f"Author index: {idx_size} names loaded")
            else:
                st.warning("Author index empty — author-aware search disabled")
        except AttributeError:
            st.warning("Hybrid retrieval not available (clear cache)")


        # Collection filter
        st.markdown("**Filter by Collections**")
        collection_options = ["All Collections", "Specific Collections"]
        collection_choice = st.radio(
            "Collection Filter",
            collection_options,
            label_visibility="collapsed"
        )

        collections = None
        if collection_choice == "Specific Collections":
            collection_input = st.text_area(
                "Collection Names (one per line)",
                help="Enter collection names, one per line",
                height=100
            )
            if collection_input:
                collections = [c.strip() for c in collection_input.split("\n") if c.strip()]
                st.info(f"Filtering by {len(collections)} collection(s)")

        # Year filter
        st.markdown("**Filter by Year**")
        use_year_filter = st.checkbox("Enable year filter")

        min_year = None
        max_year = None
        if use_year_filter:
            col1, col2 = st.columns(2)
            with col1:
                min_year = st.number_input("From Year", min_value=1900, max_value=2030, value=2000, step=1)
            with col2:
                max_year = st.number_input("To Year", min_value=1900, max_value=2030, value=2025, step=1)

        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            n_results = st.slider(
                "Number of chunks to retrieve",
                min_value=5,
                max_value=30,
                value=10,
                help="More chunks provide more context but increase API costs"
            )

        # Clear history
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.qa_history = []
            st.session_state.current_answer = None
            st.rerun()

    # Main content
    # Question input
    question = st.text_area(
        "Your Question",
        height=100,
        placeholder="Example: What policing strategies are most effective at reducing violent crime?",
        help="Ask a specific question about your literature"
    )

    # Check for example question from main page
    if "example_question" in st.session_state and not question:
        question = st.session_state.example_question
        del st.session_state.example_question
        st.rerun()

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        ask_button = st.button("🔍 Ask Question", type="primary", use_container_width=True)
    with col2:
        use_history = st.checkbox("Use conversation history", value=False)

    # Process question
    if ask_button and question:
        # Show author detection feedback
        try:
            detected = qa_engine.retriever._detect_author_names(question)
            if detected:
                st.info(f"Author-aware search activated for: **{', '.join(detected)}**")
            else:
                st.caption("No author names detected in query — using semantic search")
        except AttributeError:
            st.caption("Author detection unavailable")

        with st.spinner("🤔 Thinking... Retrieving relevant papers and generating answer..."):
            try:
                # Generate answer
                if use_history and st.session_state.qa_history:
                    answer = qa_engine.answer_with_conversation_history(
                        question=question,
                        conversation_history=st.session_state.qa_history,
                        collections=collections,
                        n_results=n_results,
                    )
                else:
                    answer = qa_engine.answer_question(
                        question=question,
                        collections=collections,
                        n_results=n_results,
                        min_year=min_year,
                        max_year=max_year,
                    )

                # Store in session state
                st.session_state.current_answer = answer
                st.session_state.qa_history.append({
                    "question": question,
                    "answer": answer.answer,
                    "sources": answer.sources,
                })

                st.success(f"✅ Answer generated in {answer.generation_time:.2f} seconds!")

                log_usage(
                    user=st.session_state.get("user_name", "unknown"),
                    page="qa",
                    query=question,
                    extra={"sources": len(answer.sources), "time": round(answer.generation_time, 2)},
                )

            except Exception as e:
                st.error(f"❌ Error generating answer: {e}")
                logger.error(f"Error in Q&A: {e}", exc_info=True)

    # Display current answer
    if st.session_state.current_answer:
        answer = st.session_state.current_answer

        st.markdown("---")
        st.markdown("## 💡 Answer")

        # Answer text
        st.markdown(answer.answer)

        # Metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sources Used", len(answer.sources))
        with col2:
            st.metric("Chunks Retrieved", answer.chunks_used)
        with col3:
            st.metric("Generation Time", f"{answer.generation_time:.2f}s")

        # Sources
        if answer.sources:
            st.markdown("---")
            st.markdown(f"## 📚 Sources ({len(answer.sources)} papers)")

            for i, source in enumerate(answer.sources):
                display_source(source, i)

        # Export options
        st.markdown("---")
        st.markdown("### 📥 Export")

        col1, col2 = st.columns(2)

        with col1:
            # Export as markdown
            export_md = f"""# Question
{answer.question}

# Answer
{answer.answer}

# Sources
"""
            for i, source in enumerate(answer.sources, 1):
                export_md += f"\n{i}. {source.get_full_citation()}\n"

            st.download_button(
                "Download as Markdown",
                export_md,
                file_name="answer.md",
                mime="text/markdown",
                use_container_width=True,
            )

        with col2:
            # Copy to clipboard
            if st.button("📋 Copy Answer to Clipboard", use_container_width=True):
                st.info("Use the markdown export button to save the full answer with sources!")

    # Show conversation history
    if st.session_state.qa_history:
        st.markdown("---")
        st.markdown("## 📜 Conversation History")

        with st.expander(f"View {len(st.session_state.qa_history)} previous questions"):
            for i, qa in enumerate(reversed(st.session_state.qa_history), 1):
                st.markdown(f"### Q{len(st.session_state.qa_history) - i + 1}: {qa['question']}")
                st.markdown(qa['answer'][:300] + "..." if len(qa['answer']) > 300 else qa['answer'])
                st.markdown(f"*{len(qa.get('sources', []))} sources*")
                st.markdown("---")

    # Help section
    with st.expander("💡 Tips for Better Results"):
        st.markdown("""
        ### How to Ask Good Questions

        **Be Specific:**
        - ✅ "What policing strategies are most effective at reducing violent crime?"
        - ❌ "Tell me about crime"

        **Focus on Your Research Domain:**
        - ✅ "What is the evidence on hot spots policing effectiveness?"
        - ❌ "What is crime?"

        **Use Academic Framing:**
        - ✅ "What theories explain the relationship between incarceration and recidivism?"
        - ❌ "Does prison work?"

        ### Using Filters

        - **Collections**: Narrow results by year
        - **Year Range**: Focus on recent research or historical periods
        - **Chunks**: More chunks = more comprehensive but slower and more expensive

        ### Conversation History

        - Enable "Use conversation history" for follow-up questions
        - The system will consider previous Q&A for context
        - Great for exploring a topic in depth

        ### Understanding Citations

        - All citations use [Author, Year] format
        - Click on sources to see full details
        - Check the abstract to verify relevance
        """)


if __name__ == "__main__":
    main()
