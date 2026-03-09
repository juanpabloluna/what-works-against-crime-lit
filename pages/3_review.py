"""
Research Review page - Get feedback on research drafts.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from loguru import logger

from src.agents.review_engine import ReviewEngine
from src.rag.retriever import Retriever
from src.utils.usage_logger import log_usage


st.set_page_config(
    page_title="Research Review - Literature Expert",
    page_icon="✍️",
    layout="wide",
)

from src.utils.auth import require_auth
require_auth()


@st.cache_resource
def get_review_engine():
    """Initialize and cache the review engine."""
    try:
        retriever = Retriever()
        return ReviewEngine(retriever=retriever)
    except Exception as e:
        st.error(f"Error initializing review engine: {e}")
        logger.error(f"Failed to initialize review engine: {e}", exc_info=True)
        return None


def display_claim_review(claim_review, index):
    """Display a single claim review."""
    with st.expander(f"💭 Claim {index}: {claim_review.claim_text[:80]}..."):
        st.markdown(f"**Full Claim:** {claim_review.claim_text}")

        st.markdown("**Assessment:**")
        st.markdown(claim_review.assessment)

        if claim_review.supporting_evidence:
            st.markdown("**Supporting Evidence:**")
            for i, evidence in enumerate(claim_review.supporting_evidence[:3], 1):
                st.markdown(f"{i}. {evidence}")

        if claim_review.contradicting_evidence:
            st.markdown("**Contradicting Evidence:**")
            for i, evidence in enumerate(claim_review.contradicting_evidence[:3], 1):
                st.markdown(f"{i}. {evidence}")

        if claim_review.sources:
            st.markdown(f"**Sources:** {len(claim_review.sources)} papers")


def main():
    """Main review page."""

    st.title("✍️ Research Review")
    st.markdown("Get feedback on your research drafts with evidence from your literature corpus.")

    # Initialize session state
    if "review_report" not in st.session_state:
        st.session_state.review_report = None
    if "quick_check" not in st.session_state:
        st.session_state.quick_check = None

    # Get review engine
    review_engine = get_review_engine()
    if not review_engine:
        st.error("⚠️ Review Engine failed to initialize. Please check your configuration.")
        st.stop()

    # Sidebar configuration
    with st.sidebar:
        st.markdown("### Review Settings")

        # Collection filter
        st.markdown("**Filter by Collections**")
        collection_choice = st.radio(
            "Collection Filter",
            ["All Collections", "Specific Collections"],
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

        st.markdown("---")
        st.markdown("### Review Type")

        review_type = st.radio(
            "Choose review type",
            ["Full Review", "Quick Citation Check"],
            help="Full review analyzes claims and provides comprehensive feedback. Quick check focuses on citations."
        )

    # Main content
    tab1, tab2, tab3 = st.tabs(["📝 Submit Draft", "📊 Results", "💡 Tips"])

    with tab1:
        st.markdown("## Submit Your Research Draft")

        # Draft input
        draft_text = st.text_area(
            "Paste your research text here",
            height=300,
            placeholder="Paste your research draft, paper section, or argument here...",
            help="Can be a full paper, a section, or specific paragraphs you want feedback on"
        )

        # Word count
        if draft_text:
            word_count = len(draft_text.split())
            st.info(f"📊 Word count: {word_count:,} words ({len(draft_text):,} characters)")

        # Review button
        col1, col2 = st.columns([2, 1])
        with col1:
            if review_type == "Full Review":
                review_button = st.button(
                    "🔍 Get Full Review",
                    type="primary",
                    use_container_width=True
                )
            else:
                review_button = st.button(
                    "⚡ Quick Citation Check",
                    type="primary",
                    use_container_width=True
                )

        # Process review
        if review_button and draft_text:
            if review_type == "Full Review":
                with st.spinner("🤔 Analyzing your research... This may take 1-2 minutes..."):
                    try:
                        report = review_engine.review_research(
                            draft_text=draft_text,
                            collections=collections,
                        )

                        st.session_state.review_report = report
                        st.success(f"✅ Review completed in {report.generation_time:.2f} seconds!")
                        st.info("Switch to the 'Results' tab to see your feedback.")

                        log_usage(
                            user=st.session_state.get("user_name", "unknown"),
                            page="review",
                            query=draft_text[:200],
                            extra={"type": "full_review", "claims": len(report.claim_reviews)},
                        )

                    except Exception as e:
                        st.error(f"❌ Error generating review: {e}")
                        logger.error(f"Error in review: {e}", exc_info=True)

            else:  # Quick citation check
                with st.spinner("⚡ Checking citations..."):
                    try:
                        check_result = review_engine.quick_citation_check(
                            draft_text=draft_text,
                            collections=collections,
                        )

                        st.session_state.quick_check = check_result
                        st.success("✅ Citation check completed!")
                        st.info("Switch to the 'Results' tab to see the results.")

                        log_usage(
                            user=st.session_state.get("user_name", "unknown"),
                            page="review",
                            query=draft_text[:200],
                            extra={"type": "citation_check"},
                        )

                    except Exception as e:
                        st.error(f"❌ Error in citation check: {e}")
                        logger.error(f"Error in citation check: {e}", exc_info=True)

    with tab2:
        st.markdown("## Review Results")

        if review_type == "Full Review" and st.session_state.review_report:
            report = st.session_state.review_report

            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Claims Analyzed", len(report.claim_reviews))
            with col2:
                st.metric("Suggested Citations", len(report.citation_suggestions))
            with col3:
                st.metric("Literature Gaps", len(report.literature_gaps))
            with col4:
                st.metric("Review Time", f"{report.generation_time:.1f}s")

            st.markdown("---")

            # Overall assessment
            st.markdown("## 📋 Overall Assessment")
            st.markdown(report.overall_assessment)

            st.markdown("---")

            # Claim reviews
            if report.claim_reviews:
                st.markdown(f"## 💭 Claim Analysis ({len(report.claim_reviews)} claims)")
                for i, claim_review in enumerate(report.claim_reviews, 1):
                    display_claim_review(claim_review, i)

            st.markdown("---")

            # Literature gaps
            if report.literature_gaps:
                st.markdown(f"## 🔍 Literature Gaps ({len(report.literature_gaps)})")
                st.markdown("Areas where your draft could benefit from engagement with existing literature:")
                for gap in report.literature_gaps:
                    st.markdown(f"- {gap}")

            st.markdown("---")

            # Citation suggestions
            if report.citation_suggestions:
                st.markdown(f"## 📚 Suggested Citations ({len(report.citation_suggestions)} papers)")
                st.markdown("Papers that appear relevant to your draft and might strengthen your argument:")

                for i, source in enumerate(report.citation_suggestions, 1):
                    with st.expander(f"📄 {source.title}"):
                        authors = ", ".join(source.authors[:3])
                        if len(source.authors) > 3:
                            authors += " et al."
                        st.markdown(f"**Authors:** {authors}")
                        st.markdown(f"**Year:** {source.year}")
                        st.markdown(f"**Citation:** {source.get_citation_text()}")

                        if source.abstract:
                            st.markdown("**Abstract:**")
                            st.markdown(source.abstract[:400] + "...")

            # Export
            st.markdown("---")
            st.markdown("### 📥 Export Review")

            export_text = f"""# Research Review Report

## Overall Assessment

{report.overall_assessment}

## Claim Analysis

"""
            for i, claim_review in enumerate(report.claim_reviews, 1):
                export_text += f"""### Claim {i}

**Claim:** {claim_review.claim_text}

**Assessment:** {claim_review.assessment}

"""

            export_text += """## Literature Gaps

"""
            for gap in report.literature_gaps:
                export_text += f"- {gap}\n"

            export_text += """\n## Suggested Citations\n\n"""
            for i, source in enumerate(report.citation_suggestions, 1):
                export_text += f"{i}. {source.get_full_citation()}\n"

            st.download_button(
                "📄 Download Review as Markdown",
                export_text,
                file_name="research_review.md",
                mime="text/markdown",
                use_container_width=True,
            )

        elif review_type == "Quick Citation Check" and st.session_state.quick_check:
            check = st.session_state.quick_check

            st.markdown("## ⚡ Citation Check Results")

            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Citations", check["total_citations"])
            with col2:
                st.metric("Unique Citations", check["unique_citations"])
            with col3:
                st.metric("Relevant Papers Found", check["relevant_papers_found"])
            with col4:
                density_pct = check["citation_density"] * 100
                st.metric("Citation Density", f"{density_pct:.1f}%")

            st.markdown("---")

            # Citation coverage
            st.markdown("### 📊 Citation Coverage")
            st.progress(check["already_cited"] / max(1, check["relevant_papers_found"]))
            st.markdown(f"You've cited **{check['already_cited']} out of {check['relevant_papers_found']}** relevant papers found in your library.")

            # Suggestions
            if check["suggested_to_add"]:
                st.markdown("---")
                st.markdown(f"### 📚 Papers You Might Want to Cite ({len(check['suggested_to_add'])})")

                for source in check["suggested_to_add"]:
                    st.markdown(f"- **{source.title}** - {source.get_citation_text()}")

        else:
            st.info("👈 Submit your draft in the 'Submit Draft' tab to see results here.")

    with tab3:
        st.markdown("""
        ## 💡 Tips for Getting Good Feedback

        ### What Makes a Good Draft for Review

        **Length:**
        - 500+ words: Enough for meaningful feedback
        - 1,000-3,000 words: Ideal for comprehensive review
        - Full papers: Works, but may hit token limits

        **Content Type:**
        - ✅ Argument sections with claims
        - ✅ Literature review sections
        - ✅ Introduction/theory sections
        - ❌ Methods sections (less relevant)
        - ❌ Pure data tables or statistics

        **Stage:**
        - Early drafts: Get broad feedback
        - Near-final drafts: Check citation coverage
        - Published work: Retrospective analysis

        ### Two Types of Review

        **Full Review:**
        - Analyzes main claims
        - Finds supporting/contradicting evidence
        - Identifies literature gaps
        - Suggests citations
        - Takes 1-2 minutes
        - Costs ~$0.40-0.80

        **Quick Citation Check:**
        - Counts existing citations
        - Finds relevant papers
        - Suggests additional citations
        - Fast (<30 seconds)
        - Costs ~$0.10-0.20

        ### Interpreting Results

        **Claim Analysis:**
        - Shows how each claim relates to literature
        - Supporting evidence strengthens your argument
        - Contradicting evidence = engage with counterarguments
        - No evidence = potential gap or new contribution

        **Literature Gaps:**
        - Areas to strengthen with citations
        - May indicate novel contributions
        - Could suggest additional reading needed

        **Citation Suggestions:**
        - Relevant papers from your library
        - Read abstracts to verify relevance
        - Not all suggestions will fit your argument
        - Use judgment on what to include

        ### Using Collection Filters

        **For Focused Feedback:**
        - Filter to collections related to your topic
        - Gets more relevant citations
        - Reduces noise from unrelated work

        **For Comprehensive Coverage:**
        - Use "All Collections"
        - May find unexpected connections
        - Broader literature review

        ### Best Practices

        **Before Submitting:**
        1. ✅ Include your key arguments
        2. ✅ Have some existing citations (to compare)
        3. ✅ Focus on substantive claims
        4. ❌ Don't submit incomplete fragments

        **After Getting Feedback:**
        1. ✅ Read suggested papers' abstracts
        2. ✅ Verify evidence actually supports your claims
        3. ✅ Engage with contradicting evidence
        4. ✅ Fill identified gaps
        5. ❌ Don't blindly accept all suggestions

        ### Example Use Cases

        **PhD Students:**
        - Review dissertation chapters
        - Check literature review coverage
        - Find supporting evidence for claims
        - Identify gaps before defense

        **Researchers:**
        - Pre-submission paper check
        - Response to reviewers (find evidence)
        - Grant proposal literature section
        - Conference paper preparation

        **Iterative Writing:**
        1. Write initial draft
        2. Get review
        3. Add suggested citations
        4. Revise claims based on feedback
        5. Quick citation check
        6. Repeat until satisfied

        ### Limitations

        **Remember:**
        - AI can make mistakes - verify everything
        - Only searches YOUR library (not all literature)
        - Relevance judgments may differ from yours
        - Can't replace human peer review
        - Use as a tool, not a substitute for thinking

        ### Performance Tips

        **For Faster Reviews:**
        - Use Quick Citation Check
        - Submit shorter excerpts
        - Filter to specific collections

        **For Better Quality:**
        - Submit focused sections
        - Include key arguments
        - Have clear claims
        - Use relevant collections

        ### Cost Management

        - Full review of 2000 words: ~$0.50
        - Quick check: ~$0.15
        - Budget accordingly if reviewing multiple drafts
        - Consider reviewing sections separately

        ---

        ## Ready to get feedback? Submit your draft! 📝
        """)


if __name__ == "__main__":
    main()
