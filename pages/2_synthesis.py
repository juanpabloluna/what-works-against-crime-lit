"""
Literature Synthesis page - Generate comprehensive literature reviews.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from loguru import logger

from src.agents.synthesis_engine import SynthesisEngine
from src.rag.retriever import Retriever
from src.utils.usage_logger import log_usage


st.set_page_config(
    page_title="Literature Synthesis - Literature Expert",
    page_icon="📝",
    layout="wide",
)

from src.utils.auth import require_auth
require_auth()


@st.cache_resource
def get_synthesis_engine():
    """Initialize and cache the synthesis engine."""
    try:
        retriever = Retriever()
        return SynthesisEngine(retriever=retriever)
    except Exception as e:
        st.error(f"Error initializing synthesis engine: {e}")
        logger.error(f"Failed to initialize synthesis engine: {e}", exc_info=True)
        return None


def main():
    """Main synthesis page."""

    st.title("📝 Literature Synthesis")
    st.markdown("Generate comprehensive literature reviews on specific topics.")

    # Initialize session state
    if "literature_review" not in st.session_state:
        st.session_state.literature_review = None

    # Get synthesis engine
    synthesis_engine = get_synthesis_engine()
    if not synthesis_engine:
        st.error("⚠️ Synthesis Engine failed to initialize. Please check your configuration.")
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

        # Year filter
        st.markdown("**Filter by Year**")
        use_year_filter = st.checkbox("Enable year filter")

        min_year = None
        max_year = None
        if use_year_filter:
            col1, col2 = st.columns(2)
            with col1:
                min_year = st.number_input("From Year", min_value=1900, max_value=2030, value=2000)
            with col2:
                max_year = st.number_input("To Year", min_value=1900, max_value=2030, value=2025)

        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            n_papers = st.slider(
                "Number of papers to include",
                min_value=10,
                max_value=50,
                value=30,
                help="More papers = more comprehensive but slower and more expensive"
            )

            custom_sections = st.checkbox("Customize sections")

    # Main content
    tab1, tab2 = st.tabs(["📄 Generate Review", "💡 Tips & Examples"])

    with tab1:
        # Topic input
        topic = st.text_area(
            "Research Topic or Question",
            height=100,
            placeholder="Example: Evidence on the effectiveness of hot spots policing",
            help="Enter the topic or research question for your literature review"
        )

        # Custom sections
        sections = None
        if custom_sections:
            st.markdown("**Custom Section Titles** (one per line)")
            sections_input = st.text_area(
                "Section titles",
                value="Introduction\nKey Concepts and Theoretical Framework\nEmpirical Findings\nDebates and Contradictions\nGaps and Future Directions",
                height=150,
                label_visibility="collapsed"
            )
            if sections_input:
                sections = [s.strip() for s in sections_input.split("\n") if s.strip()]

        # Generate button
        col1, col2 = st.columns([2, 1])
        with col1:
            generate_button = st.button("📝 Generate Literature Review", type="primary", use_container_width=True)

        # Generate review
        if generate_button and topic:
            try:
                detected = synthesis_engine.retriever._detect_author_names(topic)
                if detected:
                    st.info(f"Author-aware search activated for: **{', '.join(detected)}**")
                if len(synthesis_engine.retriever._author_lookup) == 0:
                    st.warning("Author index is empty — hybrid retrieval disabled. Try clearing cache on the home page.")
            except AttributeError:
                pass  # Stale cached engine without hybrid retrieval

            with st.spinner("🔍 Generating literature review... This may take 1-2 minutes..."):
                try:
                    review = synthesis_engine.generate_literature_review(
                        topic=topic,
                        collections=collections,
                        sections=sections,
                        n_papers=n_papers,
                        min_year=min_year,
                        max_year=max_year,
                    )

                    st.session_state.literature_review = review
                    st.success(f"✅ Literature review generated in {review.generation_time:.2f} seconds!")

                    log_usage(
                        user=st.session_state.get("user_name", "unknown"),
                        page="synthesis",
                        query=topic,
                        extra={"papers": len(review.bibliography), "time": round(review.generation_time, 2)},
                    )

                except Exception as e:
                    st.error(f"❌ Error generating review: {e}")
                    logger.error(f"Error in synthesis: {e}", exc_info=True)

        # Display review
        if st.session_state.literature_review:
            review = st.session_state.literature_review

            st.markdown("---")
            st.markdown(f"# {review.title}")

            # Metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Papers Reviewed", len(review.bibliography))
            with col2:
                st.metric("Sections", len(review.sections))
            with col3:
                st.metric("Generation Time", f"{review.generation_time:.1f}s")

            st.markdown("---")

            # Introduction
            st.markdown("## Introduction")
            st.markdown(review.introduction)

            # Sections
            for section_title, section_content in review.sections:
                st.markdown(f"## {section_title}")
                st.markdown(section_content)

            # Conclusion
            st.markdown("## Conclusion")
            st.markdown(review.conclusion)

            # Bibliography
            st.markdown("## References")
            for i, citation in enumerate(review.bibliography, 1):
                st.markdown(f"{i}. {citation}")

            # Export
            st.markdown("---")
            st.markdown("### 📥 Export")

            # Build markdown export
            export_md = f"""# {review.title}

## Introduction

{review.introduction}

"""
            for section_title, section_content in review.sections:
                export_md += f"""## {section_title}

{section_content}

"""
            export_md += f"""## Conclusion

{review.conclusion}

## References

"""
            for i, citation in enumerate(review.bibliography, 1):
                export_md += f"{i}. {citation}\n"

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📄 Download as Markdown",
                    export_md,
                    file_name=f"literature_review_{topic[:30].replace(' ', '_')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

            with col2:
                # Build Word-compatible format
                if st.button("📋 Copy to Clipboard", use_container_width=True):
                    st.info("Use the Download button to save the review!")

    with tab2:
        st.markdown("""
        ## 💡 Tips for Generating Literature Reviews

        ### Choosing a Good Topic

        **Be Specific:**
        - ✅ "Theories of crime prevention and deterrence"
        - ❌ "Crime in Latin America" (too broad)

        **Frame as a Question or Theme:**
        - ✅ "What explains variation in gang violence?"
        - ✅ "The relationship between policing strategies and crime reduction"

        ### Using Filters Effectively

        **Collections:**
        - Focus on thematic collections for targeted reviews
        - Use "All Collections" for comprehensive coverage

        **Year Range:**
        - Recent literature (last 5 years) for current debates
        - Historical range for tracing evolution of ideas
        - Specific periods for historical analysis

        **Number of Papers:**
        - 10-15 papers: Focused review on specific aspect
        - 20-30 papers: Standard comprehensive review (recommended)
        - 40-50 papers: Very comprehensive, slower, more expensive

        ### Customizing Sections

        Default sections work well for most academic reviews:
        - Introduction
        - Key Concepts and Theoretical Framework
        - Empirical Findings
        - Debates and Contradictions
        - Gaps and Future Directions

        **Custom sections** you might want:
        - Methodological Approaches
        - Regional Variations
        - Historical Evolution
        - Policy Implications
        - Case Studies

        ### What to Expect

        **Quality:**
        - Synthesizes across multiple papers
        - Identifies key debates and themes
        - Includes proper citations
        - Academic writing style

        **Not perfect:**
        - Always review and edit the output
        - Verify citations are accurate
        - Add your own analysis and interpretation
        - Check for any errors or misrepresentations

        ### Example Topics

        **Theoretical Reviews:**
        - "Theories of situational crime prevention"
        - "The security-development nexus in Latin America"
        - "Social control theories in gang research"

        **Empirical Reviews:**
        - "Quantitative studies of homicide rates in Latin America"
        - "Ethnographic research on gang life"
        - "Natural experiments in policing interventions"

        **Methodological Reviews:**
        - "Methods for studying illicit markets"
        - "Challenges in crime data collection in Latin America"
        - "Randomized controlled trials in criminology"

        **Regional/Temporal Reviews:**
        - "Literature on drug trafficking in Mexico, 2010-2020"
        - "Evolution of favela research in Rio de Janeiro"
        - "Central American gang research since MS-13"

        ### Using the Output

        **Academic Writing:**
        1. Download the markdown file
        2. Edit and refine in your word processor
        3. Add your own analysis
        4. Verify all citations
        5. Use as literature review chapter

        **Research Proposals:**
        - Use to demonstrate knowledge of field
        - Identify gaps to justify your research
        - Build theoretical framework
        - Develop research questions

        **Teaching:**
        - Create reading guides
        - Summarize literature for students
        - Identify key debates for discussion
        - Generate topic overviews

        ### Performance & Cost

        **Generation Time:**
        - Usually 60-120 seconds
        - Depends on number of papers
        - Longer for more sections

        **API Costs:**
        - Standard review: ~$0.30-0.50
        - Comprehensive review: ~$0.60-1.00
        - Budget accordingly for multiple reviews

        ### Quality Checks

        Before using the review:
        1. ✅ Check citations are formatted correctly
        2. ✅ Verify key claims against source papers
        3. ✅ Ensure coherent narrative flow
        4. ✅ Look for any contradictions
        5. ✅ Add missing papers you know are important
        """)

        st.markdown("---")
        st.markdown("### Ready to try? Go to the 'Generate Review' tab!")


if __name__ == "__main__":
    main()
