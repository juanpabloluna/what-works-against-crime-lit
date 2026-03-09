"""Literature synthesis engine for generating literature reviews."""

import time
from typing import List, Optional, Dict

from anthropic import Anthropic, NotFoundError, AuthenticationError
from loguru import logger
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from src.config.settings import settings
from src.data.models import LiteratureReview, ZoteroItem
from src.rag.context_builder import ContextBuilder
from src.rag.retriever import Retriever
from src.agents.prompts import (
    SYNTHESIS_SYSTEM_PROMPT,
    SYNTHESIS_USER_PROMPT,
    COMPARISON_PROMPT,
)


class SynthesisEngine:
    """Engine for synthesizing literature and generating reviews."""

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        context_builder: Optional[ContextBuilder] = None,
    ):
        """
        Initialize synthesis engine.

        Args:
            retriever: Retriever instance (creates new if None)
            context_builder: ContextBuilder instance (creates new if None)
        """
        self.retriever = retriever or Retriever()
        self.context_builder = context_builder or ContextBuilder()

        # Initialize Anthropic client
        self.client = Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.llm_model

        logger.info(f"Synthesis Engine initialized with model: {self.model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_not_exception_type((NotFoundError, AuthenticationError)),
    )
    def _call_claude(self, system: str, user_message: str, max_tokens: int = 4000) -> str:
        """
        Call Claude API with retry logic.

        Args:
            system: System prompt
            user_message: User message
            max_tokens: Maximum tokens to generate

        Returns:
            Claude's response text
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.7,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )

        return response.content[0].text

    def generate_literature_review(
        self,
        topic: str,
        collections: Optional[List[str]] = None,
        sections: Optional[List[str]] = None,
        n_papers: int = 30,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
    ) -> LiteratureReview:
        """
        Generate a comprehensive literature review on a topic.

        Args:
            topic: Research topic or question
            collections: Filter by collection names
            sections: Custom section names (uses defaults if None)
            n_papers: Number of papers to retrieve
            min_year: Minimum publication year
            max_year: Maximum publication year

        Returns:
            LiteratureReview object
        """
        start_time = time.time()

        logger.info(f"Generating literature review on: {topic}")

        # Default sections for academic literature review
        if sections is None:
            sections = [
                "Introduction",
                "Key Concepts and Theoretical Framework",
                "Empirical Findings",
                "Debates and Contradictions",
                "Gaps and Future Directions",
            ]

        # Retrieve comprehensive set of papers
        logger.info(f"Retrieving {n_papers} papers on the topic...")
        results = self.retriever.get_by_topic(
            topic=topic,
            n_results=n_papers,
            collections=collections,
        )

        if not results:
            logger.warning("No papers retrieved for literature review")
            return LiteratureReview(
                title=f"Literature Review: {topic}",
                topic=topic,
                introduction="No relevant literature was found on this topic.",
                sections=[],
                conclusion="",
                bibliography=[],
                generation_time=time.time() - start_time,
            )

        # Build rich context for synthesis
        context = self.context_builder.build_context(results, include_metadata=True)
        sources = self.context_builder.extract_unique_sources(results)

        logger.info(f"Retrieved {len(results)} chunks from {len(sources)} papers")

        # Generate introduction
        intro_prompt = f"""Write an introduction for a literature review on: {topic}

Context from {len(sources)} papers:
{context[:5000]}

The introduction should:
1. Define the topic and its significance
2. State the scope of this review ({len(sources)} papers)
3. Preview the main themes to be discussed

Keep it to 2-3 paragraphs."""

        introduction = self._call_claude(
            system=SYNTHESIS_SYSTEM_PROMPT,
            user_message=intro_prompt,
            max_tokens=800,
        )

        # Generate each section
        review_sections = []
        for section_title in sections:
            logger.info(f"Generating section: {section_title}")

            section_prompt = SYNTHESIS_USER_PROMPT.format(
                topic=topic,
                num_papers=len(sources),
                context=context,
                sections=section_title,
            )

            section_text = self._call_claude(
                system=SYNTHESIS_SYSTEM_PROMPT,
                user_message=section_prompt,
                max_tokens=2000,
            )

            review_sections.append((section_title, section_text))

        # Generate conclusion
        conclusion_prompt = f"""Write a conclusion for this literature review on: {topic}

Based on the following sections:
{chr(10).join([f"- {title}" for title, _ in review_sections])}

And these sources ({len(sources)} papers):
{chr(10).join([f"- {s.get_citation_text()}: {s.title}" for s in sources[:10]])}

The conclusion should:
1. Synthesize the main findings
2. Highlight key debates or unresolved questions
3. Suggest directions for future research

Keep it to 2-3 paragraphs."""

        conclusion = self._call_claude(
            system=SYNTHESIS_SYSTEM_PROMPT,
            user_message=conclusion_prompt,
            max_tokens=800,
        )

        # Format bibliography
        bibliography = [source.get_full_citation() for source in sources]

        generation_time = time.time() - start_time

        logger.info(f"Literature review generated in {generation_time:.2f}s")

        return LiteratureReview(
            title=f"Literature Review: {topic}",
            topic=topic,
            introduction=introduction,
            sections=review_sections,
            conclusion=conclusion,
            bibliography=bibliography,
            generation_time=generation_time,
        )

    def compare_papers(
        self,
        topic: str,
        paper_ids: List[int],
        aspect: Optional[str] = None,
    ) -> str:
        """
        Compare specific papers on a topic or aspect.

        Args:
            topic: Topic for comparison
            paper_ids: List of Zotero item IDs to compare
            aspect: Specific aspect to compare (methods, findings, etc.)

        Returns:
            Comparative analysis text
        """
        logger.info(f"Comparing {len(paper_ids)} papers on: {topic}")

        # Retrieve chunks from specific papers
        # For now, use general retrieval (would need enhancement to filter by item_id)
        results = self.retriever.retrieve(
            query=topic,
            n_results=20,
            diversity_ranking=False,
        )

        context = self.context_builder.build_context(results)

        comparison_aspect = aspect or topic
        prompt = COMPARISON_PROMPT.format(
            topic=comparison_aspect,
            papers_context=context,
        )

        comparison = self._call_claude(
            system=SYNTHESIS_SYSTEM_PROMPT,
            user_message=prompt,
            max_tokens=2000,
        )

        return comparison

    def identify_trends(
        self,
        topic: str,
        collections: Optional[List[str]] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
    ) -> str:
        """
        Identify trends in the literature over time.

        Args:
            topic: Research topic
            collections: Filter collections
            min_year: Start year
            max_year: End year

        Returns:
            Analysis of trends
        """
        logger.info(f"Identifying trends for: {topic}")

        results = self.retriever.retrieve(
            query=topic,
            n_results=30,
            collections=collections,
        )

        # Group by year
        papers_by_year: Dict[int, List] = {}
        for result in results:
            year = result.chunk.year
            if year and (not min_year or year >= min_year) and (not max_year or year <= max_year):
                if year not in papers_by_year:
                    papers_by_year[year] = []
                papers_by_year[year].append(result)

        # Build temporal context
        temporal_context = []
        for year in sorted(papers_by_year.keys()):
            year_papers = papers_by_year[year]
            temporal_context.append(f"\n--- Papers from {year} ---")
            for result in year_papers[:3]:  # Top 3 per year
                temporal_context.append(f"{result.chunk.title}")
                temporal_context.append(result.chunk.text[:200] + "...")

        context_str = "\n".join(temporal_context)

        prompt = f"""Analyze trends in the literature on: {topic}

Literature organized by year:
{context_str}

Identify:
1. How research focus has shifted over time
2. Emerging themes or questions
3. Methodological evolution
4. Key turning points or influential works

Provide a chronological analysis of trends."""

        trends = self._call_claude(
            system=SYNTHESIS_SYSTEM_PROMPT,
            user_message=prompt,
            max_tokens=2000,
        )

        return trends

    def find_research_gaps(
        self,
        topic: str,
        collections: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Identify gaps in the literature on a topic.

        Args:
            topic: Research topic
            collections: Filter collections

        Returns:
            List of identified gaps
        """
        logger.info(f"Finding research gaps for: {topic}")

        results = self.retriever.retrieve(
            query=topic,
            n_results=25,
            collections=collections,
        )

        context = self.context_builder.build_context(results)

        prompt = f"""Based on this literature on {topic}, identify research gaps.

Literature:
{context}

List 5-7 specific research gaps or unanswered questions. For each:
1. Describe the gap
2. Explain why it matters
3. Suggest potential research approaches

Format as a numbered list."""

        gaps_text = self._call_claude(
            system=SYNTHESIS_SYSTEM_PROMPT,
            user_message=prompt,
            max_tokens=1500,
        )

        # Parse into list (simple split by numbered items)
        gaps = []
        for line in gaps_text.split('\n'):
            if line.strip() and (line[0].isdigit() or line.startswith('-')):
                gaps.append(line.strip())

        return gaps
