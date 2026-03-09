"""Research review engine for providing feedback on drafts."""

import time
from typing import List, Optional

from anthropic import Anthropic, NotFoundError, AuthenticationError
from loguru import logger
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from src.config.settings import settings
from src.data.models import ResearchReviewReport, ClaimReview, ZoteroItem
from src.rag.context_builder import ContextBuilder
from src.rag.retriever import Retriever
from src.agents.prompts import (
    REVIEW_SYSTEM_PROMPT,
    REVIEW_USER_PROMPT,
    CLAIM_EXTRACTION_PROMPT,
    LITERATURE_GAP_PROMPT,
)


class ReviewEngine:
    """Engine for reviewing research drafts against literature."""

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        context_builder: Optional[ContextBuilder] = None,
    ):
        """
        Initialize review engine.

        Args:
            retriever: Retriever instance (creates new if None)
            context_builder: ContextBuilder instance (creates new if None)
        """
        self.retriever = retriever or Retriever()
        self.context_builder = context_builder or ContextBuilder()

        # Initialize Anthropic client
        self.client = Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.llm_model

        logger.info(f"Review Engine initialized with model: {self.model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_not_exception_type((NotFoundError, AuthenticationError)),
    )
    def _call_claude(self, system: str, user_message: str, max_tokens: int = 4000) -> str:
        """Call Claude API with retry logic."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.7,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )

        return response.content[0].text

    def review_research(
        self,
        draft_text: str,
        collections: Optional[List[str]] = None,
        focus_areas: Optional[List[str]] = None,
    ) -> ResearchReviewReport:
        """
        Review a research draft against the literature corpus.

        Args:
            draft_text: The research text to review
            collections: Filter by collection names
            focus_areas: Specific areas to focus review on

        Returns:
            ResearchReviewReport with detailed feedback
        """
        start_time = time.time()

        logger.info(f"Reviewing research draft ({len(draft_text)} characters)")

        # Step 1: Extract main claims from the draft
        logger.info("Extracting claims from draft...")
        claims = self._extract_claims(draft_text)

        # Step 2: For each claim, find relevant literature
        logger.info(f"Analyzing {len(claims)} claims against literature...")
        claim_reviews = []

        for claim in claims[:5]:  # Limit to top 5 claims to manage API costs
            claim_review = self._review_claim(claim, collections)
            claim_reviews.append(claim_review)

        # Step 3: Find literature similar to the overall draft
        logger.info("Finding relevant papers for the draft...")
        draft_results = self.retriever.get_similar_to_text(
            text=draft_text,
            n_results=15,
            collections=collections,
        )

        # Step 4: Build context for overall review
        context = self.context_builder.build_context(draft_results)
        sources = self.context_builder.extract_unique_sources(draft_results)

        # Step 5: Generate overall review
        logger.info("Generating comprehensive review...")
        review_prompt = REVIEW_USER_PROMPT.format(
            draft_text=draft_text,
            context=context,
        )

        overall_review = self._call_claude(
            system=REVIEW_SYSTEM_PROMPT,
            user_message=review_prompt,
            max_tokens=3000,
        )

        # Step 6: Identify literature gaps
        logger.info("Identifying literature gaps...")
        gaps_prompt = LITERATURE_GAP_PROMPT.format(
            draft_text=draft_text[:2000],  # Use excerpt to save tokens
            context=context[:3000],
        )

        gaps_text = self._call_claude(
            system=REVIEW_SYSTEM_PROMPT,
            user_message=gaps_prompt,
            max_tokens=1000,
        )

        # Parse gaps
        literature_gaps = [line.strip() for line in gaps_text.split('\n')
                          if line.strip() and (line[0].isdigit() or line.startswith('-'))]

        # Step 7: Suggest relevant papers to cite
        citation_suggestions = self._suggest_citations(draft_text, sources)

        generation_time = time.time() - start_time

        logger.info(f"Review completed in {generation_time:.2f}s")

        return ResearchReviewReport(
            original_text=draft_text,
            claim_reviews=claim_reviews,
            literature_gaps=literature_gaps,
            citation_suggestions=citation_suggestions[:10],  # Top 10
            overall_assessment=overall_review,
            generation_time=generation_time,
        )

    def _extract_claims(self, text: str) -> List[str]:
        """Extract main claims from research text."""
        prompt = CLAIM_EXTRACTION_PROMPT.format(text=text[:3000])  # Use excerpt

        claims_text = self._call_claude(
            system="You are an expert at analyzing academic writing.",
            user_message=prompt,
            max_tokens=1000,
        )

        # Parse claims (simple line-based parsing)
        claims = []
        for line in claims_text.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering
                claim = line.lstrip('0123456789.-) ')
                if claim:
                    claims.append(claim)

        logger.debug(f"Extracted {len(claims)} claims")
        return claims

    def _review_claim(
        self,
        claim: str,
        collections: Optional[List[str]] = None,
    ) -> ClaimReview:
        """Review a single claim against the literature."""
        # Find literature relevant to this claim
        results = self.retriever.retrieve(
            query=claim,
            n_results=5,
            collections=collections,
        )

        if not results:
            return ClaimReview(
                claim_text=claim,
                supporting_evidence=[],
                contradicting_evidence=[],
                sources=[],
                assessment="No relevant literature found to evaluate this claim.",
            )

        # Build context
        structured_context = self.context_builder.build_structured_context(results)

        # Analyze claim against literature
        prompt = f"""Analyze this claim against the provided literature:

Claim: "{claim}"

Relevant Literature:
"""
        for i, ctx in enumerate(structured_context, 1):
            prompt += f"\n{i}. {ctx['source']['citation']}: {ctx['text'][:200]}..."

        prompt += """

Provide:
1. Supporting evidence (if any)
2. Contradicting evidence (if any)
3. Brief assessment of the claim's validity based on literature

Be specific and cite sources."""

        assessment = self._call_claude(
            system=REVIEW_SYSTEM_PROMPT,
            user_message=prompt,
            max_tokens=800,
        )

        # Extract supporting and contradicting evidence
        # (Simple parsing - could be enhanced)
        supporting = []
        contradicting = []

        for ctx in structured_context:
            # This is simplified - in practice, would need NLP to determine support/contradict
            supporting.append(f"{ctx['source']['citation']}: {ctx['text'][:150]}...")

        sources = self.context_builder.extract_unique_sources(results)

        return ClaimReview(
            claim_text=claim,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            sources=sources,
            assessment=assessment,
        )

    def _suggest_citations(
        self,
        draft_text: str,
        available_sources: List[ZoteroItem],
    ) -> List[ZoteroItem]:
        """
        Suggest which papers should be cited in the draft.

        Args:
            draft_text: The research draft
            available_sources: Sources retrieved as relevant

        Returns:
            Ranked list of sources to cite
        """
        # Rank sources by relevance (already done by retrieval)
        # Could enhance with additional scoring

        suggestions = []
        for source in available_sources:
            # Simple heuristic: if title keywords appear in draft, boost relevance
            title_words = set(source.title.lower().split())
            draft_words = set(draft_text.lower().split())
            overlap = len(title_words & draft_words)

            if overlap > 2:  # Some keyword overlap
                suggestions.append(source)

        # If not enough suggestions, add all sources
        if len(suggestions) < 5:
            for source in available_sources:
                if source not in suggestions:
                    suggestions.append(source)

        return suggestions

    def quick_citation_check(
        self,
        draft_text: str,
        collections: Optional[List[str]] = None,
    ) -> dict:
        """
        Quick check for citation coverage in a draft.

        Args:
            draft_text: Research text
            collections: Filter collections

        Returns:
            Dict with citation metrics and suggestions
        """
        logger.info("Running quick citation check...")

        # Count existing citations (look for [Author, Year] patterns)
        import re
        citation_pattern = r'\[([A-Z][a-z]+(?:\s+(?:&|and|et al\.)\s+[A-Z][a-z]+)*,\s*\d{4})\]'
        existing_citations = re.findall(citation_pattern, draft_text)

        # Find relevant papers
        results = self.retriever.get_similar_to_text(
            text=draft_text,
            n_results=10,
            collections=collections,
        )

        sources = self.context_builder.extract_unique_sources(results)

        # Check which sources are already cited
        cited_sources = []
        uncited_sources = []

        for source in sources:
            citation_text = source.get_citation_text()
            if citation_text in str(existing_citations):
                cited_sources.append(source)
            else:
                uncited_sources.append(source)

        return {
            "total_citations": len(existing_citations),
            "unique_citations": len(set(existing_citations)),
            "relevant_papers_found": len(sources),
            "already_cited": len(cited_sources),
            "suggested_to_add": uncited_sources[:5],
            "citation_density": len(existing_citations) / max(1, len(draft_text.split())),
        }

    def find_supporting_evidence(
        self,
        claim: str,
        collections: Optional[List[str]] = None,
        n_results: int = 5,
    ) -> List[dict]:
        """
        Find evidence supporting a specific claim.

        Args:
            claim: The claim to find support for
            collections: Filter collections
            n_results: Number of results

        Returns:
            List of evidence with sources
        """
        results = self.retriever.retrieve(
            query=claim,
            n_results=n_results,
            collections=collections,
        )

        evidence_list = []
        for result in results:
            evidence_list.append({
                "text": result.chunk.text,
                "source": result.chunk.title,
                "citation": f"[{', '.join(result.chunk.authors[:2])}, {result.chunk.year}]",
                "relevance": result.similarity,
            })

        return evidence_list
