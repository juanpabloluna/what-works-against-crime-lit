"""Question answering engine using Claude API."""

import time
from typing import List, Optional

from anthropic import Anthropic, NotFoundError, AuthenticationError
from loguru import logger
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from src.config.settings import settings
from src.data.models import Answer, ZoteroItem
from src.rag.context_builder import ContextBuilder
from src.rag.retriever import Retriever
from src.agents.prompts import QA_SYSTEM_PROMPT, QA_USER_PROMPT


class QAEngine:
    """Question answering engine with retrieval and generation."""

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        context_builder: Optional[ContextBuilder] = None,
    ):
        """
        Initialize QA engine.

        Args:
            retriever: Retriever instance (creates new if None)
            context_builder: ContextBuilder instance (creates new if None)
        """
        self.retriever = retriever or Retriever()
        self.context_builder = context_builder or ContextBuilder()

        # Initialize Anthropic client
        self.client = Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.llm_model

        logger.info(f"QA Engine initialized with model: {self.model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_not_exception_type((NotFoundError, AuthenticationError)),
    )
    def _call_claude(self, system: str, user_message: str) -> str:
        """
        Call Claude API with retry logic.

        Args:
            system: System prompt
            user_message: User message

        Returns:
            Claude's response text
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )

        return response.content[0].text

    def answer_question(
        self,
        question: str,
        collections: Optional[List[str]] = None,
        n_results: int = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
    ) -> Answer:
        """
        Answer a question using retrieval and generation.

        Args:
            question: User's question
            collections: Filter by collection names
            n_results: Number of chunks to retrieve
            min_year: Minimum publication year
            max_year: Maximum publication year

        Returns:
            Answer object with response and sources
        """
        start_time = time.time()

        logger.info(f"Answering question: {question[:100]}...")

        # Retrieve relevant chunks
        results, context = self.retriever.retrieve_with_context(
            query=question,
            n_results=n_results,
            collections=collections,
            min_year=min_year,
            max_year=max_year,
        )

        if not results:
            logger.warning("No relevant documents found")
            return Answer(
                question=question,
                answer="I couldn't find relevant information in the literature to answer this question. Please try rephrasing your question or adjusting the collection filters.",
                sources=[],
                chunks_used=0,
                generation_time=time.time() - start_time,
            )

        # Build prompt
        user_prompt = QA_USER_PROMPT.format(context=context, question=question)

        # Generate answer
        logger.debug(f"Calling Claude with {len(context)} chars of context")
        answer_text = self._call_claude(
            system=QA_SYSTEM_PROMPT, user_message=user_prompt
        )

        # Extract unique sources
        sources = self.context_builder.extract_unique_sources(results)

        generation_time = time.time() - start_time

        logger.info(
            f"Answer generated in {generation_time:.2f}s using {len(results)} chunks from {len(sources)} sources"
        )

        return Answer(
            question=question,
            answer=answer_text,
            sources=sources,
            chunks_used=len(results),
            generation_time=generation_time,
        )

    def answer_with_conversation_history(
        self,
        question: str,
        conversation_history: List[dict],
        collections: Optional[List[str]] = None,
        n_results: int = None,
    ) -> Answer:
        """
        Answer a question with conversation context.

        Useful for follow-up questions.

        Args:
            question: Current question
            conversation_history: List of previous Q&A pairs [{"question": ..., "answer": ...}]
            collections: Filter collections
            n_results: Number of results

        Returns:
            Answer object
        """
        # Build context-aware query by combining with previous questions
        if conversation_history:
            prev_questions = [qa["question"] for qa in conversation_history[-2:]]
            # Use current question as primary, but retrieve considers history
            enhanced_query = f"{' '.join(prev_questions)} {question}"
        else:
            enhanced_query = question

        # Retrieve with enhanced query
        results, context = self.retriever.retrieve_with_context(
            query=enhanced_query, n_results=n_results, collections=collections
        )

        if not results:
            logger.warning("No relevant documents found")
            return Answer(
                question=question,
                answer="I couldn't find relevant information to answer this follow-up question.",
                sources=[],
                chunks_used=0,
                generation_time=0,
            )

        # Build prompt with conversation history
        history_text = ""
        if conversation_history:
            history_text = "\n\nPrevious conversation:\n"
            for qa in conversation_history[-2:]:
                history_text += f"Q: {qa['question']}\nA: {qa['answer']}\n\n"

        user_prompt = QA_USER_PROMPT.format(context=context, question=question)
        user_prompt = history_text + user_prompt

        # Generate answer
        start_time = time.time()
        answer_text = self._call_claude(
            system=QA_SYSTEM_PROMPT, user_message=user_prompt
        )

        sources = self.context_builder.extract_unique_sources(results)

        return Answer(
            question=question,
            answer=answer_text,
            sources=sources,
            chunks_used=len(results),
            generation_time=time.time() - start_time,
        )

    def compare_sources(
        self, question: str, source_ids: List[int]
    ) -> str:
        """
        Compare specific sources on a topic.

        Args:
            question: Comparison question
            source_ids: List of Zotero item IDs to compare

        Returns:
            Comparative analysis text
        """
        # Retrieve chunks from specific sources
        all_results = []
        for item_id in source_ids:
            # Get chunks for this specific item
            # This would require a new method in vector_store
            pass  # TODO: Implement if needed

        # For now, use regular retrieval
        return self.answer_question(question).answer

    def explain_answer(self, answer: Answer) -> str:
        """
        Generate explanation of how the answer was derived.

        Args:
            answer: Answer object

        Returns:
            Explanation text
        """
        explanation_parts = []

        explanation_parts.append(
            f"This answer was generated using {answer.chunks_used} relevant passages "
            f"from {len(answer.sources)} different sources in the literature."
        )

        explanation_parts.append("\n\nSources consulted:")
        for i, source in enumerate(answer.sources[:5], 1):
            citation = source.get_citation_text()
            explanation_parts.append(f"{i}. {source.title} - {citation}")

        if len(answer.sources) > 5:
            explanation_parts.append(f"... and {len(answer.sources) - 5} more sources")

        explanation_parts.append(
            f"\n\nGeneration time: {answer.generation_time:.2f} seconds"
        )

        return "\n".join(explanation_parts)
