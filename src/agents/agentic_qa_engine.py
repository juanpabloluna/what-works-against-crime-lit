"""
Agentic Q&A Engine using Claude with tool use.

This implements agentic RAG where Claude orchestrates its own retrieval strategy:
- Claude decides what to search for
- Claude can iteratively refine searches
- Claude determines when it has enough information
- Claude synthesizes the final answer
"""

from typing import List, Dict, Any, Optional
import json
from anthropic import Anthropic, NotFoundError, AuthenticationError
from loguru import logger
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from src.agents.agentic_tools import AgenticRAGTools
from src.config.settings import settings


class AgenticQAEngine:
    """
    Agentic Q&A engine using Claude with tool use for literature research.

    Instead of pre-retrieving chunks, this engine gives Claude tools to:
    - Search the literature corpus
    - Refine searches
    - Get paper details
    - Decide when it has enough information
    """

    def __init__(self):
        """Initialize the agentic Q&A engine."""
        self.client = Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.llm_model
        self.tools = AgenticRAGTools()
        self.max_iterations = 5  # Prevent infinite loops

        logger.info(f"Initialized AgenticQAEngine with model: {self.model}")

    def answer_question(
        self,
        question: str,
        collections: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Answer a question using agentic RAG.

        Args:
            question: The user's question
            collections: Optional collection filter
            verbose: If True, return detailed execution trace

        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Processing question with agentic RAG: {question[:100]}...")

        # System prompt for the agent
        system_prompt = self._build_system_prompt(collections)

        # Initialize conversation
        messages = [{"role": "user", "content": question}]

        # Track execution
        tool_calls = []
        iterations = 0

        # Agentic loop: Claude uses tools until it has enough information
        while iterations < self.max_iterations:
            iterations += 1

            if verbose:
                logger.info(f"Iteration {iterations}/{self.max_iterations}")

            # Call Claude with tools
            response = self._call_claude_with_tools(system_prompt, messages)

            # Check stop reason
            if response.stop_reason == "end_turn":
                # Claude has finished and provided a final answer
                final_answer = self._extract_text_from_response(response)
                logger.info(f"Agent completed in {iterations} iterations")

                return {
                    "answer": final_answer,
                    "iterations": iterations,
                    "tool_calls": tool_calls,
                    "sources": self._extract_sources_from_calls(tool_calls),
                }

            elif response.stop_reason == "tool_use":
                # Claude wants to use a tool
                # Add assistant's response to messages
                messages.append(
                    {
                        "role": "assistant",
                        "content": response.content,
                    }
                )

                # Execute tools
                tool_results = []
                for content_block in response.content:
                    if content_block.type == "tool_use":
                        tool_name = content_block.name
                        tool_input = content_block.input
                        tool_id = content_block.id

                        # Execute the tool
                        result = self.tools.execute_tool(tool_name, tool_input)

                        # Track for transparency
                        tool_calls.append(
                            {
                                "iteration": iterations,
                                "tool": tool_name,
                                "input": tool_input,
                                "result": result,
                            }
                        )

                        # Add result to send back to Claude
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": json.dumps(result, ensure_ascii=False),
                            }
                        )

                # Add tool results to messages
                messages.append({"role": "user", "content": tool_results})

            else:
                # Unexpected stop reason
                logger.warning(f"Unexpected stop reason: {response.stop_reason}")
                break

        # If we hit max iterations, return what we have
        logger.warning(f"Reached max iterations ({self.max_iterations})")
        final_text = self._extract_text_from_response(response)

        return {
            "answer": final_text or "Unable to generate answer within iteration limit.",
            "iterations": iterations,
            "tool_calls": tool_calls,
            "sources": self._extract_sources_from_calls(tool_calls),
            "warning": "Reached maximum iterations",
        }

    def _build_system_prompt(self, collections: Optional[List[str]] = None) -> str:
        """Build system prompt for the agent."""
        prompt = """You are an expert research assistant specializing in crime prevention, policing, public safety interventions, and evidence-based criminology.

You have access to a corpus of academic papers through the tools provided. Your role is to:

1. **Understand the question**: Carefully analyze what the user is asking
2. **Search strategically**: Choose the right search approach:
   - **search_literature**: For focused queries (default: 10 results, max: 50)
   - **multi_query_search**: For comprehensive coverage - searches 2-5 different queries and combines results
   - You can search multiple times with different queries
   - For broad questions, use multi_query_search to explore from multiple angles
3. **Synthesize information**: Once you have enough information, provide a comprehensive answer
4. **Cite sources**: Always mention which papers your answer draws from (author, year, title)
5. **Be scholarly**: Use precise academic language and acknowledge limitations

**Search Strategy Guidelines:**
- Narrow questions: 1-2 focused searches with search_literature
- Broad questions: Use multi_query_search with diverse queries to ensure comprehensive coverage
- Complex questions: Combine both approaches - start with multi_query_search, then refine with targeted searches

When you have gathered sufficient information from the literature, provide your final answer. You do NOT need to use tools if you already have enough information.
"""

        if collections:
            prompt += f"\n\nFocus on papers from these collections: {', '.join(collections)}"

        return prompt

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_not_exception_type((NotFoundError, AuthenticationError)),
    )
    def _call_claude_with_tools(
        self, system: str, messages: List[Dict[str, Any]]
    ) -> Any:
        """Call Claude API with tool use enabled."""
        return self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=messages,
            tools=self.tools.get_tool_definitions(),
        )

    def _extract_text_from_response(self, response: Any) -> str:
        """Extract text content from Claude's response."""
        text_parts = []
        for content_block in response.content:
            if hasattr(content_block, "text"):
                text_parts.append(content_block.text)
        return "\n\n".join(text_parts)

    def _extract_sources_from_calls(self, tool_calls: List[Dict]) -> List[Dict]:
        """Extract unique sources from tool calls."""
        sources = {}
        for call in tool_calls:
            if call["tool"] == "search_literature":
                results = call.get("result", {}).get("results", [])
                for result in results:
                    item_id = result.get("item_id")
                    if item_id and item_id not in sources:
                        sources[item_id] = {
                            "title": result.get("title"),
                            "authors": result.get("authors"),
                            "year": result.get("year"),
                        }

        return list(sources.values())
