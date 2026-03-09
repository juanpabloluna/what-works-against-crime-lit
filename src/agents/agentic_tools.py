"""
Agentic RAG Tools - Tools that Claude can use to search and explore literature.

In agentic RAG, Claude has access to tools and decides:
- What to search for
- How many results to retrieve
- Whether to refine the search
- When it has enough information to answer
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

from src.embeddings.embedding_service import EmbeddingService
from src.embeddings.vector_store import VectorStore
from src.data.models import RetrievalResult
from src.config.settings import settings


class AgenticRAGTools:
    """Tools for agentic literature exploration."""

    def __init__(self):
        """Initialize the tools with access to vector store and metadata."""
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self._metadata = self._load_metadata()
        logger.info("Initialized AgenticRAGTools")

    def _load_metadata(self) -> Dict[int, Dict]:
        """Load papers_metadata.json for paper details lookup."""
        candidates = [
            Path(__file__).parent.parent.parent / "data" / "papers_metadata.json",
            Path.cwd() / "data" / "papers_metadata.json",
            settings.chromadb_path.parent / "papers_metadata.json",
        ]
        for p in candidates:
            if p.exists():
                try:
                    with open(p) as f:
                        records = json.load(f)
                    lookup = {r["item_id"]: r for r in records}
                    logger.info(f"Loaded {len(lookup)} paper metadata records from {p}")
                    return lookup
                except Exception as e:
                    logger.warning(f"Failed to load metadata from {p}: {e}")
        logger.warning("papers_metadata.json not found — paper details disabled")
        return {}

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get tool definitions in Anthropic's tool use format."""
        return [
            {
                "name": "search_literature",
                "description": "Semantic search across the literature corpus on crime prevention, policing, and public safety interventions. Returns relevant passages from academic papers.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query - describe what information you're looking for",
                        },
                        "n_results": {
                            "type": "integer",
                            "description": "Number of results to return (default: 10, max: 50)",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_paper_details",
                "description": "Get detailed metadata about a specific paper by its ID.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "item_id": {
                            "type": "integer",
                            "description": "The item_id of the paper from search results",
                        }
                    },
                    "required": ["item_id"],
                },
            },
            {
                "name": "get_papers_by_year_range",
                "description": "Search for papers published within a specific year range.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "start_year": {"type": "integer", "description": "Start year (inclusive)"},
                        "end_year": {"type": "integer", "description": "End year (inclusive)"},
                        "n_results": {"type": "integer", "description": "Number of results", "default": 5},
                    },
                    "required": ["query", "start_year", "end_year"],
                },
            },
            {
                "name": "multi_query_search",
                "description": "Perform multiple searches with different queries to get comprehensive coverage. Automatically deduplicates results.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of 2-5 different search queries",
                            "minItems": 2,
                            "maxItems": 5,
                        },
                        "n_results_per_query": {
                            "type": "integer",
                            "description": "Results per query (default: 10)",
                            "default": 10,
                        },
                    },
                    "required": ["queries"],
                },
            },
        ]

    def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call from Claude."""
        logger.info(f"Executing tool: {tool_name} with input: {tool_input}")

        if tool_name == "search_literature":
            return self._search_literature(**tool_input)
        elif tool_name == "get_paper_details":
            return self._get_paper_details(**tool_input)
        elif tool_name == "get_papers_by_year_range":
            return self._get_papers_by_year_range(**tool_input)
        elif tool_name == "multi_query_search":
            return self._multi_query_search(**tool_input)
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    def _search_literature(self, query: str, n_results: int = 10, **kwargs) -> Dict[str, Any]:
        """Search the literature corpus."""
        try:
            n_results = min(max(n_results, 1), 50)
            results = self.vector_store.query_by_text(
                query_text=query,
                embedding_service=self.embedding_service,
                n_results=n_results,
            )
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "text": result.chunk.text,
                    "title": result.chunk.title,
                    "authors": ", ".join(result.chunk.authors),
                    "year": result.chunk.year,
                    "item_id": result.chunk.item_id,
                    "relevance_score": result.similarity,
                    "section": result.chunk.section,
                })
            return {"results": formatted_results, "total_found": len(formatted_results), "query": query}
        except Exception as e:
            logger.error(f"Error in search_literature: {e}", exc_info=True)
            return {"error": str(e)}

    def _get_paper_details(self, item_id: int) -> Dict[str, Any]:
        """Get detailed metadata about a specific paper from papers_metadata.json."""
        if not self._metadata:
            return {"error": "Paper metadata not available"}
        record = self._metadata.get(item_id)
        if not record:
            return {"error": f"Paper with item_id {item_id} not found"}
        return record

    def _get_papers_by_year_range(self, query: str, start_year: int, end_year: int, n_results: int = 5) -> Dict[str, Any]:
        """Search papers within a year range."""
        try:
            results = self.vector_store.query_by_text(
                query_text=query, embedding_service=self.embedding_service,
                n_results=n_results, min_year=start_year, max_year=end_year,
            )
            formatted_results = []
            for result in results[:n_results]:
                formatted_results.append({
                    "text": result.chunk.text, "title": result.chunk.title,
                    "authors": ", ".join(result.chunk.authors), "year": result.chunk.year,
                    "item_id": result.chunk.item_id, "relevance_score": result.similarity,
                })
            return {"results": formatted_results, "total_found": len(formatted_results),
                    "query": query, "year_range": f"{start_year}-{end_year}"}
        except Exception as e:
            logger.error(f"Error in get_papers_by_year_range: {e}", exc_info=True)
            return {"error": str(e)}

    def _multi_query_search(self, queries: List[str], n_results_per_query: int = 10) -> Dict[str, Any]:
        """Perform multiple searches and combine results."""
        try:
            all_results = []
            seen_chunk_ids = set()
            for query in queries:
                results = self.vector_store.query_by_text(
                    query_text=query, embedding_service=self.embedding_service,
                    n_results=n_results_per_query,
                )
                for result in results:
                    if result.chunk.chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(result.chunk.chunk_id)
                        all_results.append(result)

            formatted_results = []
            for result in all_results:
                formatted_results.append({
                    "text": result.chunk.text, "title": result.chunk.title,
                    "authors": ", ".join(result.chunk.authors), "year": result.chunk.year,
                    "item_id": result.chunk.item_id, "relevance_score": result.similarity,
                    "section": result.chunk.section,
                })
            return {
                "results": formatted_results, "total_found": len(formatted_results),
                "queries_executed": queries, "unique_chunks": len(seen_chunk_ids),
                "summary": f"Searched {len(queries)} queries, found {len(formatted_results)} unique chunks from {len(set(r['item_id'] for r in formatted_results))} papers",
            }
        except Exception as e:
            logger.error(f"Error in multi_query_search: {e}", exc_info=True)
            return {"error": str(e)}
