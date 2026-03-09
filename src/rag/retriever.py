"""Retrieval engine for querying the vector database."""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

from loguru import logger

from src.config.settings import settings
from src.data.models import RetrievalResult
from src.embeddings.embedding_service import EmbeddingService
from src.embeddings.vector_store import VectorStore
from src.rag.context_builder import ContextBuilder


class Retriever:
    """High-level retrieval interface."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_service: Optional[EmbeddingService] = None,
        context_builder: Optional[ContextBuilder] = None,
    ):
        """
        Initialize retriever.

        Args:
            vector_store: VectorStore instance (creates new if None)
            embedding_service: EmbeddingService instance (creates new if None)
            context_builder: ContextBuilder instance (creates new if None)
        """
        self.vector_store = vector_store or VectorStore()
        self.embedding_service = embedding_service or EmbeddingService()
        self.context_builder = context_builder or ContextBuilder()
        self._author_lookup: Dict[str, str] = {}  # lowercase → canonical
        self._load_author_index()

        logger.info(
            f"Retriever initialized (author index: {len(self._author_lookup)} names)"
        )

    def _load_author_index(self) -> None:
        """Load known author last names from papers_metadata.json."""
        # Try multiple paths for robustness across local/cloud
        candidates = [
            Path(__file__).parent.parent.parent / "data" / "papers_metadata.json",
            Path.cwd() / "data" / "papers_metadata.json",
            settings.chromadb_path.parent / "papers_metadata.json",
            Path("/mount/src/criminal-governance-lit/data/papers_metadata.json"),
        ]
        metadata_path = None
        for p in candidates:
            if p.exists():
                metadata_path = p
                break

        if not metadata_path:
            logger.warning(
                f"papers_metadata.json not found. Tried: {[str(p) for p in candidates]}"
            )
            return

        try:
            with open(metadata_path) as f:
                records = json.load(f)
            for r in records:
                for author in r.get("authors", []):
                    if "," in author:
                        name = author.split(",")[0].strip()
                    else:
                        parts = author.split()
                        name = parts[-1].strip() if parts else ""
                    if len(name) > 2:
                        self._author_lookup[name.lower()] = name
            logger.info(
                f"Loaded {len(self._author_lookup)} author last names from {metadata_path}"
            )
        except Exception as e:
            logger.warning(f"Could not load author index: {e}")

    # Common English words that collide with author last names in the corpus
    _AUTHOR_STOPWORDS = frozenset({
        "main", "fair", "green", "young", "long", "black", "white", "cross",
        "strong", "short", "rich", "price", "key", "grant", "mark", "frank",
        "rose", "stone", "wolf", "hunt", "page", "ward", "lane", "lee",
        "ray", "wade", "reed", "cook", "duke", "ford", "clay", "cole",
        "bond", "camp", "marsh", "wells", "miles", "mason", "foster",
        "taylor", "morris", "moore", "lynch", "levy", "stein", "neal",
        "the", "and", "are", "for", "not", "but", "can", "had", "has",
        "his", "how", "its", "may", "new", "now", "old", "way", "who",
        "any", "few", "use", "per", "yet", "war",
    })

    def _detect_author_names(self, query: str) -> List[str]:
        """Detect author last names in the query (case-insensitive)."""
        words = re.findall(r"\b[a-zA-Z]{3,}\b", query)
        detected = []
        seen = set()
        for word in words:
            key = word.lower()
            if key in self._AUTHOR_STOPWORDS:
                continue
            if key in self._author_lookup and key not in seen:
                detected.append(self._author_lookup[key])  # canonical form
                seen.add(key)
        if detected:
            logger.info(f"Detected author names in query: {detected}")
        return detected

    def retrieve(
        self,
        query: str,
        n_results: int = None,
        collections: Optional[List[str]] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        diversity_ranking: bool = True,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks for a query.

        Automatically detects author names and uses hybrid retrieval
        (semantic + author-focused) when authors are mentioned.

        Args:
            query: Query text
            n_results: Number of results to return (default from settings)
            collections: Filter by collection names
            min_year: Minimum publication year
            max_year: Maximum publication year
            diversity_ranking: Whether to rank for diversity across sources

        Returns:
            List of retrieval results
        """
        n_results = n_results or settings.top_k

        logger.info(f"Retrieving documents for query: {query[:100]}...")

        # Detect author names for hybrid retrieval
        author_names = self._detect_author_names(query)

        # Query vector store (hybrid if authors detected)
        results = self.vector_store.query_by_text(
            query_text=query,
            embedding_service=self.embedding_service,
            n_results=n_results * 2 if diversity_ranking else n_results,
            collections=collections,
            min_year=min_year,
            max_year=max_year,
            author_names=author_names if author_names else None,
        )

        if not results:
            logger.warning("No results found for query")
            return []

        # Apply diversity ranking if requested
        if diversity_ranking and len(results) > n_results:
            results = self.context_builder.rank_by_diversity(results, top_k=n_results)

        # Deduplicate
        results = self.context_builder.deduplicate_chunks(results)

        logger.info(f"Retrieved {len(results)} relevant chunks")
        return results

    def retrieve_with_context(
        self,
        query: str,
        n_results: int = None,
        collections: Optional[List[str]] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
    ) -> tuple[List[RetrievalResult], str]:
        """
        Retrieve relevant chunks and build context string.

        Returns tuple of (results, context_string)
        """
        results = self.retrieve(
            query=query,
            n_results=n_results,
            collections=collections,
            min_year=min_year,
            max_year=max_year,
        )

        context = self.context_builder.build_context(results)

        return results, context

    def get_similar_to_text(
        self,
        text: str,
        n_results: int = 5,
        collections: Optional[List[str]] = None,
    ) -> List[RetrievalResult]:
        """
        Find chunks similar to a given text.

        Useful for finding related literature to a draft.
        """
        logger.info(f"Finding similar documents to provided text ({len(text)} chars)")

        return self.retrieve(
            query=text,
            n_results=n_results,
            collections=collections,
        )

    def get_by_topic(
        self,
        topic: str,
        n_results: int = 20,
        collections: Optional[List[str]] = None,
    ) -> List[RetrievalResult]:
        """
        Get comprehensive set of documents on a topic.

        Uses higher n_results for literature review synthesis.
        """
        logger.info(f"Retrieving comprehensive results for topic: {topic}")

        return self.retrieve(
            query=topic,
            n_results=n_results,
            collections=collections,
            diversity_ranking=True,
        )

    def multi_query_retrieve(
        self,
        queries: List[str],
        n_results_per_query: int = 5,
        collections: Optional[List[str]] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve using multiple related queries and merge results.

        Useful for comprehensive topic coverage.
        """
        logger.info(f"Multi-query retrieval with {len(queries)} queries")

        all_results = []
        seen_chunk_ids = set()

        for query in queries:
            results = self.retrieve(
                query=query,
                n_results=n_results_per_query,
                collections=collections,
                diversity_ranking=False,  # We'll do global diversity ranking
            )

            # Add unique results
            for result in results:
                if result.chunk.chunk_id not in seen_chunk_ids:
                    all_results.append(result)
                    seen_chunk_ids.add(result.chunk.chunk_id)

        # Re-rank by relevance (using average similarity if needed)
        all_results.sort(key=lambda r: r.similarity, reverse=True)

        logger.info(f"Multi-query retrieved {len(all_results)} unique chunks")
        return all_results

    def get_stats(self) -> dict:
        """Get retriever statistics."""
        vs_stats = self.vector_store.get_stats()
        emb_info = self.embedding_service.get_model_info()

        return {
            "vector_store": vs_stats,
            "embedding_model": emb_info,
        }
