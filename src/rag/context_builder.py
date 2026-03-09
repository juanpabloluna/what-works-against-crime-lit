"""Context builder for assembling retrieved chunks into LLM context."""

from typing import List, Dict, Set
from collections import defaultdict

from loguru import logger

from src.data.models import RetrievalResult, ZoteroItem


class ContextBuilder:
    """Build context from retrieved chunks for LLM."""

    def __init__(self, max_context_length: int = 20000):
        """
        Initialize context builder.

        Args:
            max_context_length: Maximum context length in characters
        """
        self.max_context_length = max_context_length

    def build_context(
        self, results: List[RetrievalResult], include_metadata: bool = True
    ) -> str:
        """
        Build context string from retrieval results.

        Args:
            results: List of retrieval results
            include_metadata: Whether to include source metadata

        Returns:
            Formatted context string
        """
        if not results:
            return ""

        context_parts = []
        current_length = 0

        # Group chunks by source document to avoid repetition
        chunks_by_doc = defaultdict(list)
        for result in results:
            doc_key = (result.chunk.item_id, result.chunk.title)
            chunks_by_doc[doc_key].append(result)

        # Build context from each document
        for (item_id, title), doc_results in chunks_by_doc.items():
            # Sort chunks by index for coherent reading
            doc_results.sort(key=lambda r: r.chunk.chunk_index)

            # Get first chunk for metadata
            first_chunk = doc_results[0].chunk

            # Add source header
            if include_metadata:
                authors = ", ".join(first_chunk.authors[:2])
                if len(first_chunk.authors) > 2:
                    authors += " et al."
                year = first_chunk.year or "n.d."
                citation = f"[{authors}, {year}]"

                header = f"\n--- Source: {title} {citation} ---\n"
                context_parts.append(header)
                current_length += len(header)

            # Add chunk texts
            for result in doc_results:
                chunk_text = result.chunk.text.strip()

                # Check if adding this chunk would exceed limit
                if current_length + len(chunk_text) > self.max_context_length:
                    logger.debug(
                        f"Context length limit reached ({current_length} chars)"
                    )
                    break

                # Add section label if available
                if result.chunk.section and include_metadata:
                    section_label = f"\n[{result.chunk.section.upper()}]\n"
                    context_parts.append(section_label)
                    current_length += len(section_label)

                context_parts.append(chunk_text)
                context_parts.append("\n\n")
                current_length += len(chunk_text) + 2

            if current_length >= self.max_context_length:
                break

        context = "".join(context_parts)
        logger.debug(f"Built context with {current_length} characters from {len(chunks_by_doc)} documents")
        return context

    def build_structured_context(self, results: List[RetrievalResult]) -> List[Dict]:
        """
        Build structured context with explicit source attribution.

        Returns list of dicts with 'source', 'text', 'relevance' fields.
        """
        if not results:
            return []

        structured_context = []

        for result in results:
            chunk = result.chunk

            # Build citation
            authors = ", ".join(chunk.authors[:2])
            if len(chunk.authors) > 2:
                authors += " et al."
            year = chunk.year or "n.d."
            citation = f"{authors} ({year})"

            structured_context.append({
                "source": {
                    "title": chunk.title,
                    "authors": chunk.authors,
                    "year": chunk.year,
                    "citation": citation,
                    "zotero_key": chunk.zotero_key,
                },
                "text": chunk.text,
                "section": chunk.section,
                "relevance_score": result.similarity,
                "chunk_index": chunk.chunk_index,
            })

        return structured_context

    def extract_unique_sources(self, results: List[RetrievalResult]) -> List[ZoteroItem]:
        """
        Extract unique source documents from retrieval results.

        Args:
            results: List of retrieval results

        Returns:
            List of unique ZoteroItem objects
        """
        seen_items: Set[int] = set()
        unique_sources = []

        for result in results:
            if result.chunk.item_id not in seen_items:
                seen_items.add(result.chunk.item_id)

                # Create ZoteroItem from chunk metadata
                source = ZoteroItem(
                    item_id=result.chunk.item_id,
                    zotero_key=result.chunk.zotero_key,
                    title=result.chunk.title,
                    authors=result.chunk.authors,
                    year=result.chunk.year,
                    collections=result.chunk.collections,
                    tags=result.chunk.tags,
                    pdf_path=result.chunk.pdf_path,
                )
                unique_sources.append(source)

        logger.debug(f"Extracted {len(unique_sources)} unique sources from {len(results)} chunks")
        return unique_sources

    def format_sources_bibliography(self, sources: List[ZoteroItem]) -> str:
        """
        Format sources as a bibliography.

        Args:
            sources: List of ZoteroItem objects

        Returns:
            Formatted bibliography string
        """
        if not sources:
            return ""

        bibliography_lines = ["## Sources\n"]

        for i, source in enumerate(sources, 1):
            citation = source.get_full_citation()
            bibliography_lines.append(f"{i}. {citation}")

        return "\n".join(bibliography_lines)

    def deduplicate_chunks(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Remove duplicate or highly overlapping chunks.

        Args:
            results: List of retrieval results

        Returns:
            Deduplicated list of results
        """
        if not results:
            return []

        deduplicated = []
        seen_texts = set()

        for result in results:
            # Use first 200 chars as fingerprint
            fingerprint = result.chunk.text[:200].strip()

            if fingerprint not in seen_texts:
                seen_texts.add(fingerprint)
                deduplicated.append(result)

        if len(deduplicated) < len(results):
            logger.debug(f"Deduplicated {len(results)} chunks to {len(deduplicated)}")

        return deduplicated

    def rank_by_diversity(
        self, results: List[RetrievalResult], top_k: int = 10
    ) -> List[RetrievalResult]:
        """
        Select diverse set of chunks (different sources, sections).

        Args:
            results: List of retrieval results
            top_k: Number of results to return

        Returns:
            Diverse subset of results
        """
        if len(results) <= top_k:
            return results

        selected = []
        seen_sources = set()
        seen_sections = set()

        # First pass: one chunk per source
        for result in results:
            source_key = result.chunk.item_id
            if source_key not in seen_sources:
                selected.append(result)
                seen_sources.add(source_key)
                if result.chunk.section:
                    seen_sections.add((source_key, result.chunk.section))

            if len(selected) >= top_k:
                break

        # Second pass: add more chunks if needed, prefer different sections
        if len(selected) < top_k:
            for result in results:
                if result in selected:
                    continue

                section_key = (result.chunk.item_id, result.chunk.section)
                if section_key not in seen_sections:
                    selected.append(result)
                    seen_sections.add(section_key)

                if len(selected) >= top_k:
                    break

        # Third pass: fill remaining slots by relevance
        if len(selected) < top_k:
            for result in results:
                if result not in selected:
                    selected.append(result)
                if len(selected) >= top_k:
                    break

        logger.debug(f"Selected {len(selected)} diverse chunks from {len(results)}")
        return selected
