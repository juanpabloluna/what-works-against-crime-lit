"""Text chunking for academic documents."""

import re
from typing import List, Optional

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from src.config.settings import settings
from src.data.models import DocumentChunk, ZoteroItem, PDFDocument


class DocumentChunker:
    """Chunk academic documents for retrieval."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        """
        Initialize document chunker.

        Args:
            chunk_size: Target chunk size in tokens (default from settings)
            chunk_overlap: Overlap between chunks in tokens (default from settings)
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        # Initialize tokenizer for accurate token counting
        self.encoding = tiktoken.get_encoding("cl100k_base")

        # Create text splitter with separators optimized for academic text
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * 4,  # Approximate chars (4 chars ≈ 1 token)
            chunk_overlap=self.chunk_overlap * 4,
            length_function=self._count_tokens,
            separators=[
                "\n\n\n",  # Multiple line breaks (section boundaries)
                "\n\n",  # Paragraph breaks
                "\n",  # Line breaks
                ". ",  # Sentence breaks
                "! ",
                "? ",
                "; ",
                ", ",  # Clause breaks
                " ",  # Word breaks
                "",  # Character breaks (last resort)
            ],
        )

        logger.info(
            f"Initialized DocumentChunker: chunk_size={self.chunk_size}, overlap={self.chunk_overlap}"
        )

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.encoding.encode(text))

    def chunk_document(
        self, pdf_doc: PDFDocument, item: ZoteroItem
    ) -> List[DocumentChunk]:
        """
        Chunk a PDF document into retrievable pieces.

        Args:
            pdf_doc: Extracted PDF document
            item: Zotero metadata for the document

        Returns:
            List of document chunks
        """
        # Try to detect sections first
        sections = self._detect_sections(pdf_doc.full_text)

        chunks = []
        chunk_index = 0

        if sections:
            # Chunk within detected sections
            for section_name, section_text in sections.items():
                section_chunks = self._chunk_text(section_text)

                for chunk_text in section_chunks:
                    chunk = self._create_chunk(
                        text=chunk_text,
                        item=item,
                        chunk_index=chunk_index,
                        section=section_name,
                    )
                    chunks.append(chunk)
                    chunk_index += 1
        else:
            # No sections detected, chunk the whole document
            text_chunks = self._chunk_text(pdf_doc.full_text)

            for chunk_text in text_chunks:
                chunk = self._create_chunk(
                    text=chunk_text,
                    item=item,
                    chunk_index=chunk_index,
                    section=None,
                )
                chunks.append(chunk)
                chunk_index += 1

        # Update total chunks count
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        logger.debug(
            f"Created {len(chunks)} chunks from document: {item.title[:50]}..."
        )
        return chunks

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks using the text splitter."""
        if not text.strip():
            return []

        chunks = self.text_splitter.split_text(text)
        return chunks

    def _create_chunk(
        self,
        text: str,
        item: ZoteroItem,
        chunk_index: int,
        section: Optional[str] = None,
    ) -> DocumentChunk:
        """Create a DocumentChunk object."""
        # Use zotero_key + pdf hash to ensure uniqueness (some items have multiple PDFs)
        import hashlib
        pdf_hash = hashlib.md5(str(item.pdf_path).encode()).hexdigest()[:8]
        chunk_id = f"{item.zotero_key}_{pdf_hash}_chunk_{chunk_index}"

        return DocumentChunk(
            chunk_id=chunk_id,
            text=text,
            item_id=item.item_id,
            zotero_key=item.zotero_key,
            title=item.title,
            authors=item.authors,
            year=item.year,
            collections=item.collections,
            tags=item.tags,
            section=section,
            chunk_index=chunk_index,
            total_chunks=0,  # Will be updated after all chunks are created
            pdf_path=item.pdf_path,
        )

    def _detect_sections(self, text: str) -> dict[str, str]:
        """
        Detect major sections in academic paper.

        Returns dict mapping section names to their content.
        """
        # Common academic paper section patterns
        section_patterns = [
            (r"\bAbstract\b", "abstract"),
            (r"\b(1\.?\s+)?Introduction\b", "introduction"),
            (r"\b(Literature Review|Related Work|Theoretical Framework|Background)\b", "literature_review"),
            (r"\b(Methodology|Methods|Research Design|Data and Methods)\b", "methodology"),
            (r"\b(Results|Findings|Empirical Results)\b", "results"),
            (r"\b(Discussion|Analysis|Discussion and Analysis)\b", "discussion"),
            (r"\b(Conclusion|Conclusions|Concluding Remarks)\b", "conclusion"),
            (r"\b(References|Bibliography|Works Cited)\b", "references"),
        ]

        # Find all section headers
        section_matches = []
        for pattern, section_name in section_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                # Check if this looks like a section header (at start of line or preceded by whitespace)
                start = match.start()
                if start == 0 or text[start - 1] in "\n\r":
                    section_matches.append((start, section_name, match.group()))

        if not section_matches:
            return {}

        # Sort by position
        section_matches.sort(key=lambda x: x[0])

        # Extract text between sections
        sections = {}
        for i, (pos, name, header) in enumerate(section_matches):
            # Find end of this section (start of next section or end of text)
            if i < len(section_matches) - 1:
                end_pos = section_matches[i + 1][0]
            else:
                end_pos = len(text)

            # Extract section content (skip the header itself)
            content = text[pos + len(header) : end_pos].strip()

            # Only include if we got meaningful content
            if content and len(content) > 50:
                sections[name] = content

        return sections

    def estimate_chunks(self, text: str) -> int:
        """Estimate how many chunks will be created from text."""
        token_count = self._count_tokens(text)
        effective_chunk_size = self.chunk_size - self.chunk_overlap
        return max(1, (token_count + effective_chunk_size - 1) // effective_chunk_size)

    def validate_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """
        Validate that chunks meet quality criteria.

        Returns True if chunks are valid, False otherwise.
        """
        if not chunks:
            logger.warning("No chunks created")
            return False

        for chunk in chunks:
            # Check minimum length
            if len(chunk.text.strip()) < 50:
                logger.warning(
                    f"Chunk {chunk.chunk_id} is too short ({len(chunk.text)} chars)"
                )
                return False

            # Check token count (warning only, don't fail validation)
            tokens = self._count_tokens(chunk.text)
            if tokens > self.chunk_size * 2:  # Allow 100% overage for warning
                logger.warning(
                    f"Chunk {chunk.chunk_id} exceeds recommended token limit ({tokens} > {self.chunk_size * 2})"
                )
                # Don't return False - just warn

        return True
