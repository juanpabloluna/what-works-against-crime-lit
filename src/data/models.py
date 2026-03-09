"""Data models for the Literature Expert Agent."""

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, field_validator


class ZoteroItem(BaseModel):
    """Metadata for a Zotero library item."""

    item_id: int = Field(..., description="Zotero item ID")
    zotero_key: str = Field(..., description="Zotero unique key")
    title: str = Field(..., description="Item title")
    authors: List[str] = Field(default_factory=list, description="List of authors")
    year: Optional[int] = Field(None, description="Publication year")
    abstract: Optional[str] = Field(None, description="Abstract")
    publication: Optional[str] = Field(None, description="Publication/Journal name")
    doi: Optional[str] = Field(None, description="DOI")
    url: Optional[str] = Field(None, description="URL")
    collections: List[str] = Field(
        default_factory=list, description="Zotero collections"
    )
    tags: List[str] = Field(default_factory=list, description="Tags")
    pdf_path: Optional[str] = Field(None, description="Path to PDF file")
    date_added: Optional[datetime] = Field(None, description="Date added to Zotero")
    date_modified: Optional[datetime] = Field(
        None, description="Date last modified"
    )

    @field_validator("year", mode="before")
    @classmethod
    def parse_year(cls, v):
        """Extract year from various date formats."""
        if v is None:
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            # Try to extract year from string
            import re

            match = re.search(r"\b(19|20)\d{2}\b", v)
            if match:
                return int(match.group())
        return None

    def get_citation_text(self) -> str:
        """Generate citation text for this item."""
        author_text = ", ".join(self.authors[:2])
        if len(self.authors) > 2:
            author_text += " et al."
        year_text = str(self.year) if self.year else "n.d."
        return f"{author_text} ({year_text})"

    def get_full_citation(self) -> str:
        """Generate full bibliographic citation."""
        parts = []

        # Authors
        if self.authors:
            author_text = ", ".join(self.authors)
            parts.append(author_text)

        # Year
        year_text = f"({self.year})" if self.year else "(n.d.)"
        parts.append(year_text)

        # Title
        parts.append(f"\"{self.title}\"")

        # Publication
        if self.publication:
            parts.append(f"*{self.publication}*")

        # DOI or URL
        if self.doi:
            parts.append(f"DOI: {self.doi}")
        elif self.url:
            parts.append(f"URL: {self.url}")

        return ". ".join(parts) + "."


class PDFPage(BaseModel):
    """Represents a page in a PDF document."""

    page_number: int = Field(..., description="Page number (1-indexed)")
    text: str = Field(..., description="Extracted text content")
    char_count: int = Field(..., description="Number of characters")


class PDFDocument(BaseModel):
    """Extracted PDF document with metadata."""

    pdf_path: str = Field(..., description="Path to PDF file")
    full_text: str = Field(..., description="Full document text")
    pages: List[PDFPage] = Field(..., description="Individual pages")
    total_pages: int = Field(..., description="Total number of pages")
    total_chars: int = Field(..., description="Total characters")
    extraction_date: datetime = Field(
        default_factory=datetime.now, description="When extraction occurred"
    )
    success: bool = Field(default=True, description="Whether extraction succeeded")
    error_message: Optional[str] = Field(
        None, description="Error message if extraction failed"
    )


class DocumentChunk(BaseModel):
    """A chunk of text from a document."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    text: str = Field(..., description="Chunk text content")
    item_id: int = Field(..., description="Source Zotero item ID")
    zotero_key: str = Field(..., description="Source Zotero key")
    title: str = Field(..., description="Source document title")
    authors: List[str] = Field(default_factory=list, description="Authors")
    year: Optional[int] = Field(None, description="Publication year")
    collections: List[str] = Field(default_factory=list, description="Collections")
    tags: List[str] = Field(default_factory=list, description="Tags")
    section: Optional[str] = Field(None, description="Document section")
    chunk_index: int = Field(..., description="Index of this chunk in document")
    total_chunks: int = Field(..., description="Total chunks in document")
    pdf_path: Optional[str] = Field(None, description="Source PDF path")
    page_numbers: Optional[List[int]] = Field(
        None, description="Page numbers this chunk spans"
    )

    def get_metadata_dict(self) -> Dict[str, Any]:
        """Get metadata as dictionary for vector store."""
        return {
            "item_id": self.item_id,
            "zotero_key": self.zotero_key,
            "title": self.title,
            "authors": ";".join(self.authors),
            "year": self.year or 0,
            "collections": ";".join(self.collections),
            "tags": ";".join(self.tags),
            "section": self.section or "",
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "pdf_path": self.pdf_path or "",
        }


class RetrievalResult(BaseModel):
    """Result from vector database retrieval."""

    chunk: DocumentChunk = Field(..., description="Retrieved chunk")
    distance: float = Field(..., description="Distance score (lower is better)")
    similarity: float = Field(..., description="Similarity score (0-1)")

    @classmethod
    def from_chroma_result(
        cls, document: str, metadata: Dict[str, Any], distance: float
    ) -> "RetrievalResult":
        """Create from ChromaDB query result."""
        # Convert metadata back to lists
        authors = metadata["authors"].split(";") if metadata["authors"] else []
        collections = (
            metadata["collections"].split(";") if metadata["collections"] else []
        )
        tags = metadata["tags"].split(";") if metadata["tags"] else []

        chunk = DocumentChunk(
            chunk_id=f"doc_{metadata['item_id']}_chunk_{metadata['chunk_index']}",
            text=document,
            item_id=metadata["item_id"],
            zotero_key=metadata["zotero_key"],
            title=metadata["title"],
            authors=authors,
            year=metadata["year"] if metadata["year"] > 0 else None,
            collections=collections,
            tags=tags,
            section=metadata.get("section"),
            chunk_index=metadata["chunk_index"],
            total_chunks=metadata["total_chunks"],
            pdf_path=metadata.get("pdf_path"),
        )

        # Convert L2 distance to similarity score (0-1 range)
        # L2 distances for normalized embeddings range ~0-2
        similarity = max(0.0, 1 - distance / 2)

        return cls(chunk=chunk, distance=distance, similarity=similarity)


class Answer(BaseModel):
    """Answer to a user question."""

    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    sources: List[ZoteroItem] = Field(
        default_factory=list, description="Source documents"
    )
    chunks_used: int = Field(..., description="Number of chunks retrieved")
    generation_time: float = Field(..., description="Time taken to generate answer")


class LiteratureReview(BaseModel):
    """Generated literature review."""

    title: str = Field(..., description="Review title")
    topic: str = Field(..., description="Review topic")
    introduction: str = Field(..., description="Introduction section")
    sections: List[tuple[str, str]] = Field(
        ..., description="Review sections (title, content)"
    )
    conclusion: str = Field(..., description="Conclusion section")
    bibliography: List[str] = Field(
        default_factory=list, description="Full citations"
    )
    generation_time: float = Field(..., description="Time taken to generate")


class ClaimReview(BaseModel):
    """Review of a single claim."""

    claim_text: str = Field(..., description="The claim being reviewed")
    supporting_evidence: List[str] = Field(
        default_factory=list, description="Supporting evidence from literature"
    )
    contradicting_evidence: List[str] = Field(
        default_factory=list, description="Contradicting evidence from literature"
    )
    sources: List[ZoteroItem] = Field(
        default_factory=list, description="Relevant sources"
    )
    assessment: str = Field(..., description="Overall assessment of the claim")


class ResearchReviewReport(BaseModel):
    """Complete research review report."""

    original_text: str = Field(..., description="Original research text")
    claim_reviews: List[ClaimReview] = Field(
        default_factory=list, description="Reviews of individual claims"
    )
    literature_gaps: List[str] = Field(
        default_factory=list, description="Identified gaps in literature coverage"
    )
    citation_suggestions: List[ZoteroItem] = Field(
        default_factory=list, description="Suggested papers to cite"
    )
    overall_assessment: str = Field(
        ..., description="Overall assessment of the research"
    )
    generation_time: float = Field(..., description="Time taken to generate")


class ProcessingStats(BaseModel):
    """Statistics from processing the library."""

    total_items: int = Field(..., description="Total items in library")
    items_with_pdfs: int = Field(..., description="Items with PDF attachments")
    pdfs_processed: int = Field(..., description="PDFs successfully processed")
    pdfs_failed: int = Field(..., description="PDFs that failed processing")
    total_chunks: int = Field(..., description="Total chunks created")
    total_embeddings: int = Field(..., description="Total embeddings generated")
    processing_time: float = Field(..., description="Total processing time in seconds")
    failed_items: List[Dict[str, str]] = Field(
        default_factory=list, description="List of failed items with reasons"
    )
