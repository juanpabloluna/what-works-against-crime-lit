"""PDF text extraction using PyMuPDF."""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from loguru import logger

from src.data.models import PDFDocument, PDFPage


class PDFExtractor:
    """Extract text content from PDF files."""

    def __init__(self):
        """Initialize PDF extractor."""
        self.min_text_ratio = 0.1  # Minimum text per page to consider it not scanned

    def extract_text(self, pdf_path: str) -> PDFDocument:
        """
        Extract text from a PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            PDFDocument with extracted text and metadata
        """
        pdf_path_obj = Path(pdf_path)

        if not pdf_path_obj.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            return PDFDocument(
                pdf_path=pdf_path,
                full_text="",
                pages=[],
                total_pages=0,
                total_chars=0,
                success=False,
                error_message="File not found",
            )

        try:
            # Open PDF
            doc = fitz.open(pdf_path)
            pages = []

            for page_num in range(len(doc)):
                page = doc[page_num]

                # Extract text
                text = page.get_text("text")

                # Clean the text
                cleaned_text = self._clean_text(text)

                pages.append(
                    PDFPage(
                        page_number=page_num + 1,  # 1-indexed
                        text=cleaned_text,
                        char_count=len(cleaned_text),
                    )
                )

            # Combine all pages
            full_text = "\n\n".join([p.text for p in pages if p.text.strip()])
            total_chars = sum(p.char_count for p in pages)

            # Check if PDF might be scanned (very little text)
            num_pages = len(doc)
            if num_pages > 0:
                avg_chars_per_page = total_chars / num_pages
                if avg_chars_per_page < 100:
                    logger.warning(
                        f"PDF appears to be scanned (avg {avg_chars_per_page:.1f} chars/page): {pdf_path}"
                    )

            doc.close()

            logger.debug(
                f"Extracted {total_chars} characters from {len(pages)} pages: {pdf_path_obj.name}"
            )

            return PDFDocument(
                pdf_path=pdf_path,
                full_text=full_text,
                pages=pages,
                total_pages=len(pages),
                total_chars=total_chars,
                extraction_date=datetime.now(),
                success=True,
            )

        except fitz.FileDataError as e:
            logger.error(f"Corrupted or encrypted PDF: {pdf_path} - {e}")
            return PDFDocument(
                pdf_path=pdf_path,
                full_text="",
                pages=[],
                total_pages=0,
                total_chars=0,
                success=False,
                error_message=f"Corrupted or encrypted: {str(e)}",
            )

        except Exception as e:
            logger.error(f"Error extracting PDF {pdf_path}: {e}")
            return PDFDocument(
                pdf_path=pdf_path,
                full_text="",
                pages=[],
                total_pages=0,
                total_chars=0,
                success=False,
                error_message=str(e),
            )

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted PDF text.

        - Remove excessive whitespace
        - Fix common PDF extraction issues
        - Normalize line breaks
        """
        if not text:
            return ""

        # Replace multiple spaces with single space
        text = re.sub(r" +", " ", text)

        # Fix hyphenation at line breaks (word- word -> word-word)
        text = re.sub(r"(\w+)-\s+(\w+)", r"\1-\2", text)

        # Fix line breaks in middle of sentences
        # (but preserve paragraph breaks)
        text = re.sub(r"([a-z,])\n([a-z])", r"\1 \2", text)

        # Normalize multiple newlines to max 2 (paragraph break)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Strip leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        return text.strip()

    def detect_sections(self, pdf_doc: PDFDocument) -> dict[str, str]:
        """
        Attempt to detect major sections in an academic paper.

        Returns a dict mapping section names to their text content.
        """
        text = pdf_doc.full_text

        sections = {}

        # Common section headers in academic papers
        section_patterns = {
            "abstract": r"\bAbstract\b",
            "introduction": r"\b(Introduction|1\s+Introduction)\b",
            "literature_review": r"\b(Literature Review|Related Work|Background)\b",
            "methodology": r"\b(Methodology|Methods|Research Design)\b",
            "results": r"\b(Results|Findings)\b",
            "discussion": r"\b(Discussion|Analysis)\b",
            "conclusion": r"\b(Conclusion|Conclusions)\b",
            "references": r"\b(References|Bibliography|Works Cited)\b",
        }

        # Find section positions
        section_positions = {}
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                section_positions[section_name] = match.start()

        # Sort sections by position
        sorted_sections = sorted(section_positions.items(), key=lambda x: x[1])

        # Extract text between sections
        for i, (section_name, start_pos) in enumerate(sorted_sections):
            if i < len(sorted_sections) - 1:
                # Text until next section
                end_pos = sorted_sections[i + 1][1]
                sections[section_name] = text[start_pos:end_pos].strip()
            else:
                # Last section gets remaining text
                sections[section_name] = text[start_pos:].strip()

        return sections

    def is_likely_scanned(self, pdf_doc: PDFDocument) -> bool:
        """Check if PDF is likely a scanned document with no OCR."""
        if pdf_doc.total_pages == 0:
            return True

        avg_chars = pdf_doc.total_chars / pdf_doc.total_pages
        return avg_chars < 100  # Less than 100 chars per page suggests scanned

    def get_metadata(self, pdf_path: str) -> Optional[dict]:
        """Extract PDF metadata (title, author, subject, etc.)."""
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            doc.close()
            return metadata
        except Exception as e:
            logger.error(f"Error reading PDF metadata {pdf_path}: {e}")
            return None
