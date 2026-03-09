"""CSV-based metadata reader for literature processing.

Replaces ZoteroReader — reads paper metadata from a user-provided CSV file
and locates PDFs in a local folder.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from src.data.models import ZoteroItem


class CSVReader:
    """Read paper metadata from CSV and locate PDFs in a folder."""

    def __init__(self, csv_path: Path, pdf_folder: Path):
        self.csv_path = Path(csv_path)
        self.pdf_folder = Path(pdf_folder)

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        if not self.pdf_folder.exists():
            raise FileNotFoundError(f"PDF folder not found: {self.pdf_folder}")

        logger.info(f"CSVReader: csv={self.csv_path}, pdfs={self.pdf_folder}")

    def get_items_with_pdfs(self) -> List[ZoteroItem]:
        """Read CSV and return ZoteroItem objects for entries with matching PDFs."""
        items = []
        missing = []

        with open(self.csv_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=1):
                filename = row.get("filename", "").strip()
                if not filename:
                    logger.warning(f"Row {idx}: missing filename, skipping")
                    continue

                # Support both absolute paths and relative filenames
                pdf_path = Path(filename)
                if not pdf_path.is_absolute():
                    pdf_path = self.pdf_folder / filename
                if not pdf_path.exists():
                    missing.append(filename)
                    continue

                authors = self._parse_authors(row.get("authors", ""))
                year = self._parse_year(row.get("year", ""))

                item = ZoteroItem(
                    item_id=idx,
                    zotero_key=f"csv_{idx:04d}",
                    title=row.get("title", filename).strip(),
                    authors=authors,
                    year=year,
                    abstract=row.get("abstract", "").strip() or None,
                    publication=row.get("publication", "").strip() or None,
                    doi=row.get("doi", "").strip() or None,
                    url=row.get("url", "").strip() or None,
                    collections=[],
                    tags=[],
                    pdf_path=str(pdf_path),
                )
                items.append(item)

        if missing:
            logger.warning(f"{len(missing)} PDFs not found: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        logger.info(f"Loaded {len(items)} items with PDFs from CSV ({len(missing)} missing)")
        return items

    def export_metadata_json(self, items: List[ZoteroItem], output_path: Path) -> None:
        """Export metadata to papers_metadata.json for the web app."""
        records = []
        for item in items:
            records.append({
                "item_id": item.item_id,
                "item_type": "journalArticle",
                "title": item.title,
                "authors": item.authors,
                "year": item.year,
                "publication": item.publication or "",
                "doi": item.doi or "",
                "url": item.url or "",
                "abstract": item.abstract or "",
                "collections": item.collections,
            })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        logger.info(f"Exported {len(records)} records to {output_path}")

    @staticmethod
    def _parse_authors(raw: str) -> List[str]:
        """Parse semicolon-separated author names."""
        if not raw.strip():
            return []
        return [a.strip() for a in raw.split(";") if a.strip()]

    @staticmethod
    def _parse_year(raw: str) -> Optional[int]:
        """Parse year from string."""
        raw = raw.strip()
        if not raw:
            return None
        try:
            return int(raw[:4])
        except (ValueError, IndexError):
            return None
