#!/usr/bin/env python3
"""
Process PDF library from CSV metadata: Extract PDFs, chunk documents,
generate embeddings, and populate vector database.

Usage:
    python scripts/process_library.py
    python scripts/process_library.py --limit 5          # test with 5 papers
    python scripts/process_library.py --reset             # reset DB and reprocess
    python scripts/process_library.py --csv path.csv --pdfs /path/to/pdfs
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from tqdm import tqdm

from src.config.settings import settings
from src.data.chunker import DocumentChunker
from src.data.csv_reader import CSVReader
from src.data.models import ProcessingStats
from src.data.pdf_extractor import PDFExtractor
from src.embeddings.embedding_service import EmbeddingService
from src.embeddings.vector_store import VectorStore


def process_library(
    csv_path: Optional[Path] = None,
    pdf_folder: Optional[Path] = None,
    reset: bool = False,
    limit: Optional[int] = None,
) -> ProcessingStats:
    """Process PDF library from CSV metadata and populate vector database."""
    start_time = time.time()

    csv_path = csv_path or settings.csv_path
    pdf_folder = pdf_folder or settings.pdf_folder

    if not csv_path or not pdf_folder:
        raise ValueError("csv_path and pdf_folder must be set (via args, .env, or settings)")

    settings.ensure_directories()

    # Initialize components
    logger.info("Initializing components...")
    csv_reader = CSVReader(csv_path=csv_path, pdf_folder=pdf_folder)
    pdf_extractor = PDFExtractor()
    chunker = DocumentChunker()
    embedding_service = EmbeddingService()
    vector_store = VectorStore()

    if reset:
        logger.warning("Resetting vector database...")
        vector_store.reset()

    # Read CSV and get items with PDFs
    items = csv_reader.get_items_with_pdfs()

    if not items:
        logger.error("No items found with PDFs")
        return ProcessingStats(
            total_items=0, items_with_pdfs=0, pdfs_processed=0,
            pdfs_failed=0, total_chunks=0, total_embeddings=0,
            processing_time=time.time() - start_time, failed_items=[],
        )

    if limit:
        items = items[:limit]
        logger.info(f"Limited to {limit} items for testing")

    # Export papers_metadata.json for the web app
    metadata_path = Path("data/papers_metadata.json")
    csv_reader.export_metadata_json(items, metadata_path)

    logger.info(f"Processing {len(items)} items...")

    pdfs_processed = 0
    pdfs_failed = 0
    all_chunks = []
    failed_items = []

    for item in tqdm(items, desc="Processing PDFs"):
        try:
            if not item.pdf_path:
                failed_items.append({"item_id": str(item.item_id), "title": item.title, "reason": "No PDF path"})
                pdfs_failed += 1
                continue

            pdf_doc = pdf_extractor.extract_text(item.pdf_path)

            if not pdf_doc.success:
                failed_items.append({"item_id": str(item.item_id), "title": item.title, "reason": pdf_doc.error_message})
                pdfs_failed += 1
                continue

            if pdf_extractor.is_likely_scanned(pdf_doc):
                logger.warning(f"PDF appears scanned (item {item.item_id}): {item.title[:50]}")

            chunks = chunker.chunk_document(pdf_doc, item)

            if not chunks:
                failed_items.append({"item_id": str(item.item_id), "title": item.title, "reason": "No chunks created"})
                pdfs_failed += 1
                continue

            chunker.validate_chunks(chunks)
            all_chunks.extend(chunks)
            pdfs_processed += 1

            logger.debug(f"Processed item {item.item_id}: {len(chunks)} chunks from {pdf_doc.total_pages} pages")

        except Exception as e:
            logger.error(f"Error processing item {item.item_id}: {e}", exc_info=True)
            failed_items.append({"item_id": str(item.item_id), "title": item.title, "reason": str(e)})
            pdfs_failed += 1

    logger.info(f"PDF processing complete: {pdfs_processed} succeeded, {pdfs_failed} failed")
    logger.info(f"Total chunks created: {len(all_chunks)}")

    if not all_chunks:
        logger.error("No chunks to process!")
        return ProcessingStats(
            total_items=len(items), items_with_pdfs=len(items),
            pdfs_processed=pdfs_processed, pdfs_failed=pdfs_failed,
            total_chunks=0, total_embeddings=0,
            processing_time=time.time() - start_time, failed_items=failed_items,
        )

    # Generate embeddings
    logger.info("Generating embeddings...")
    embeddings = embedding_service.embed_chunks_with_progress(all_chunks)
    logger.info(f"Generated {len(embeddings)} embeddings")

    # Add to vector store
    logger.info("Adding chunks to vector database...")
    embeddings_list = [emb.tolist() if hasattr(emb, "tolist") else emb for emb in embeddings]
    vector_store.add_chunks(all_chunks, embeddings_list)

    vs_stats = vector_store.get_stats()
    logger.info(f"Vector store stats: {vs_stats}")

    processing_time = time.time() - start_time
    stats = ProcessingStats(
        total_items=len(items), items_with_pdfs=len(items),
        pdfs_processed=pdfs_processed, pdfs_failed=pdfs_failed,
        total_chunks=len(all_chunks), total_embeddings=len(embeddings),
        processing_time=processing_time, failed_items=failed_items,
    )

    logger.info(f"Processing complete in {processing_time:.2f} seconds")
    logger.info(f"Success rate: {pdfs_processed}/{len(items)} ({pdfs_processed/len(items)*100:.1f}%)")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Process PDF library and build vector database")
    parser.add_argument("--csv", type=str, help="Path to CSV metadata file")
    parser.add_argument("--pdfs", type=str, help="Path to PDF folder")
    parser.add_argument("--reset", action="store_true", help="Reset vector database before processing")
    parser.add_argument("--limit", type=int, help="Limit number of items (for testing)")
    parser.add_argument("--stats-file", type=str, default="processing_stats.json")

    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else None
    pdf_folder = Path(args.pdfs) if args.pdfs else None

    try:
        stats = process_library(csv_path=csv_path, pdf_folder=pdf_folder, reset=args.reset, limit=args.limit)

        with open(args.stats_file, "w") as f:
            json.dump(stats.model_dump(), f, indent=2, default=str)

        print("\n" + "=" * 60)
        print("PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total items: {stats.total_items}")
        print(f"PDFs processed: {stats.pdfs_processed}")
        print(f"PDFs failed: {stats.pdfs_failed}")
        print(f"Total chunks: {stats.total_chunks}")
        print(f"Total embeddings: {stats.total_embeddings}")
        print(f"Processing time: {stats.processing_time:.2f} seconds")
        print("=" * 60)

        if stats.failed_items:
            print(f"\nFailed items ({len(stats.failed_items)}):")
            for item in stats.failed_items[:10]:
                print(f"  - {item['title'][:60]}: {item['reason']}")
            if len(stats.failed_items) > 10:
                print(f"  ... and {len(stats.failed_items) - 10} more")

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
