"""Vector store interface using ChromaDB."""

from pathlib import Path
from typing import List, Optional, Dict, Any

import chromadb
try:
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    ChromaSettings = None
from loguru import logger

from src.config.settings import settings
from src.data.models import DocumentChunk, RetrievalResult


class VectorStore:
    """ChromaDB vector store for document chunks."""

    def __init__(
        self,
        collection_name: str = "what_works_literature",
        persist_directory: Optional[Path] = None,
    ):
        """
        Initialize vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database (default from settings)
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or settings.chromadb_path

        # Ensure directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        logger.info(f"Initializing ChromaDB at: {self.persist_directory}")
        client_kwargs = {"path": str(self.persist_directory)}
        if ChromaSettings is not None:
            client_kwargs["settings"] = ChromaSettings(anonymized_telemetry=False)
        self.client = chromadb.PersistentClient(**client_kwargs)

        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(
                f"Loaded existing collection '{self.collection_name}' with {self.collection.count()} documents"
            )
        except Exception:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Academic papers from Zotero library",
                    "embedding_model": settings.embedding_model,
                },
            )
            logger.info(f"Created new collection '{self.collection_name}'")

    def add_chunks(
        self, chunks: List[DocumentChunk], embeddings: List[List[float]]
    ) -> None:
        """
        Add document chunks with their embeddings to the vector store.

        Args:
            chunks: List of DocumentChunk objects
            embeddings: List of embedding vectors (must match chunks length)
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) must match number of embeddings ({len(embeddings)})"
            )

        if not chunks:
            logger.warning("No chunks to add")
            return

        # Prepare data for ChromaDB
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [chunk.get_metadata_dict() for chunk in chunks]

        # Add to collection in batches (ChromaDB has a limit)
        batch_size = 5000
        for i in range(0, len(chunks), batch_size):
            batch_end = min(i + batch_size, len(chunks))

            self.collection.add(
                ids=ids[i:batch_end],
                documents=documents[i:batch_end],
                embeddings=embeddings[i:batch_end],
                metadatas=metadatas[i:batch_end],
            )

            logger.debug(f"Added batch {i // batch_size + 1}: {batch_end - i} chunks")

        logger.info(f"Added {len(chunks)} chunks to vector store")

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """
        Query the vector store for similar chunks.

        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where: Metadata filters (e.g., {"year": {"$gte": 2020}})
            where_document: Document content filters

        Returns:
            List of RetrievalResult objects
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            where_document=where_document,
        )

        # Convert to RetrievalResult objects
        retrieval_results = []

        if results and results["documents"] and results["documents"][0]:
            for i, document in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]

                result = RetrievalResult.from_chroma_result(
                    document=document, metadata=metadata, distance=distance
                )
                retrieval_results.append(result)

        logger.debug(f"Query returned {len(retrieval_results)} results")
        return retrieval_results

    def query_by_text(
        self,
        query_text: str,
        embedding_service,
        n_results: int = 10,
        collections: Optional[List[str]] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        author_names: Optional[List[str]] = None,
    ) -> List[RetrievalResult]:
        """
        Query using text (will generate embedding automatically).

        Args:
            query_text: Text query
            embedding_service: EmbeddingService instance
            n_results: Number of results
            collections: Filter by collection names
            min_year: Minimum publication year
            max_year: Maximum publication year
            author_names: Author last names to boost in results

        Returns:
            List of RetrievalResult objects
        """
        # Generate embedding once
        query_embedding = embedding_service.embed_text(query_text).tolist()

        # Build metadata filters
        where_filters = {}

        if min_year:
            where_filters["year"] = {"$gte": min_year}
        if max_year:
            if "year" in where_filters:
                where_filters["year"]["$lte"] = max_year
            else:
                where_filters["year"] = {"$lte": max_year}

        where = where_filters if where_filters else None
        collections_filter = collections

        # --- Hybrid retrieval when author names are detected ---
        if author_names:
            return self._hybrid_author_query(
                query_embedding=query_embedding,
                author_names=author_names,
                n_results=n_results,
                where=where,
                collections_filter=collections_filter,
            )

        # --- Standard semantic retrieval ---
        results = self.query(
            query_embedding=query_embedding,
            n_results=n_results if not collections_filter else n_results * 3,
            where=where,
        )

        # Post-filter by collections if specified
        if collections_filter:
            filtered_results = []
            for result in results:
                chunk_collections = result.chunk.collections
                if any(
                    coll in chunk_collections for coll in collections_filter
                ):
                    filtered_results.append(result)
                if len(filtered_results) >= n_results:
                    break
            return filtered_results

        return results

    def _hybrid_author_query(
        self,
        query_embedding: List[float],
        author_names: List[str],
        n_results: int,
        where: Optional[Dict[str, Any]] = None,
        collections_filter: Optional[List[str]] = None,
    ) -> List[RetrievalResult]:
        """
        Hybrid retrieval combining semantic search with author-focused retrieval.

        Three-pronged approach:
        1. Semantic search (standard)
        2. Text search (where_document $contains author name)
        3. Post-filter semantic results for author metadata matches

        Results are merged with author-matched results prioritized.
        """
        # 1. Standard semantic query (fetch more to allow for author filtering)
        semantic_results = self.query(
            query_embedding=query_embedding,
            n_results=n_results * 3,
            where=where,
        )

        # 2. Author text search using where_document
        author_text_results = []
        for name in author_names:
            try:
                text_results = self.query(
                    query_embedding=query_embedding,
                    n_results=n_results,
                    where=where,
                    where_document={"$contains": name},
                )
                author_text_results.extend(text_results)
            except Exception as e:
                logger.warning(f"Author text search failed for '{name}': {e}")

        # 3. Merge with author-matched results first
        seen_ids = set()
        merged = []

        # Priority 1: Semantic results where author is in metadata
        for r in semantic_results:
            authors_str = ";".join(r.chunk.authors).lower()
            if any(name.lower() in authors_str for name in author_names):
                if r.chunk.chunk_id not in seen_ids:
                    merged.append(r)
                    seen_ids.add(r.chunk.chunk_id)

        # Priority 2: Text-match results (chunks that mention the author)
        for r in author_text_results:
            if r.chunk.chunk_id not in seen_ids:
                merged.append(r)
                seen_ids.add(r.chunk.chunk_id)

        # Priority 3: Remaining semantic results
        for r in semantic_results:
            if r.chunk.chunk_id not in seen_ids:
                merged.append(r)
                seen_ids.add(r.chunk.chunk_id)

        # Post-filter by collections if specified
        if collections_filter:
            merged = [
                r for r in merged
                if any(coll in r.chunk.collections for coll in collections_filter)
            ]

        logger.info(
            f"Hybrid author query: {len(merged)} results "
            f"(authors detected: {author_names})"
        )

        return merged[:n_results * 2]  # Return extra for diversity ranking

    def get_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get a specific chunk by its ID."""
        try:
            result = self.collection.get(ids=[chunk_id])
            if result and result["documents"]:
                return RetrievalResult.from_chroma_result(
                    document=result["documents"][0],
                    metadata=result["metadatas"][0],
                    distance=0.0,
                ).chunk
        except Exception as e:
            logger.error(f"Error getting chunk {chunk_id}: {e}")

        return None

    def delete_by_item_id(self, item_id: int) -> int:
        """
        Delete all chunks for a specific Zotero item.

        Args:
            item_id: Zotero item ID

        Returns:
            Number of chunks deleted
        """
        # Get all chunks for this item
        try:
            result = self.collection.get(where={"item_id": item_id})
            if result and result["ids"]:
                self.collection.delete(ids=result["ids"])
                logger.info(f"Deleted {len(result['ids'])} chunks for item {item_id}")
                return len(result["ids"])
        except Exception as e:
            logger.error(f"Error deleting chunks for item {item_id}: {e}")

        return 0

    def count(self) -> int:
        """Get total number of chunks in the vector store."""
        return self.collection.count()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        count = self.count()

        # Get ALL metadata to compute accurate stats
        all_data = self.collection.get(include=["metadatas"])

        unique_items = set()
        unique_collections = set()
        years = []

        if all_data and all_data["metadatas"]:
            for metadata in all_data["metadatas"]:
                unique_items.add(metadata.get("item_id"))
                if metadata.get("collections"):
                    for coll in metadata["collections"].split(";"):
                        if coll.strip():
                            unique_collections.add(coll.strip())
                if metadata.get("year") and metadata["year"] > 0:
                    years.append(metadata["year"])

        return {
            "total_chunks": count,
            "sample_unique_items": len(unique_items),
            "sample_collections": list(unique_collections),
            "year_range": (min(years), max(years)) if years else (None, None),
        }

    def reset(self) -> None:
        """Delete all data and reset the collection."""
        logger.warning(f"Resetting collection '{self.collection_name}'")
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={
                "description": "Academic papers from Zotero library",
                "embedding_model": settings.embedding_model,
            },
        )
        logger.info(f"Collection '{self.collection_name}' reset")
