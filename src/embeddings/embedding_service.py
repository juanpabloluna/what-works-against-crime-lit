"""Embedding generation service using sentence-transformers."""

from typing import List, Union

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config.settings import settings


class EmbeddingService:
    """Generate embeddings for text using sentence-transformers."""

    def __init__(self, model_name: str = None):
        """
        Initialize embedding service.

        Args:
            model_name: Name of sentence-transformers model (default from settings)
        """
        self.model_name = model_name or settings.embedding_model
        self.batch_size = settings.batch_size

        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(
            f"Embedding model loaded. Dimension: {self.embedding_dim}, Batch size: {self.batch_size}"
        )

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Numpy array of embedding vector
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,  # Normalize for cosine similarity
            show_progress_bar=False,
        )
        return embedding

    def embed_batch(
        self, texts: List[str], show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar

        Returns:
            Numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])

        logger.info(f"Generating embeddings for {len(texts)} texts...")

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings

    def embed_chunks_with_progress(self, chunks: List, text_field: str = "text") -> List[np.ndarray]:
        """
        Generate embeddings for chunks with detailed progress tracking.

        Args:
            chunks: List of chunk objects with text field
            text_field: Name of the field containing text

        Returns:
            List of embedding vectors
        """
        texts = [getattr(chunk, text_field) for chunk in chunks]

        embeddings_list = []

        # Process in batches with progress bar
        num_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        with tqdm(total=len(texts), desc="Generating embeddings") as pbar:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                batch_embeddings = self.model.encode(
                    batch,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )

                # Ensure batch_embeddings is always 2D for consistent handling
                if len(batch_embeddings.shape) == 1:
                    # Single embedding - reshape to 2D
                    batch_embeddings = batch_embeddings.reshape(1, -1)

                # Now always extend with 2D array
                embeddings_list.extend(batch_embeddings)

                pbar.update(len(batch))

        return embeddings_list

    def similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score between 0 and 1
        """
        # Cosine similarity (embeddings are already normalized)
        return float(np.dot(embedding1, embedding2))

    def batch_similarity(
        self, query_embedding: np.ndarray, embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Calculate similarity between query and multiple embeddings.

        Args:
            query_embedding: Query embedding vector
            embeddings: Array of embedding vectors (shape: num_embeddings x embedding_dim)

        Returns:
            Array of similarity scores
        """
        # Matrix multiplication for batch cosine similarity
        return np.dot(embeddings, query_embedding)

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "max_seq_length": self.model.max_seq_length,
            "batch_size": self.batch_size,
        }
