"""Configuration settings for the What Works Against Crime Literature Expert."""

import os
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _inject_streamlit_secrets():
    """Inject Streamlit Cloud secrets into os.environ before Settings loads."""
    try:
        import streamlit as st
        for key, value in st.secrets.items():
            if isinstance(value, str) and key not in os.environ:
                os.environ[key] = value
    except Exception:
        pass


_inject_streamlit_secrets()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
    )

    # API Configuration
    anthropic_api_key: str = Field(default="", description="Anthropic API key")

    # PDF / CSV Configuration (for ingestion)
    pdf_folder: Optional[Path] = Field(
        default=None,
        description="Path to folder containing PDFs",
    )
    csv_path: Optional[Path] = Field(
        default=None,
        description="Path to CSV metadata file",
    )

    # Model Configuration
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers model for embeddings",
    )
    llm_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Claude model for text generation",
    )

    # Vector Database
    chromadb_path: Path = Field(
        default=Path("./data/chromadb"),
        description="Path to ChromaDB storage",
    )

    # Processing Configuration
    chunk_size: int = Field(default=1000, description="Target chunk size in tokens")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks in tokens")
    batch_size: int = Field(default=32, description="Batch size for embedding generation")

    # Cache Configuration
    cache_path: Path = Field(default=Path("./data/cache"))
    logs_path: Path = Field(default=Path("./data/logs"))

    # Retrieval Configuration
    top_k: int = Field(default=15, description="Number of chunks to retrieve for Q&A")
    similarity_threshold: float = Field(default=0.3, description="Minimum similarity score")

    # API Configuration
    max_tokens: int = Field(default=4000, description="Maximum tokens for LLM generation")
    temperature: float = Field(default=0.7, description="Temperature for LLM generation")

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.chromadb_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)
        (Path("./data/exports")).mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
