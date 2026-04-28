"""Configuration helpers for Phase 2 infrastructure."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field

from fast_memory_write_env.schemas import StrictBaseModel


PINECONE_ENV_VARS = (
    "PINECONE_API_KEY",
    "PINECONE_INDEX_NAME",
    "PINECONE_CLOUD",
    "PINECONE_REGION",
)


class MissingPineconeConfigError(RuntimeError):
    """Raised when Pinecone config is required but incomplete."""


class PineconeConfig(StrictBaseModel):
    """Pinecone settings for the real retrieval backend."""

    api_key: str = Field(min_length=1)
    index_name: str = Field(min_length=1)
    cloud: str = Field(min_length=1)
    region: str = Field(min_length=1)
    dimension: int = Field(default=64, ge=1)
    metric: str = "cosine"
    namespace: str = "fast-memory-write-env"
    create_if_missing: bool = False

    @classmethod
    def from_env(
        cls,
        *,
        dimension: int = 64,
        namespace: str = "fast-memory-write-env",
        create_if_missing: bool = False,
    ) -> PineconeConfig:
        """Load required Pinecone configuration from environment variables."""

        missing = [name for name in PINECONE_ENV_VARS if not os.getenv(name)]
        if missing:
            joined = ", ".join(missing)
            raise MissingPineconeConfigError(f"Missing Pinecone environment variables: {joined}")
        return cls(
            api_key=os.environ["PINECONE_API_KEY"],
            index_name=os.environ["PINECONE_INDEX_NAME"],
            cloud=os.environ["PINECONE_CLOUD"],
            region=os.environ["PINECONE_REGION"],
            dimension=dimension,
            namespace=namespace,
            create_if_missing=create_if_missing,
        )


class EnvConfig(StrictBaseModel):
    """Filesystem and backend choices for local environment construction."""

    sqlite_path: Path
    pinecone: PineconeConfig | None = None


def pinecone_env_present() -> bool:
    """Return true when all required Pinecone env vars are present."""

    return all(os.getenv(name) for name in PINECONE_ENV_VARS)


def load_pinecone_config(
    *,
    required: bool = False,
    dimension: int = 64,
    namespace: str = "fast-memory-write-env",
    create_if_missing: bool = False,
) -> PineconeConfig | None:
    """Load Pinecone config if available, optionally requiring it."""

    if not pinecone_env_present():
        if required:
            return PineconeConfig.from_env(
                dimension=dimension,
                namespace=namespace,
                create_if_missing=create_if_missing,
            )
        return None
    return PineconeConfig.from_env(
        dimension=dimension,
        namespace=namespace,
        create_if_missing=create_if_missing,
    )
