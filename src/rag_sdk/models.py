"""Plain data objects shared across the SDK."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Page:
    """Text of a single PDF page (page_number is 1-based)."""

    page_number: int
    text: str


@dataclass
class Chunk:
    """A piece of a document that gets embedded and stored."""

    text: str
    doc_id: str
    source: str
    page_number: int
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """One retrieval hit returned by :meth:`RAGClient.search`."""

    text: str
    score: float
    source: str
    page_number: int
    chunk_index: int
    doc_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "score": self.score,
            "source": self.source,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "doc_id": self.doc_id,
            "metadata": self.metadata,
        }


@dataclass
class IngestResult:
    """Summary of an ingested document."""

    doc_id: str
    source: str
    num_pages: int
    num_chunks: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "num_pages": self.num_pages,
            "num_chunks": self.num_chunks,
        }


__all__: List[str] = ["Page", "Chunk", "SearchResult", "IngestResult"]
