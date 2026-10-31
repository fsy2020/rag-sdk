"""rag-sdk: minimal PDF retrieval pipeline on top of Qdrant."""
from .chunker import TextChunker
from .client import RAGClient
from .embeddings import Embedder, FastEmbedEmbedder, HashEmbedder, OpenAIEmbedder
from .models import Chunk, IngestResult, Page, SearchResult
from .pdf_parser import parse_pdf
from .remote import RAGRemoteClient
from .vector_store import QdrantVectorStore, make_qdrant_client

__version__ = "0.1.0"

__all__ = [
    "RAGClient",
    "RAGRemoteClient",
    "Embedder",
    "FastEmbedEmbedder",
    "OpenAIEmbedder",
    "HashEmbedder",
    "TextChunker",
    "QdrantVectorStore",
    "make_qdrant_client",
    "parse_pdf",
    "Page",
    "Chunk",
    "SearchResult",
    "IngestResult",
]
