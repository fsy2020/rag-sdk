"""The main entry point: :class:`RAGClient` wires the whole pipeline together."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Union

from qdrant_client import QdrantClient

from .chunker import TextChunker
from .embeddings import Embedder, FastEmbedEmbedder, HashEmbedder, OpenAIEmbedder
from .models import IngestResult, SearchResult
from .pdf_parser import parse_pdf
from .vector_store import QdrantVectorStore, make_qdrant_client


class RAGClient:
    """PDF -> parse -> chunk -> embed -> Qdrant, and top-k retrieval.

    Example::

        from rag_sdk import RAGClient

        rag = RAGClient(qdrant_url="http://localhost:6333")
        rag.ingest_pdf("paper.pdf")
        for hit in rag.search("What is the main contribution?", top_k=5):
            print(hit.score, hit.page_number, hit.text[:80])
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        *,
        qdrant_url: Optional[str] = None,
        qdrant_path: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        qdrant_client: Optional[QdrantClient] = None,
        collection_name: str = "rag_chunks",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        self.embedder = embedder or FastEmbedEmbedder()
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        client = qdrant_client or make_qdrant_client(qdrant_url, qdrant_path, qdrant_api_key)
        self.store = QdrantVectorStore(client, collection_name, self.embedder.dim)

    # ------------------------------------------------------------------ factory
    @classmethod
    def from_env(cls, **overrides: Any) -> "RAGClient":
        """Build a client from environment variables.

        RAG_EMBEDDER          fastembed (default) | openai | hash
        RAG_EMBEDDING_MODEL   model name for fastembed / openai
        OPENAI_API_KEY / OPENAI_BASE_URL   for the openai embedder
        QDRANT_URL / QDRANT_API_KEY        Qdrant server
        QDRANT_PATH           embedded on-disk Qdrant (used if QDRANT_URL unset)
        RAG_COLLECTION        collection name (default rag_chunks)
        RAG_CHUNK_SIZE / RAG_CHUNK_OVERLAP
        """
        kind = os.getenv("RAG_EMBEDDER", "fastembed").lower()
        model = os.getenv("RAG_EMBEDDING_MODEL")
        if "embedder" not in overrides:
            if kind == "openai":
                overrides["embedder"] = OpenAIEmbedder(**({"model": model} if model else {}))
            elif kind == "hash":
                overrides["embedder"] = HashEmbedder()
            elif kind == "fastembed":
                overrides["embedder"] = FastEmbedEmbedder(**({"model_name": model} if model else {}))
            else:
                raise ValueError(f"Unknown RAG_EMBEDDER: {kind}")
        defaults: Dict[str, Any] = {
            "qdrant_url": os.getenv("QDRANT_URL"),
            "qdrant_path": os.getenv("QDRANT_PATH"),
            "qdrant_api_key": os.getenv("QDRANT_API_KEY"),
            "collection_name": os.getenv("RAG_COLLECTION", "rag_chunks"),
            "chunk_size": int(os.getenv("RAG_CHUNK_SIZE", "500")),
            "chunk_overlap": int(os.getenv("RAG_CHUNK_OVERLAP", "50")),
        }
        defaults.update(overrides)
        return cls(**defaults)

    # ------------------------------------------------------------------- ingest
    def ingest_pdf(
        self,
        source: Union[str, Path, bytes, BinaryIO],
        *,
        source_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IngestResult:
        """Parse, chunk, embed and store one PDF. Re-ingesting the same file replaces it."""
        if isinstance(source, (str, Path)):
            data = Path(source).read_bytes()
            source_name = source_name or Path(source).name
        elif isinstance(source, (bytes, bytearray)):
            data = bytes(source)
        else:
            data = source.read()
            source_name = source_name or Path(getattr(source, "name", "") or "upload.pdf").name
        source_name = source_name or "upload.pdf"
        doc_id = hashlib.sha256(data).hexdigest()[:32]

        pages = parse_pdf(data)                                      # 2. parse
        chunks = self.chunker.split_pages(pages, doc_id, source_name)  # 3. chunk
        for c in chunks:
            c.metadata = dict(metadata or {})

        self.store.delete_document(doc_id)
        if chunks:
            vectors = self.embedder.embed_documents([c.text for c in chunks])  # 4. embed
            self.store.upsert(chunks, vectors)                                # 5. store
        return IngestResult(doc_id=doc_id, source=source_name, num_pages=len(pages), num_chunks=len(chunks))

    # ------------------------------------------------------------------- search
    def search(self, query: str, top_k: int = 5, *, doc_id: Optional[str] = None) -> List[SearchResult]:
        """6. Return the ``top_k`` chunks most similar to ``query``."""
        if not query.strip():
            raise ValueError("query must not be empty")
        vector = self.embedder.embed_query(query)
        return self.store.search(vector, top_k=top_k, doc_id=doc_id)

    def delete_document(self, doc_id: str) -> None:
        self.store.delete_document(doc_id)

    def count(self) -> int:
        """Number of chunks stored in the collection."""
        return self.store.count()
