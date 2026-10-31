"""Thin HTTP client for the FastAPI service in :mod:`rag_sdk.server`.

Use this when the RAG pipeline runs as a separate service and your project
should not load models or talk to Qdrant directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import httpx

from .models import IngestResult, SearchResult


class RAGRemoteClient:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 300.0) -> None:
        self._http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def upload_pdf(self, path: Union[str, Path]) -> IngestResult:
        path = Path(path)
        with path.open("rb") as f:
            resp = self._http.post("/documents", files={"file": (path.name, f, "application/pdf")})
        resp.raise_for_status()
        return IngestResult(**resp.json())

    def search(self, query: str, top_k: int = 5, doc_id: Optional[str] = None) -> List[SearchResult]:
        resp = self._http.post("/search", json={"query": query, "top_k": top_k, "doc_id": doc_id})
        resp.raise_for_status()
        return [SearchResult(**r) for r in resp.json()["results"]]

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "RAGRemoteClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
