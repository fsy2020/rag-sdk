"""Step 5: store and search vectors in Qdrant."""
from __future__ import annotations

import uuid
import warnings
from typing import List, Optional, Sequence

from qdrant_client import QdrantClient, models

from .models import Chunk, SearchResult

# namespace for deterministic point ids -> re-ingesting a file is idempotent
_NAMESPACE = uuid.UUID("6f1c1f7e-3b8a-4d1e-9a0a-5f0c2b7d9e11")


def make_qdrant_client(
    url: Optional[str] = None,
    path: Optional[str] = None,
    api_key: Optional[str] = None,
) -> QdrantClient:
    """``url`` -> Qdrant server; ``path`` -> embedded on-disk mode; neither -> in-memory."""
    if url:
        return QdrantClient(url=url, api_key=api_key)
    if path:
        return QdrantClient(path=path)
    return QdrantClient(location=":memory:")


class QdrantVectorStore:
    def __init__(self, client: QdrantClient, collection_name: str, dim: int) -> None:
        self.client = client
        self.collection_name = collection_name
        self.dim = dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            info = self.client.get_collection(self.collection_name)
            existing = info.config.params.vectors.size  # single unnamed vector
            if existing != self.dim:
                raise ValueError(
                    f"Collection '{self.collection_name}' has dim {existing}, but the embedder "
                    f"produces {self.dim}. Use another collection_name or the same model."
                )
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE),
        )
        with warnings.catch_warnings():  # local mode warns that indexes are no-ops
            warnings.simplefilter("ignore")
            self.client.create_payload_index(self.collection_name, "doc_id", models.PayloadSchemaType.KEYWORD)

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]], batch_size: int = 128) -> None:
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(_NAMESPACE, f"{c.doc_id}:{c.chunk_index}")),
                vector=list(v),
                payload={
                    "text": c.text,
                    "doc_id": c.doc_id,
                    "source": c.source,
                    "page_number": c.page_number,
                    "chunk_index": c.chunk_index,
                    "metadata": c.metadata,
                },
            )
            for c, v in zip(chunks, vectors)
        ]
        for i in range(0, len(points), batch_size):
            self.client.upsert(self.collection_name, points=points[i : i + batch_size], wait=True)

    def delete_document(self, doc_id: str) -> None:
        self.client.delete(
            self.collection_name,
            points_selector=models.FilterSelector(filter=_doc_filter(doc_id)),
            wait=True,
        )

    def search(self, vector: Sequence[float], top_k: int = 5, doc_id: Optional[str] = None) -> List[SearchResult]:
        hits = self.client.query_points(
            self.collection_name,
            query=list(vector),
            limit=top_k,
            query_filter=_doc_filter(doc_id) if doc_id else None,
            with_payload=True,
        ).points
        return [
            SearchResult(
                text=h.payload["text"],
                score=float(h.score),
                source=h.payload["source"],
                page_number=h.payload["page_number"],
                chunk_index=h.payload["chunk_index"],
                doc_id=h.payload["doc_id"],
                metadata=h.payload.get("metadata") or {},
            )
            for h in hits
        ]

    def count(self) -> int:
        return self.client.count(self.collection_name, exact=True).count


def _doc_filter(doc_id: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
