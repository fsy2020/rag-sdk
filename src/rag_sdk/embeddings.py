"""Step 4: embedding models.

Every embedder implements the same tiny interface (:class:`Embedder`), so you
can swap a local model for a hosted API without touching the rest of the SDK.

* :class:`FastEmbedEmbedder`  – local ONNX model via ``fastembed`` (default,
  no API key; downloads the model from HuggingFace on first use).
* :class:`OpenAIEmbedder`     – any OpenAI-compatible ``/embeddings`` endpoint
  (OpenAI, Azure-compatible gateways, DashScope, SiliconFlow, Ollama, vLLM ...).
* :class:`HashEmbedder`       – deterministic, dependency-free bag-of-ngrams
  vectors. Not semantic; meant for tests and offline demos.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

Vector = List[float]


class Embedder(ABC):
    """Interface every embedding backend implements."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Vector dimension."""

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> List[Vector]:
        """Embed chunks that will be stored."""

    def embed_query(self, text: str) -> Vector:
        """Embed a search query (some models use a different prompt)."""
        return self.embed_documents([text])[0]


class FastEmbedEmbedder(Embedder):
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", **kwargs) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as e:  # pragma: no cover
            raise ImportError("FastEmbedEmbedder needs: pip install 'rag-sdk[fastembed]'") from e
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name, **kwargs)
        self._dim: Optional[int] = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed_query("dim probe"))
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> List[Vector]:
        return [v.tolist() for v in self._model.passage_embed(list(texts))]

    def embed_query(self, text: str) -> Vector:
        return next(iter(self._model.query_embed(text))).tolist()


class OpenAIEmbedder(Embedder):
    """Calls ``POST {base_url}/embeddings`` with the OpenAI request format."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dim: Optional[int] = None,
        batch_size: int = 64,
        timeout: float = 60.0,
    ) -> None:
        import httpx

        self.model = model
        self.batch_size = batch_size
        self._dim = dim
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(base_url=base_url, headers=headers, timeout=timeout)

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed_query("dim probe"))
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> List[Vector]:
        out: List[Vector] = []
        texts = list(texts)
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            resp = self._http.post("/embeddings", json={"model": self.model, "input": batch})
            resp.raise_for_status()
            data = sorted(resp.json()["data"], key=lambda d: d["index"])
            out.extend(d["embedding"] for d in data)
        return out


class HashEmbedder(Embedder):
    """Hashing-trick embeddings over character n-grams (CJK-friendly).

    Texts that share words/characters get similar vectors, which is enough to
    exercise the full pipeline without downloading a model. Not semantic.
    """

    def __init__(self, dim: int = 384, ngram_range: Sequence[int] = (1, 2)) -> None:
        self._dim = dim
        self.ngram_range = tuple(ngram_range)

    @property
    def dim(self) -> int:
        return self._dim

    def _tokens(self, text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r"[a-z0-9]+", text)  # latin words
        cjk = re.findall(r"[一-鿿]", text)  # CJK chars -> char n-grams
        lo, hi = self.ngram_range
        for n in range(lo, hi + 1):
            tokens += ["".join(cjk[i : i + n]) for i in range(len(cjk) - n + 1)]
        return tokens

    def embed_documents(self, texts: Sequence[str]) -> List[Vector]:
        vectors = []
        for text in texts:
            vec = [0.0] * self._dim
            for tok in self._tokens(text):
                h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "little")
                vec[h % self._dim] += 1.0 if (h >> 63) == 0 else -1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            vectors.append([x / norm for x in vec])
        return vectors
