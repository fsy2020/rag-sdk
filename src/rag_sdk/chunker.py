"""Step 3: split page text into overlapping chunks.

A small recursive character splitter: try to cut on paragraph breaks first,
then lines, then sentence punctuation (Chinese and English), then spaces, and
finally hard-cut. Chunks never cross page boundaries so every chunk keeps an
accurate page number.
"""
from __future__ import annotations

from typing import List, Sequence

from .models import Chunk, Page

DEFAULT_SEPARATORS: Sequence[str] = (
    "\n\n", "\n", "。", "！", "？", "；", ". ", "! ", "? ", "; ", "，", ", ", " ", "",
)


class TextChunker:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Sequence[str] = DEFAULT_SEPARATORS,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be in [0, chunk_size)")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = list(separators)

    # -- public -----------------------------------------------------------
    def split_text(self, text: str) -> List[str]:
        pieces = self._split(text, self.separators)
        return self._merge(pieces)

    def split_pages(self, pages: Sequence[Page], doc_id: str, source: str) -> List[Chunk]:
        chunks: List[Chunk] = []
        for page in pages:
            for piece in self.split_text(page.text):
                chunks.append(
                    Chunk(
                        text=piece,
                        doc_id=doc_id,
                        source=source,
                        page_number=page.page_number,
                        chunk_index=len(chunks),
                    )
                )
        return chunks

    # -- internals --------------------------------------------------------
    def _split(self, text: str, separators: List[str]) -> List[str]:
        """Break text into pieces no longer than chunk_size (separators kept)."""
        if len(text) <= self.chunk_size:
            return [text]
        sep, rest = separators[0], separators[1:]
        if sep == "":
            step = self.chunk_size
            return [text[i : i + step] for i in range(0, len(text), step)]
        if sep not in text:
            return self._split(text, rest)
        parts = text.split(sep)
        pieces: List[str] = []
        for i, part in enumerate(parts):
            if i < len(parts) - 1:
                part += sep  # keep punctuation attached to the sentence
            if not part:
                continue
            if len(part) <= self.chunk_size:
                pieces.append(part)
            else:
                pieces.extend(self._split(part, rest))
        return pieces

    def _merge(self, pieces: List[str]) -> List[str]:
        """Greedily pack pieces into chunks, carrying a tail overlap forward."""
        chunks: List[str] = []
        current: List[str] = []
        length = 0
        for piece in pieces:
            if current and length + len(piece) > self.chunk_size:
                chunks.append("".join(current).strip())
                # drop pieces from the front until only `overlap` chars remain
                while current and (length > self.chunk_overlap or length + len(piece) > self.chunk_size):
                    length -= len(current.pop(0))
            current.append(piece)
            length += len(piece)
        if current:
            chunks.append("".join(current).strip())
        return [c for c in chunks if c]
