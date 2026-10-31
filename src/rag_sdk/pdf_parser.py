"""Step 2: parse a PDF into per-page text."""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import BinaryIO, List, Union

from pypdf import PdfReader

from .models import Page

PdfSource = Union[str, Path, bytes, BinaryIO]


def _normalize(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t　]+", " ", text)  # collapse horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)  # at most one blank line
    return text.strip()


def parse_pdf(source: PdfSource) -> List[Page]:
    """Extract text from every page. Pages without text (e.g. scans) are skipped."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    reader = PdfReader(source)
    pages: List[Page] = []
    for i, page in enumerate(reader.pages, start=1):
        text = _normalize(page.extract_text() or "")
        if text:
            pages.append(Page(page_number=i, text=text))
    return pages
