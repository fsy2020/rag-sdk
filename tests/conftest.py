import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from pdf_utils import SAMPLE_PAGES, make_pdf  # noqa: E402


@pytest.fixture
def sample_pdf(tmp_path) -> Path:
    path = tmp_path / "sample.pdf"
    path.write_bytes(make_pdf(SAMPLE_PAGES))
    return path
