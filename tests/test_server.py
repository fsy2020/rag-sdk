import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from rag_sdk import HashEmbedder, RAGClient, RAGRemoteClient  # noqa: E402
from rag_sdk.server import create_app  # noqa: E402


@pytest.fixture
def api():
    return TestClient(create_app(RAGClient(embedder=HashEmbedder(), chunk_size=120)))


def test_upload_and_search(api, sample_pdf):
    with sample_pdf.open("rb") as f:
        r = api.post("/documents", files={"file": ("sample.pdf", f, "application/pdf")})
    assert r.status_code == 200, r.text
    assert r.json()["num_pages"] == 4

    r = api.post("/search", json={"query": "Qdrant vector database Rust", "top_k": 5})
    results = r.json()["results"]
    assert len(results) == 5
    assert results[0]["page_number"] == 1


def test_rejects_non_pdf(api):
    r = api.post("/documents", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_remote_client(api, sample_pdf):
    remote = RAGRemoteClient()
    remote._http = api  # route SDK calls through the in-process test app
    info = remote.upload_pdf(sample_pdf)
    hits = remote.search("pasta hot chocolate", top_k=2)
    assert hits[0].page_number == 4 and hits[0].doc_id == info.doc_id
