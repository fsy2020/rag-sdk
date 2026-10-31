import pytest

from rag_sdk import HashEmbedder, RAGClient, TextChunker, parse_pdf


def test_parse_pdf(sample_pdf):
    pages = parse_pdf(sample_pdf)
    assert [p.page_number for p in pages] == [1, 2, 3, 4]
    assert "vector database" in pages[0].text


def test_chunker_respects_size_and_overlap():
    text = "。".join(f"这是第{i}句话，用来测试切分" for i in range(100)) + "。"
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.split_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    # consecutive chunks overlap
    assert any(chunks[i][-10:] in chunks[i + 1] for i in range(len(chunks) - 1))
    # nothing lost
    assert "第99句话" in chunks[-1]


def test_chunker_hard_cuts_long_tokens():
    chunks = TextChunker(chunk_size=10, chunk_overlap=0).split_text("x" * 35)
    assert [len(c) for c in chunks] == [10, 10, 10, 5]


def test_chunker_rejects_bad_overlap():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=10, chunk_overlap=10)


@pytest.fixture
def rag():
    return RAGClient(embedder=HashEmbedder(), chunk_size=120, chunk_overlap=20)


def test_ingest_and_search(rag, sample_pdf):
    result = rag.ingest_pdf(sample_pdf)
    assert result.num_pages == 4
    assert result.num_chunks == rag.count() >= 4

    hits = rag.search("which vector database is written in Rust?", top_k=5)
    assert len(hits) == 5
    assert hits[0].page_number == 1
    assert hits[0].source == "sample.pdf"
    assert hits == sorted(hits, key=lambda h: -h.score)

    hits = rag.search("pasta and hot chocolate", top_k=1)
    assert hits[0].page_number == 4


def test_reingest_is_idempotent(rag, sample_pdf):
    first = rag.ingest_pdf(sample_pdf)
    rag.ingest_pdf(sample_pdf.read_bytes(), source_name="sample.pdf")
    assert rag.count() == first.num_chunks


def test_filter_and_delete_by_doc(rag, sample_pdf, tmp_path):
    from pdf_utils import make_pdf

    other = tmp_path / "other.pdf"
    other.write_bytes(make_pdf([["Rust is also a systems programming language."]]))
    a = rag.ingest_pdf(sample_pdf)
    b = rag.ingest_pdf(other)

    hits = rag.search("Rust", top_k=5, doc_id=b.doc_id)
    assert {h.doc_id for h in hits} == {b.doc_id}

    rag.delete_document(b.doc_id)
    assert rag.count() == a.num_chunks


def test_persistent_local_qdrant(sample_pdf, tmp_path):
    path = str(tmp_path / "qdrant")
    rag = RAGClient(embedder=HashEmbedder(), qdrant_path=path)
    n = rag.ingest_pdf(sample_pdf).num_chunks
    rag.store.client.close()

    reopened = RAGClient(embedder=HashEmbedder(), qdrant_path=path)
    assert reopened.count() == n
    assert reopened.search("Qdrant Rust", top_k=1)[0].page_number == 1


def test_dimension_mismatch_is_reported(sample_pdf):
    from qdrant_client import QdrantClient

    client = QdrantClient(location=":memory:")
    RAGClient(embedder=HashEmbedder(dim=64), qdrant_client=client)
    with pytest.raises(ValueError, match="dim"):
        RAGClient(embedder=HashEmbedder(dim=128), qdrant_client=client)
