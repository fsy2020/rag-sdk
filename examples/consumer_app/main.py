"""An independent project calling rag-sdk.

Embedded mode (the SDK runs the pipeline in-process):
    python main.py local  path/to/file.pdf "your question"

Remote mode (talks to a running `rag-sdk serve` / uvicorn rag_sdk.server:app):
    python main.py remote path/to/file.pdf "your question" --url http://localhost:8000
"""
import argparse
import os

from rag_sdk import HashEmbedder, RAGClient, RAGRemoteClient


def build_local_client() -> RAGClient:
    if os.getenv("RAG_EMBEDDER") == "hash":  # offline demo, no model download
        return RAGClient(embedder=HashEmbedder(), qdrant_url=os.getenv("QDRANT_URL"))
    # Default: local fastembed model + Qdrant (server if QDRANT_URL is set, else in-memory)
    return RAGClient.from_env()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["local", "remote"])
    p.add_argument("pdf")
    p.add_argument("question")
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("-k", "--top-k", type=int, default=5)
    args = p.parse_args()

    rag = build_local_client() if args.mode == "local" else RAGRemoteClient(args.url)
    info = rag.upload_pdf(args.pdf) if args.mode == "remote" else rag.ingest_pdf(args.pdf)
    print(f"Ingested {info.source}: {info.num_pages} pages, {info.num_chunks} chunks\n")

    for i, hit in enumerate(rag.search(args.question, top_k=args.top_k), 1):
        print(f"#{i}  score={hit.score:.4f}  page={hit.page_number}")
        print(f"    {hit.text[:200]}\n")


if __name__ == "__main__":
    main()
