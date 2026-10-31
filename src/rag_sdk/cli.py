"""Command line: ``rag-sdk ingest file.pdf`` / ``rag-sdk search "question"``.

Configuration comes from environment variables (see RAGClient.from_env).
"""
from __future__ import annotations

import argparse
import json
import sys

from .client import RAGClient


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="rag-sdk")
    sub = p.add_subparsers(dest="cmd", required=True)
    ing = sub.add_parser("ingest", help="parse, chunk, embed and store PDFs")
    ing.add_argument("pdfs", nargs="+")
    se = sub.add_parser("search", help="retrieve the most relevant chunks")
    se.add_argument("query")
    se.add_argument("-k", "--top-k", type=int, default=5)
    se.add_argument("--json", action="store_true")
    srv = sub.add_parser("serve", help="run the HTTP API")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("rag_sdk.server:app", host=args.host, port=args.port)
        return 0

    rag = RAGClient.from_env()
    if args.cmd == "ingest":
        for pdf in args.pdfs:
            r = rag.ingest_pdf(pdf)
            print(f"{r.source}: {r.num_pages} pages -> {r.num_chunks} chunks (doc_id={r.doc_id})")
    else:
        hits = rag.search(args.query, top_k=args.top_k)
        if args.json:
            print(json.dumps([h.to_dict() for h in hits], ensure_ascii=False, indent=2))
        else:
            for i, h in enumerate(hits, 1):
                print(f"[{i}] score={h.score:.4f}  {h.source} p.{h.page_number}\n    {h.text[:200]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
