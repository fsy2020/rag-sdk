"""Step 1 + 6 over HTTP: upload a PDF and query it.

Run:  uvicorn rag_sdk.server:app --port 8000      (configure via env, see RAGClient.from_env)
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

from typing import Optional

try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from pydantic import BaseModel, Field
except ImportError as e:  # pragma: no cover
    raise ImportError("The server needs: pip install 'rag-sdk[server]'") from e

from .client import RAGClient


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(5, ge=1, le=100)
    doc_id: Optional[str] = None


def create_app(rag: Optional[RAGClient] = None) -> FastAPI:
    app = FastAPI(title="rag-sdk", version="0.1.0")
    state = {"rag": rag}

    def get_rag() -> RAGClient:
        if state["rag"] is None:  # build lazily so importing the module is cheap
            state["rag"] = RAGClient.from_env()
        return state["rag"]

    @app.get("/health")
    def health():
        return {"status": "ok", "chunks": get_rag().count()}

    @app.post("/documents")
    async def upload(file: UploadFile = File(...)):
        data = await file.read()
        if not data.startswith(b"%PDF"):
            raise HTTPException(400, "Uploaded file is not a PDF")
        return get_rag().ingest_pdf(data, source_name=file.filename or "upload.pdf").to_dict()

    @app.post("/search")
    def search(req: SearchRequest):
        if not req.query.strip():
            raise HTTPException(400, "query must not be empty")
        hits = get_rag().search(req.query, top_k=req.top_k, doc_id=req.doc_id)
        return {"query": req.query, "results": [h.to_dict() for h in hits]}

    @app.delete("/documents/{doc_id}")
    def delete(doc_id: str):
        get_rag().delete_document(doc_id)
        return {"deleted": doc_id}

    return app


app = create_app()
