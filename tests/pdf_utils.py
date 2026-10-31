"""Build tiny text PDFs without extra dependencies (ASCII text, Helvetica)."""
from typing import List


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(pages: List[List[str]]) -> bytes:
    """``pages`` is a list of pages, each a list of text lines."""
    objs: List[bytes] = []
    n = len(pages)
    font_id = 3 + 2 * n
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    for i, lines in enumerate(pages):
        content_id = 4 + 2 * i
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>".encode()
        )
        ops = ["BT", "/F1 11 Tf", "14 TL", "50 750 Td"]
        for line in lines:
            ops.append(f"({_escape(line)}) Tj T*")
        ops.append("ET")
        stream = "\n".join(ops).encode()
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


SAMPLE_PAGES = [
    [
        "Qdrant is an open source vector database written in Rust.",
        "It stores embeddings together with a JSON payload and supports filtering.",
        "Collections use cosine, dot product or euclidean distance.",
    ],
    [
        "PDF parsing extracts the raw text of every page of a document.",
        "The pypdf library reads PDF files in pure Python without native dependencies.",
    ],
    [
        "Chunking splits long documents into smaller overlapping pieces.",
        "Each chunk is converted into a vector by an embedding model.",
        "Retrieval returns the top k chunks closest to the question vector.",
    ],
    [
        "The weather in the mountains was cold and snowy during the winter holiday.",
        "We cooked pasta with tomato sauce and drank hot chocolate by the fire.",
    ],
]
