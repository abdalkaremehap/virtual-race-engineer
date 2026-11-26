"""
FIA 2024 Sporting Regulations PDF -> Local Embedded Qdrant Indexer.

Allows the entire Virtual Race Engineer web app (including 2-Hop FIA Rulebook RAG)
to run inside a single Docker container on Hugging Face Spaces, Railway, or Render
with zero external Qdrant server required.
"""

from __future__ import annotations

import os
import re

from loguru import logger
from PyPDF2 import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DEFAULT_PDF_PATH = os.path.join(PROJECT_ROOT, "data", "fia_2024_sporting_regulations.pdf")
DEFAULT_LOCAL_QDRANT_PATH = os.path.join(PROJECT_ROOT, "data", "qdrant_local")
COLLECTION_NAME = "fia_rulebook"


def chunk_fia_pdf(pdf_path: str = DEFAULT_PDF_PATH, chunk_chars: int = 850, overlap_chars: int = 120) -> list[dict]:
    """Extracts and overlaps text chunks from the FIA Sporting Regulations PDF with exact 1-based page numbers."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"FIA rulebook PDF not found at {pdf_path}")

    reader = PdfReader(pdf_path)
    chunks: list[dict] = []
    for page_idx, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        cleaned = re.sub(r"[ \t]+", " ", raw)
        cleaned = re.sub(r"\n{2,}", "\n", cleaned).strip()
        if len(cleaned) < 40:
            continue

        start = 0
        while start < len(cleaned):
            end = min(len(cleaned), start + chunk_chars)
            piece = cleaned[start:end].strip()
            if len(piece) >= 40:
                chunks.append({"page": page_idx, "text": piece})
            if end >= len(cleaned):
                break
            start = max(start + 1, end - overlap_chars)

    return chunks


def ensure_local_qdrant_index(
    client: QdrantClient,
    model: SentenceTransformer,
    collection_name: str = COLLECTION_NAME,
    pdf_path: str = DEFAULT_PDF_PATH,
) -> int:
    """Ensures `collection_name` exists and is populated in an embedded/local Qdrant client."""
    try:
        existing = {c.name for c in client.get_collections().collections}
        if collection_name in existing:
            count = client.count(collection_name=collection_name).count
            if count > 0:
                return int(count)
    except Exception:
        pass

    logger.info(f"📚 [FIA Ingest] Building embedded Qdrant index from {os.path.basename(pdf_path)}...")
    chunks = chunk_fia_pdf(pdf_path=pdf_path)
    if not chunks:
        return 0

    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts, batch_size=64, show_progress_bar=False)
    dim = int(vectors.shape[1])

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    points = [
        PointStruct(id=idx + 1, vector=vectors[idx].tolist(), payload=chunks[idx])
        for idx in range(len(chunks))
    ]
    batch_size = 256
    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=collection_name, points=points[i : i + batch_size])

    logger.success(f"✅ [FIA Ingest] Indexed {len(points)} FIA rulebook chunks into embedded Qdrant!")
    return len(points)


if __name__ == "__main__":
    os.makedirs(DEFAULT_LOCAL_QDRANT_PATH, exist_ok=True)
    m = SentenceTransformer("all-MiniLM-L6-v2")
    cli = QdrantClient(path=DEFAULT_LOCAL_QDRANT_PATH)
    total = ensure_local_qdrant_index(cli, m)
    print(f"Embedded Qdrant ready at {DEFAULT_LOCAL_QDRANT_PATH} with {total} chunks.")
