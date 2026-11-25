import os
import httpx
import numpy as np
from pdf_processor import DocumentChunk

JINA_EMBED_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL = "jina-embeddings-v5-text-small"


def get_embeddings(texts: list[str], task: str = "retrieval.passage") -> np.ndarray:
    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        raise RuntimeError("JINA_API_KEY environment variable is not set.")

    all_embeddings: list[list[float]] = []
    batch_size = 32
    with httpx.Client(timeout=30.0) as client:
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = client.post(
                JINA_EMBED_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                json={
                    "model": JINA_MODEL,
                    "task": task,
                    "normalized": True,
                    "input": batch,
                },
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            all_embeddings.extend(item["embedding"] for item in data)

    return np.array(all_embeddings, dtype=np.float32)


class LightVectorStore:
    def __init__(self, chunks: list[DocumentChunk]):
        self.chunks = chunks
        texts = [f"[{c.section_title}] {c.text}" for c in chunks]
        self.matrix = get_embeddings(texts, task="retrieval.passage") if chunks else np.empty((0, 1024))

    def similarity_search(self, query: str, top_k: int = 4) -> list[tuple[DocumentChunk, float]]:
        if not self.chunks:
            return []
        q_vec = get_embeddings([query], task="retrieval.query")[0]
        # Vectors from Jina are L2-normalized, so dot product == cosine similarity
        scores = self.matrix @ q_vec
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[int(idx)], float(scores[int(idx)])) for idx in top_indices]