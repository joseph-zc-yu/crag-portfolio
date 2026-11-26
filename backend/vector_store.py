import os
import time
import httpx
import numpy as np
from pdf_processor import DocumentChunk

GEMINI_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/openai/embeddings"


def get_embeddings(texts: list[str]) -> np.ndarray:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

    model = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
    all_embeddings: list[list[float]] = []
    batch_size = 25

    with httpx.Client(timeout=30.0) as client:
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for attempt in range(3):
                resp = client.post(
                    GEMINI_EMBED_URL,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    },
                    json={
                        "model": model,
                        "input": batch,
                    },
                )
                if resp.status_code == 429 and attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                resp.raise_for_status()
                break

            data = sorted(resp.json()["data"], key=lambda x: x.get("index", 0))
            all_embeddings.extend(item["embedding"] for item in data)

    arr = np.array(all_embeddings, dtype=np.float32)
    # L2-normalize so dot product == cosine similarity
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-12, None)


class LightVectorStore:
    def __init__(self, chunks: list[DocumentChunk]):
        self.chunks = chunks
        texts = [f"[{c.section_title}] {c.text}" for c in chunks]
        self.matrix = get_embeddings(texts) if chunks else np.empty((0, 3072), dtype=np.float32)

    def similarity_search(self, query: str, top_k: int = 4) -> list[tuple[DocumentChunk, float]]:
        if not self.chunks:
            return []
        q_vec = get_embeddings([query])[0]
        scores = self.matrix @ q_vec
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[int(idx)], float(scores[int(idx)])) for idx in top_indices]