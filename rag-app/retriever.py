import os
import json
from typing import List, Tuple

import faiss
from sentence_transformers import SentenceTransformer

from config import INDEX_PATH, METADATA_PATH, TOP_K

# ── Module-level cache for fast repeated queries ──────────────────────────────
_embedder: SentenceTransformer = None
_cached_index: faiss.IndexFlatL2 = None
_cached_metadata: List[dict] = None
_cached_index_mtime: float = 0.0

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'


def _get_embedder() -> SentenceTransformer:
    """Load the embedding model once and cache it in memory."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(MODEL_NAME)
    return _embedder


def load_index_and_metadata(
    index_path: str = INDEX_PATH,
    metadata_path: str = METADATA_PATH
) -> Tuple[faiss.IndexFlatL2, List[dict]]:
    """
    Load FAISS index and metadata, using a module-level cache.
    If the index file has been updated (new ingest), the cache is invalidated.
    """
    global _cached_index, _cached_metadata, _cached_index_mtime

    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Missing index or metadata.\n"
            f"Expected index at: {index_path}\n"
            f"Expected metadata at: {metadata_path}\n"
            f"Run ingest first: python main.py ingest --docs ./docs"
        )

    current_mtime = os.path.getmtime(index_path)
    if _cached_index is None or current_mtime != _cached_index_mtime:
        _cached_index = faiss.read_index(index_path)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            _cached_metadata = json.load(f)
        _cached_index_mtime = current_mtime

    return _cached_index, _cached_metadata


def retrieve(
    query: str,
    index: faiss.IndexFlatL2,
    metadata: List[dict],
    top_k: int = TOP_K
) -> List[dict]:
    """Embed the query and return top-k matching chunks from the FAISS index."""
    embedder = _get_embedder()
    q_emb = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb.astype('float32'), top_k)
    results: List[dict] = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        item = metadata[idx].copy()
        item['score'] = float(score)
        results.append(item)
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('query')
    args = parser.parse_args()
    index, metadata = load_index_and_metadata()
    res = retrieve(args.query, index, metadata)
    for r in res:
        print(r['score'], r['source'])
