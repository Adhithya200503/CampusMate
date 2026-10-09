"""
Retriever — query the FAISS index and return relevant chunks.
"""

from typing import List, Dict, Tuple

import faiss
import numpy as np


def retrieve(query_embedding: np.ndarray, index: faiss.IndexFlatIP,
             metadata: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Retrieve top_k chunks most similar to the query.

    Returns list of dicts with keys: text, source, page, score.
    """
    if index.ntotal == 0:
        return []

    k = min(top_k, index.ntotal)
    scores, indices = index.search(query_embedding, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        chunk = metadata[idx].copy()
        chunk["score"] = float(score)
        results.append(chunk)

    return results
