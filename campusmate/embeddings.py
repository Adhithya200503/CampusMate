"""
Embedding generator using Sentence Transformers.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List


_model = None


def get_model(model_name: str) -> SentenceTransformer:
    """Load the embedding model (cached after first call)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model


def generate_embeddings(texts: List[str], model_name: str) -> np.ndarray:
    """
    Generate normalized embeddings for a list of texts.
    Returns a numpy array of shape (N, dim) with L2-normalized vectors.
    """
    model = get_model(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    # L2 normalize for cosine similarity with IndexFlatIP
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    embeddings = embeddings / norms
    return embeddings.astype("float32")


def embed_query(query: str, model_name: str) -> np.ndarray:
    """Embed a single query string. Returns shape (1, dim)."""
    model = get_model(model_name)
    embedding = model.encode([query], convert_to_numpy=True)
    norm = np.linalg.norm(embedding, axis=1, keepdims=True)
    norm[norm == 0] = 1
    embedding = embedding / norm
    return embedding.astype("float32")
