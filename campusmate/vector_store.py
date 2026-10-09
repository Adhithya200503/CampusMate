"""
FAISS vector store — build, save, and load the index + metadata.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

import faiss
import numpy as np


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Create a FAISS IndexFlatIP (inner product = cosine sim on normalized vecs)."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index(index: faiss.IndexFlatIP, metadata: List[Dict],
               index_path: Path, metadata_path: Path) -> None:
    """Save FAISS index and chunk metadata to disk."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def load_index(index_path: Path, metadata_path: Path):
    """Load FAISS index and metadata. Returns (index, metadata) or (None, None)."""
    if not index_path.exists() or not metadata_path.exists():
        return None, None
    try:
        index = faiss.read_index(str(index_path))
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return index, metadata
    except Exception as e:
        print(f"  [WARNING] Could not load index: {e}")
        return None, None
