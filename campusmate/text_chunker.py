"""
Text chunker — splits page text into overlapping chunks.
"""

from typing import List, Dict


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 100) -> List[str]:
    """
    Split text into chunks of approximately `chunk_size` characters
    with `overlap` characters of overlap between consecutive chunks.
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap

    return chunks


def chunk_pages(pages: List[Dict], chunk_size: int = 700, overlap: int = 100) -> List[Dict]:
    """
    Chunk all pages, preserving source metadata.

    Input:  [{"text": str, "source": str, "page": int}, ...]
    Output: [{"text": str, "source": str, "page": int, "chunk_id": int}, ...]
    """
    all_chunks = []
    chunk_id = 0

    for page in pages:
        text_chunks = chunk_text(page["text"], chunk_size, overlap)
        for tc in text_chunks:
            all_chunks.append({
                "text": tc,
                "source": page["source"],
                "page": page["page"],
                "chunk_id": chunk_id,
            })
            chunk_id += 1

    return all_chunks
