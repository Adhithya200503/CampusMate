"""
RAG pipeline — orchestrates indexing and question answering.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import config
from campusmate.document_loader import load_all_pdfs, get_pdf_files, compute_file_hash
from campusmate.text_chunker import chunk_pages
from campusmate.embeddings import generate_embeddings, embed_query
from campusmate.vector_store import build_index, save_index, load_index
from campusmate.retriever import retrieve
from campusmate.llm_client import call_llm, build_prompt


def _compute_pdf_hashes(data_dir: Path) -> Dict[str, str]:
    """Compute hashes for all PDFs in data_dir."""
    hashes = {}
    for pdf in get_pdf_files(data_dir):
        hashes[pdf.name] = compute_file_hash(pdf)
    return hashes


def _load_saved_hashes(hash_path: Path) -> Dict[str, str]:
    """Load previously saved PDF hashes."""
    if not hash_path.exists():
        return {}
    try:
        with open(hash_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_hashes(hashes: Dict[str, str], hash_path: Path) -> None:
    """Save PDF hashes to disk."""
    with open(hash_path, "w") as f:
        json.dump(hashes, f, indent=2)


def needs_reindex(data_dir: Path, hash_path: Path,
                  index_path: Path, metadata_path: Path) -> bool:
    """Check if PDFs have changed and reindexing is needed."""
    if not index_path.exists() or not metadata_path.exists():
        return True
    current = _compute_pdf_hashes(data_dir)
    saved = _load_saved_hashes(hash_path)
    return current != saved


def build_full_index() -> Tuple[Optional[object], Optional[List[Dict]]]:
    """Build FAISS index from all PDFs in data/."""
    print("Loading PDF documents...")
    pages = load_all_pdfs(config.DATA_DIR)
    if not pages:
        print("  No content to index.")
        return None, None

    print("Splitting text into chunks...")
    chunks = chunk_pages(pages, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    print(f"  Created {len(chunks)} chunks.")

    print("Generating embeddings (this may take a moment)...")
    texts = [c["text"] for c in chunks]
    embeddings = generate_embeddings(texts, config.EMBEDDING_MODEL)

    print("Building FAISS index...")
    index = build_index(embeddings)

    # Metadata without the full text for storage (keep text for retrieval)
    metadata = chunks  # store full metadata including text

    save_index(index, metadata, config.FAISS_INDEX_FILE, config.METADATA_FILE)
    _save_hashes(_compute_pdf_hashes(config.DATA_DIR), config.HASH_FILE)

    print(f"  Index saved with {index.ntotal} vectors.")
    return index, metadata


def load_or_build_index() -> Tuple[Optional[object], Optional[List[Dict]]]:
    """Load existing index or build a new one if PDFs changed."""
    if needs_reindex(config.DATA_DIR, config.HASH_FILE,
                     config.FAISS_INDEX_FILE, config.METADATA_FILE):
        print("Building/loading FAISS index...")
        return build_full_index()
    else:
        print("Loading existing FAISS index...")
        index, metadata = load_index(config.FAISS_INDEX_FILE, config.METADATA_FILE)
        if index is not None:
            print(f"  Loaded index with {index.ntotal} vectors.")
        return index, metadata


def answer_question(question: str, index, metadata: List[Dict]) -> Tuple[str, List[Dict]]:
    """
    Answer a question using the RAG pipeline.
    Returns (answer_text, source_chunks).
    """
    if index is None or not metadata:
        return "No documents indexed. Add PDFs to data/ and run /reindex.", []

    # Embed the question
    q_embedding = embed_query(question, config.EMBEDDING_MODEL)

    # Retrieve relevant chunks
    results = retrieve(q_embedding, index, metadata, config.TOP_K)
    if not results:
        return "No relevant information found in the indexed documents.", []

    # Build prompt and call LLM
    prompt = build_prompt(question, results)
    answer = call_llm(config.SYSTEM_PROMPT, prompt,
                      config.LLM_API_KEY, config.LLM_MODEL, config.LLM_BASE_URL)

    return answer, results


def get_indexed_sources(metadata: Optional[List[Dict]]) -> List[str]:
    """Return unique PDF filenames from metadata."""
    if not metadata:
        return []
    sources = sorted(set(chunk["source"] for chunk in metadata))
    return sources
