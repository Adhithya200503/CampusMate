"""
PDF document loader — extracts text from PDFs in the data/ folder.
"""

import hashlib
from pathlib import Path
from typing import List, Dict

import pymupdf


def get_pdf_files(data_dir: Path) -> List[Path]:
    """Return sorted list of PDF files in data_dir."""
    return sorted(data_dir.glob("*.pdf"))


def compute_file_hash(filepath: Path) -> str:
    """Compute MD5 hash of a file for change detection."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_text_from_pdf(filepath: Path) -> List[Dict]:
    """
    Extract text from each page of a PDF.

    Returns list of dicts: {"text": str, "source": str, "page": int}
    Handles corrupted/empty PDFs gracefully.
    """
    pages = []
    filename = filepath.name
    try:
        doc = pymupdf.open(str(filepath))
    except Exception as e:
        print(f"  [WARNING] Cannot open '{filename}': {e}")
        return pages

    for page_num in range(len(doc)):
        try:
            page = doc[page_num]
            text = page.get_text("text").strip()
            if text:
                pages.append({
                    "text": text,
                    "source": filename,
                    "page": page_num + 1,  # 1-indexed
                })
        except Exception as e:
            print(f"  [WARNING] Error reading page {page_num + 1} of '{filename}': {e}")

    doc.close()

    if not pages:
        print(f"  [INFO] '{filename}' has no extractable text (may need OCR for scanned PDFs).")

    return pages


def load_all_pdfs(data_dir: Path) -> List[Dict]:
    """Load and extract text from all PDFs in data_dir."""
    pdf_files = get_pdf_files(data_dir)
    if not pdf_files:
        print("  No PDF files found in data/ folder.")
        return []

    all_pages = []
    for pdf_path in pdf_files:
        print(f"  Reading: {pdf_path.name}")
        pages = extract_text_from_pdf(pdf_path)
        all_pages.extend(pages)

    print(f"  Extracted {len(all_pages)} pages from {len(pdf_files)} PDF(s).")
    return all_pages
