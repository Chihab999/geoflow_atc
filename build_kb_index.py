"""
Build KB Index — Reproducible FAISS Index Builder
===================================================
Builds the FAISS vector index and metadata from raw text documents
in kb_documents/. Makes the full pipeline reproducible from source.

Usage:
    python build_kb_index.py

Reads:
    kb_documents/*.txt, *.md  — raw text files (one paragraph per chunk)
    kb_documents/*.pdf        — PDF files (via pymupdf/fitz extraction)

Writes:
    kb_index/text_kb.faiss    — FAISS L2 index
    kb_index/metadata.json    — chunk metadata (text, source, chunk_id)
"""

import json
import os
import re
from pathlib import Path

import numpy as np

from config import PipelineConfig

cfg = PipelineConfig()
PROJECT_ROOT = cfg.project_root


def extract_text_from_file(filepath: Path) -> str:
    """Extract text content from .txt, .md, or .pdf files."""
    suffix = filepath.suffix.lower()

    if suffix in (".txt", ".md"):
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    elif suffix == ".pdf":
        try:
            import fitz  # pymupdf
            doc = fitz.open(str(filepath))
            pages = []
            for page in doc:
                pages.append(page.get_text())
            doc.close()
            return "\n".join(pages)
        except ImportError:
            print(f"  [WARN] pymupdf not installed. Skipping PDF: {filepath.name}")
            return ""
        except Exception as e:
            print(f"  [WARN] Failed to read PDF {filepath.name}: {e}")
            return ""

    return ""


def chunk_text(text: str, source: str, max_chunk_size: int = 500,
               overlap: int = 50) -> list:
    """Split text into overlapping chunks for embedding.
    
    Uses paragraph boundaries when possible, falls back to
    sentence-level splitting for long paragraphs.
    """
    # Split on double newlines (paragraphs)
    paragraphs = re.split(r"\n\s*\n", text.strip())
    chunks = []
    chunk_id = 0

    for para in paragraphs:
        para = para.strip()
        if not para or len(para) < 20:
            continue

        if len(para) <= max_chunk_size:
            chunks.append({
                "text": para,
                "source": source,
                "chunk_id": chunk_id,
            })
            chunk_id += 1
        else:
            # Split long paragraphs into sentence-level chunks
            sentences = re.split(r"(?<=[.!?])\s+", para)
            current_chunk = ""
            for sent in sentences:
                if len(current_chunk) + len(sent) + 1 <= max_chunk_size:
                    current_chunk += (" " if current_chunk else "") + sent
                else:
                    if current_chunk:
                        chunks.append({
                            "text": current_chunk.strip(),
                            "source": source,
                            "chunk_id": chunk_id,
                        })
                        chunk_id += 1
                        # Overlap: keep last portion
                        words = current_chunk.split()
                        overlap_words = words[-min(len(words), overlap // 5):]
                        current_chunk = " ".join(overlap_words) + " " + sent
                    else:
                        current_chunk = sent
            if current_chunk.strip():
                chunks.append({
                    "text": current_chunk.strip(),
                    "source": source,
                    "chunk_id": chunk_id,
                })
                chunk_id += 1

    return chunks


def build_index():
    """Build FAISS index from all documents in kb_documents/."""
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(f"[ERR] Missing dependency: {e}")
        print("   Install with: pip install faiss-cpu sentence-transformers")
        return

    kb_docs_dir = cfg.get_kb_documents_dir()
    kb_index_dir = cfg.get_kb_index_dir()
    kb_index_dir.mkdir(parents=True, exist_ok=True)

    print(f"[DOCS] Reading documents from {kb_docs_dir}...")

    # Collect all chunks
    all_chunks = []
    supported_extensions = {".txt", ".md", ".pdf"}

    for filepath in sorted(kb_docs_dir.iterdir()):
        if filepath.suffix.lower() not in supported_extensions:
            continue
        print(f"  -> {filepath.name}")
        text = extract_text_from_file(filepath)
        if not text:
            continue
        chunks = chunk_text(text, source=filepath.name)
        print(f"    {len(chunks)} chunks extracted")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("[ERR] No text chunks found. Check kb_documents/ directory.")
        return

    print(f"\n[METRICS] Total chunks: {len(all_chunks)}")

    # Embed all chunks
    print("[EMB] Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [c["text"] for c in all_chunks]
    print(f"[ENCODE] Encoding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings, dtype="float32")

    print(f"  Embedding shape: {embeddings.shape}")

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save
    faiss_path = kb_index_dir / "text_kb.faiss"
    meta_path = kb_index_dir / "metadata.json"

    faiss.write_index(index, str(faiss_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Index built successfully!")
    print(f"  FAISS index: {faiss_path} ({faiss_path.stat().st_size / 1024:.1f} KB)")
    print(f"  Metadata:    {meta_path} ({meta_path.stat().st_size / 1024:.1f} KB)")
    print(f"  Vectors:     {index.ntotal}")
    print(f"  Dimension:   {dim}")


if __name__ == "__main__":
    build_index()
