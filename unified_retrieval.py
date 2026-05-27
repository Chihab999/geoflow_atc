"""
Unified Knowledge Base — Hybrid Retrieval with Re-Ranking
===========================================================
RAG (Retrieval-Augmented Generation) module combining:
  1. Dense retrieval via FAISS + sentence-transformers (semantic)
  2. BM25-style sparse retrieval (lexical/keyword matching)
  3. Re-ranking: weighted combination of dense + sparse scores

This provides superior recall on exact ATC-20 terminology while
maintaining semantic generalization for paraphrased queries.

Usage:
    from unified_retrieval import UnifiedKnowledgeBase
    kb = UnifiedKnowledgeBase()
    results = kb.retrieve_text("Red Placard structural criteria", k=5)
"""

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import PipelineConfig

cfg = PipelineConfig()


# ─────────────────────────────────────────────────────────────
# BM25 Sparse Retrieval (lightweight, no external dependency)
# ─────────────────────────────────────────────────────────────

class BM25:
    """BM25 scoring for sparse keyword-based retrieval.
    
    Okapi BM25 with standard parameters k1=1.5, b=0.75.
    """

    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.n_docs = len(documents)

        # Tokenize all documents
        self.doc_tokens = [self._tokenize(doc) for doc in documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lengths) / max(self.n_docs, 1)

        # Build document frequency (df) for IDF
        self.df = Counter()
        for tokens in self.doc_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.df[token] += 1

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + lowercase tokenizer."""
        return re.findall(r"\w+", text.lower())

    def _idf(self, term: str) -> float:
        """Inverse document frequency with smoothing."""
        df = self.df.get(term, 0)
        return math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)

    def score(self, query: str, doc_idx: int) -> float:
        """BM25 score for a single document."""
        query_tokens = self._tokenize(query)
        doc_tokens = self.doc_tokens[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        tf_counter = Counter(doc_tokens)

        score = 0.0
        for term in query_tokens:
            tf = tf_counter.get(term, 0)
            idf = self._idf(term)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1))
            score += idf * numerator / max(denominator, 1e-8)
        return score

    def search(self, query: str, k: int = 5) -> List[tuple]:
        """Return top-k (doc_idx, score) pairs sorted by score descending."""
        scores = [(i, self.score(query, i)) for i in range(self.n_docs)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


# ─────────────────────────────────────────────────────────────
# Unified Knowledge Base
# ─────────────────────────────────────────────────────────────

class UnifiedKnowledgeBase:
    """Hybrid retrieval KB combining dense (FAISS) and sparse (BM25) search.
    
    Dense retrieval captures semantic similarity; sparse retrieval catches
    exact terminology matches (critical for standardized ATC-20 language).
    
    Args:
        config: PipelineConfig instance (defaults to global cfg).
        dense_weight: Weight for dense retrieval scores in re-ranking.
        sparse_weight: Weight for sparse retrieval scores in re-ranking.
    """

    def __init__(self, config: PipelineConfig = None,
                 dense_weight: float = 0.6, sparse_weight: float = 0.4):
        config = config or cfg
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        index_dir = config.get_kb_index_dir()

        print("[INIT] Initializing Unified Multi-Modal Knowledge Base...")

        # Lazy-load embedding model
        self.text_model = None

        # Load FAISS Index
        self.text_index = self._load_index(index_dir / "text_kb.faiss")

        # Load Metadata
        self.text_meta = self._load_meta(index_dir / "metadata.json")

        # Build BM25 index from metadata text
        texts = [m.get("text", "") for m in self.text_meta] if self.text_meta else []
        self.bm25 = BM25(texts) if texts else None

        print(f"[OK] Unified KB ready. {len(self.text_meta)} passages indexed.")

    def _load_index(self, path: Path):
        """Load FAISS index from file."""
        if path.exists():
            return faiss.read_index(str(path))
        else:
            print(f"[WARN] Warning: Index not found at {path}.")
            return None

    def _load_meta(self, path: Path) -> list:
        """Load metadata JSON."""
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            print(f"[WARN] Warning: Metadata not found at {path}.")
            return []

    def _get_text_model(self):
        """Lazy-load sentence transformer."""
        if self.text_model is None:
            self.text_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self.text_model

    def retrieve_text(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve ATC-20 / FEMA structural criteria using hybrid search.
        
        Combines dense (FAISS semantic) and sparse (BM25 keyword) retrieval,
        then re-ranks by weighted score combination.
        
        Args:
            query: Search query string.
            k: Number of results to return.
        
        Returns:
            List of dicts with 'text', 'source', 'score', 'dense_score', 'sparse_score'.
        """
        if not self.text_meta:
            return []

        candidates = {}  # idx -> {dense_score, sparse_score}
        k_fetch = min(k * 3, len(self.text_meta))  # Over-fetch for re-ranking

        # 1. Dense retrieval (FAISS)
        if self.text_index is not None:
            model = self._get_text_model()
            emb = model.encode([query]).astype("float32")
            distances, indices = self.text_index.search(emb, k_fetch)
            for i in range(k_fetch):
                idx = int(indices[0][i])
                if idx != -1 and idx < len(self.text_meta):
                    # Convert L2 distance to similarity (lower distance = higher similarity)
                    # Normalize to [0, 1] range approximately
                    dense_score = 1.0 / (1.0 + float(distances[0][i]))
                    candidates[idx] = {"dense_score": dense_score, "sparse_score": 0.0}

        # 2. Sparse retrieval (BM25)
        if self.bm25 is not None:
            bm25_results = self.bm25.search(query, k=k_fetch)
            # Normalize BM25 scores
            max_bm25 = max((s for _, s in bm25_results), default=1.0) or 1.0
            for idx, score in bm25_results:
                normalized = score / max_bm25
                if idx in candidates:
                    candidates[idx]["sparse_score"] = normalized
                else:
                    candidates[idx] = {"dense_score": 0.0, "sparse_score": normalized}

        # 3. Re-rank by weighted combination
        scored = []
        for idx, scores in candidates.items():
            combined = (self.dense_weight * scores["dense_score"] +
                        self.sparse_weight * scores["sparse_score"])
            scored.append((idx, combined, scores["dense_score"], scores["sparse_score"]))

        scored.sort(key=lambda x: x[1], reverse=True)

        # 4. Build results
        results = []
        for idx, combined_score, dense_s, sparse_s in scored[:k]:
            if idx < len(self.text_meta):
                res = self.text_meta[idx].copy()
                res["score"] = float(combined_score)
                res["dense_score"] = float(dense_s)
                res["sparse_score"] = float(sparse_s)
                results.append(res)

        return results

    def hybrid_search(self, text_query: str = None, list_k: int = 3) -> Dict:
        """Multi-modal evidence retrieval.
        
        Args:
            text_query: Text search query.
            list_k: Number of results per modality.
        
        Returns:
            Dict with 'text_evidence' key.
        """
        evidence = {"text_evidence": []}
        if text_query:
            evidence["text_evidence"] = self.retrieve_text(text_query, k=list_k)
        return evidence

    def get_corpus_text(self) -> str:
        """Return the full KB corpus as a single lowercase string.
        
        Used by the evidence sanitizer for grounding checks.
        """
        texts = [m.get("text", "") for m in self.text_meta]
        return "\n".join(texts).lower()


if __name__ == "__main__":
    # Test the Retrieval Interface
    kb = UnifiedKnowledgeBase()

    print("\n--- 1. Testing Hybrid Text Retrieval ---")
    query = "What are the requirements for a restricted use yellow placard?"
    print(f"Query: '{query}'")
    results = kb.retrieve_text(query, k=3)
    for r in results:
        print(f"  [{r.get('source', 'unknown')}] "
              f"(Combined: {r['score']:.4f}, Dense: {r.get('dense_score', 0):.4f}, "
              f"Sparse: {r.get('sparse_score', 0):.4f})")
        print(f"    -> {r['text'][:120]}...")

    print("\n--- 2. Testing Green Placard Query ---")
    results2 = kb.retrieve_text("Green placard no apparent hazard", k=2)
    for r in results2:
        print(f"  [{r.get('source', 'unknown')}] Score: {r['score']:.4f}")
        print(f"    -> {r['text'][:120]}...")

    print("\n[OK] Retrieval tests complete!")
