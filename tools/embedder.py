"""
tools/embedder.py
-----------------
Sentence embedding module for the Agentic Research Planning Framework.

Loads the all-MiniLM-L6-v2 model lazily on first use and caches it as a
module-level singleton, so the (multi-second) model load is paid only
when embeddings are actually needed — not at import time.
Used by the Reviewer Agent for novelty scoring.

Usage:
    from tools.embedder import get_embeddings
    embeddings = get_embeddings(["text one", "text two"])
    # returns np.ndarray of shape (2, 384)
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# Module-level singleton — loaded lazily on first get_embeddings() call.
# all-MiniLM-L6-v2: lightweight, fast, 384-dim embeddings.
# Well-suited for semantic similarity on scientific text.
_MODEL = None


def get_embeddings(texts: list[str]) -> np.ndarray:
    """
    Encode a list of strings into a 2D embedding matrix.

    Args:
        texts: List of strings to encode.

    Returns:
        np.ndarray of shape (len(texts), 384).
        Each row is the embedding vector for the corresponding input string.
    """
    global _MODEL
    if not texts:
        return np.zeros((0, 384))
    if _MODEL is None:
        print("[embedder] Loading sentence-transformer model (all-MiniLM-L6-v2)...")
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        print("[embedder] Model loaded.")
    return _MODEL.encode(texts, convert_to_numpy=True)
