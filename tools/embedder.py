"""
tools/embedder.py
-----------------
Sentence embedding module for the Agentic Research Planning Framework.

Loads the all-MiniLM-L6-v2 model once at module level to avoid reloading
on every call. Used by the Reviewer Agent for novelty scoring.

Usage:
    from tools.embedder import get_embeddings
    embeddings = get_embeddings(["text one", "text two"])
    # returns np.ndarray of shape (2, 384)
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# Load once at module level — avoids reloading on every call.
# all-MiniLM-L6-v2: lightweight, fast, 384-dim embeddings.
# Well-suited for semantic similarity on scientific text.
print("[embedder] Loading sentence-transformer model (all-MiniLM-L6-v2)...")
_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
print("[embedder] Model loaded.")


def get_embeddings(texts: list[str]) -> np.ndarray:
    """
    Encode a list of strings into a 2D embedding matrix.

    Args:
        texts: List of strings to encode.

    Returns:
        np.ndarray of shape (len(texts), 384).
        Each row is the embedding vector for the corresponding input string.
    """
    if not texts:
        return np.zeros((0, 384))
    return _MODEL.encode(texts, convert_to_numpy=True)
