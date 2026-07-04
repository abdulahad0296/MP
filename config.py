"""
config.py
---------
Central configuration for the Agentic Research Planning Framework.
All constants, thresholds, and API keys are loaded here.

Usage:
    from config import GROQ_API_KEY, MAX_PAPERS, NOVELTY_THRESHOLD

Setup:
    Copy .env.example to .env and fill in your GROQ_API_KEY.
    Get a free key at: https://console.groq.com
    Never commit .env to version control.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY is not set. "
        "Create a .env file with GROQ_API_KEY=your_key_here\n"
        "Get a free key at: https://console.groq.com"
    )

# ── LLM Settings ──────────────────────────────────────────────────────────────

# Model used for all LLM calls across Librarian, Planner, and Reviewer agents.
# llama-3.3-70b-versatile is the recommended free-tier model on Groq —
# strong instruction following and reliable JSON output.
LLM_MODEL: str = "llama-3.3-70b-versatile"

# Max tokens per LLM response — sufficient for structured JSON outputs
LLM_MAX_TOKENS: int = 1024

# ── arXiv Retrieval ───────────────────────────────────────────────────────────

# Maximum papers to retrieve from arXiv per query
MAX_PAPERS: int = 20

# Hard cap sent to the arXiv API (slightly higher to allow filtering)
ARXIV_RESULTS_LIMIT: int = 25

# ── Planner Agent ─────────────────────────────────────────────────────────────

# Number of candidate research plans generated per gap
MAX_PLANS_PER_GAP: int = 3

# Number of paper abstracts sent as context when generating plans
PLANNER_CONTEXT_PAPERS: int = 4

# ── Reviewer Agent ────────────────────────────────────────────────────────────

# Legacy fixed novelty threshold (0.0–10.0 scale). No longer used to gate
# acceptance — the Reviewer Agent now computes a corpus-derived threshold
# per run (see compute_novelty_threshold() in agents/reviewer_agent.py,
# which falls back to NOVELTY_THRESHOLD_STRICT below if computation fails).
# Kept only for historical reference / as a manual override point.
NOVELTY_THRESHOLD: float = 1.5

# Maximum number of times a rejected plan is re-submitted to the Planner
MAX_REVISION_ATTEMPTS: int = 2

# ── Novelty Scoring (Phase A upgrade) ────────────────────────────────────────

# Number of top-K most similar papers used for novelty scoring.
# Instead of averaging similarity across all retrieved papers, the scorer
# takes the K most similar papers and scores against those — making it
# more discriminative against plans that closely mirror even a few papers.
TOP_K_PAPERS: int = 5

# Strict novelty threshold for the top-K scoring approach.
# Higher than the original mean-based threshold (1.5) because top-K
# similarity scores are inherently higher than full-corpus mean scores.
# Plans scoring below this are rejected as insufficiently novel.
# Recalibrate after running test topics if needed.
NOVELTY_THRESHOLD_STRICT: float = 2.5

# ── Output ────────────────────────────────────────────────────────────────────

# Path where final results are saved
OUTPUT_PATH: str = "outputs/results.json"