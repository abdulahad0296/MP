"""
tools/feasibility_checker.py
-----------------------------
Rule-based + online feasibility checker for the Agentic Research Planning Framework.

Dataset verification uses the Hugging Face Hub API to search across 500,000+
public datasets dynamically — no hardcoded list, no domain limitations.

Metric verification remains rule-based using VALID_METRIC_METHOD_PAIRS.

Usage:
    from tools.feasibility_checker import check_feasibility
    passed, notes = check_feasibility(plan)
"""

import re
import requests
from functools import lru_cache
from models.schemas import ResearchPlan

# ── Hugging Face Dataset Search ───────────────────────────────────────────────

HF_SEARCH_URL = "https://huggingface.co/api/datasets"
HF_TIMEOUT    = 6  # seconds — fast enough for pipeline use

@lru_cache(maxsize=256)
def _search_hf_datasets(query: str) -> bool:
    """
    Search the Hugging Face Hub for a dataset matching the query string.

    Uses the public HF API — no authentication required.
    Results are cached so repeated queries for the same dataset
    don't make extra network calls.

    Args:
        query: Dataset name or description to search for.

    Returns:
        True if at least one matching dataset is found, False otherwise.
    """
    try:
        response = requests.get(
            HF_SEARCH_URL,
            params={"search": query, "limit": 5},
            timeout=HF_TIMEOUT
        )
        if response.status_code == 200:
            results = response.json()
            return len(results) > 0
    except requests.exceptions.RequestException:
        pass
    return False


def _dataset_exists_online(dataset_name: str) -> bool:
    """
    Check whether a dataset name can be verified via the HF Hub.

    Tries multiple search strategies:
      1. Full dataset name as-is
      2. First meaningful phrase (up to 4 words) of the name
      3. Key noun phrases extracted from the name

    Returns True if any search finds results.
    """
    name = dataset_name.strip()

    # Skip clearly non-dataset descriptions
    if len(name.split()) > 12:
        # Too descriptive — likely a hallucinated custom dataset description
        # Still try a shortened version
        name = " ".join(name.split()[:6])

    # Strategy 1: search full name (lowercased)
    if _search_hf_datasets(name.lower()):
        return True

    # Strategy 2: strip parenthetical abbreviations like "(PDB)" and search core name
    core = re.sub(r'\(.*?\)', '', name).strip()
    if core != name and _search_hf_datasets(core.lower()):
        return True

    # Strategy 3: first 3-4 meaningful words
    words = [w for w in name.split() if len(w) > 2]
    if len(words) >= 3:
        short = " ".join(words[:4]).lower()
        if _search_hf_datasets(short):
            return True

    return False


# ── Valid metric method pairs (rule-based, kept lean) ─────────────────────────

VALID_METRIC_METHOD_PAIRS: dict[str, list[str]] = {
    "classification": [
        "accuracy", "f1", "precision", "recall", "auc", "roc",
        "top-1", "top-5", "balanced accuracy", "cohen kappa", "mcc",
    ],
    "continual": [
        "accuracy", "forgetting", "backward transfer", "forward transfer",
        "intransigence", "plasticity", "stability", "average accuracy",
        "final accuracy", "knowledge retention",
    ],
    "generation": [
        "bleu", "rouge", "meteor", "bertscore", "cider",
        "perplexity", "mauve", "fid", "psnr", "ssim", "lpips",
    ],
    "retrieval": [
        "mrr", "ndcg", "recall@k", "precision@k", "map", "hit rate",
    ],
    "regression": [
        "mse", "mae", "rmse", "r2", "mape",
    ],
    "reinforcement": [
        "cumulative reward", "win rate", "episode return",
        "average return", "success rate", "sample efficiency",
    ],
    "clustering": [
        "nmi", "ari", "silhouette", "purity",
    ],
    "few-shot": [
        "accuracy", "f1", "few-shot accuracy", "n-way k-shot accuracy",
    ],
    "hallucination": [
        "accuracy", "f1", "precision", "recall",
        "hallucination rate", "faithfulness", "factuality",
        "factual accuracy", "hallucination detection rate",
        "factscore", "bert score", "bertscore",
        "entailment score", "consistency score",
    ],
    "detection": [
        "accuracy", "f1", "precision", "recall", "auc",
        "hallucination rate", "detection rate", "false positive rate",
        "map", "mean average precision",
    ],
    "segmentation": [
        "dice", "iou", "intersection over union", "mean iou",
        "pixel accuracy", "f1", "hausdorff distance",
        "dice coefficient", "jaccard index",
    ],
    "translation": [
        "bleu", "rouge", "meteor", "bertscore",
        "wer", "word error rate", "translation accuracy",
        "chrf", "ter", "comet",
    ],
    "sign": [
        "accuracy", "f1", "precision", "recall",
        "word error rate", "wer", "recognition accuracy",
        "signer-independent accuracy", "top-1 accuracy", "bleu",
    ],
    "quantum": [
        "fidelity", "error rate", "logical error rate",
        "quantum volume", "gate fidelity",
        "error correction threshold", "accuracy",
    ],
    "medical": [
        "accuracy", "f1", "auc", "auc-roc", "auroc",
        "dice", "iou", "sensitivity", "specificity",
        "cohen kappa", "mcc", "balanced accuracy",
    ],
    "restoration": [
        "psnr", "ssim", "fid", "lpips", "mse", "mae", "rmse",
        "peak signal-to-noise ratio", "structural similarity",
    ],
    "fact": [
        "accuracy", "f1", "precision", "recall",
        "factual accuracy", "factscore", "faithfulness score",
    ],
}

# All valid metrics flattened — used as permissive fallback
_ALL_VALID_METRICS = list({m for metrics in VALID_METRIC_METHOD_PAIRS.values() for m in metrics})


# ── Main check function ───────────────────────────────────────────────────────

def check_feasibility(plan: ResearchPlan) -> tuple[bool, str]:
    """
    Check whether a ResearchPlan's dataset and evaluation metric are realistic.

    Dataset check  : Searches the Hugging Face Hub API dynamically.
                     Accepts any publicly available dataset — no hardcoded list.
    Metric check   : Rule-based match against VALID_METRIC_METHOD_PAIRS.
                     Falls back to checking against all known metrics.

    Args:
        plan: A ResearchPlan to evaluate.

    Returns:
        Tuple of (passed: bool, notes: str).
    """
    dataset_lower = plan.dataset.lower()
    method_lower  = plan.proposed_method.lower()
    metric_lower  = plan.evaluation_metric.lower()
    notes = []

    # ── Check 1: Dataset exists online (HF Hub search) ────────────
    dataset_known = _dataset_exists_online(plan.dataset)

    if not dataset_known:
        notes.append(
            f"Dataset '{plan.dataset}' could not be verified on the "
            f"Hugging Face Hub. Use a publicly available, named benchmark "
            f"dataset (e.g. from huggingface.co/datasets)."
        )

    # ── Check 2: Metric is appropriate for method type ────────────
    metric_valid        = False
    matched_method_type = None

    for method_type, valid_metrics in VALID_METRIC_METHOD_PAIRS.items():
        if method_type in method_lower:
            matched_method_type = method_type
            if any(m in metric_lower for m in valid_metrics):
                metric_valid = True
            break

    if matched_method_type is None:
        # Permissive fallback: check against all known metrics
        if any(m in metric_lower for m in _ALL_VALID_METRICS):
            metric_valid = True
        else:
            notes.append(
                f"Evaluation metric '{plan.evaluation_metric}' could not be "
                f"matched to a recognised metric type. Use a standard metric "
                f"such as accuracy, F1, BLEU, ROUGE, MSE, IoU, PSNR, or WER."
            )
    elif not metric_valid:
        notes.append(
            f"Evaluation metric '{plan.evaluation_metric}' does not appear "
            f"appropriate for a '{matched_method_type}' method. "
            f"Expected one of: {VALID_METRIC_METHOD_PAIRS[matched_method_type]}."
        )

    passed    = dataset_known and metric_valid
    notes_str = " | ".join(notes) if notes else "All feasibility checks passed."
    return passed, notes_str
