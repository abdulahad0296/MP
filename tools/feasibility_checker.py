"""
tools/feasibility_checker.py
-----------------------------
5-component feasibility checker for the Agentic Research Planning Framework.
Phase B upgrade: expanded from 2 checks to 5 independent components.

Components:
    1. Data Availability      — HF Hub API search (500,000+ datasets)
    2. Metric Validity        — rule-based: 43 method types, 199 metrics
    3. Computational Feasibility — blacklist of prohibitively expensive operations
    4. Methodological Practicality — whitelist of recognised ML/AI techniques
    5. Time Feasibility       — flags for unrealistic research scope

Usage:
    from tools.feasibility_checker import check_feasibility
    passed, notes, components = check_feasibility(plan)
    # components is a dict with per-component (passed, note) tuples
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


def get_topic_datasets(topic: str, limit: int = 8) -> list[str]:
    """
    Query the HF Hub for datasets relevant to a topic and return their names.

    Used by the Planner Agent to ground dataset suggestions in real, verifiable
    benchmarks before plan generation — preventing the LLM from hallucinating
    dataset names that will later fail the data_availability check.

    Args:
        topic : Research topic string (e.g. "sign language recognition").
        limit : Maximum number of dataset names to return.

    Returns:
        List of dataset id strings from HF Hub (may be empty if unreachable).
    """
    try:
        response = requests.get(
            HF_SEARCH_URL,
            params={"search": topic, "limit": limit},
            timeout=HF_TIMEOUT
        )
        if response.status_code == 200:
            results = response.json()
            return [r["id"] for r in results if "id" in r]
    except requests.exceptions.RequestException:
        pass
    return []


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
    "sign language": [
        "accuracy", "f1", "precision", "recall",
        "word error rate", "wer", "recognition accuracy",
        "signer-independent accuracy", "top-1 accuracy", "top-5 accuracy",
        "bleu", "rouge", "auc", "auc-roc", "auroc",
        "mean average precision", "map", "mcc",
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
    # ── Quantum computing ──────────────────────────────────────────
    "quantum": [
        "fidelity", "error rate", "logical error rate",
        "quantum volume", "gate fidelity", "circuit fidelity",
        "error correction threshold", "error correction fidelity",
        "entanglement fidelity", "process fidelity",
        "syndrome detection rate", "code distance",
        "accuracy", "f1",
    ],
    # ── Bioinformatics / computational biology ─────────────────────
    "protein": [
        "rmsd", "gdt-ts", "gdt-ha", "tm-score",
        "mcc", "aupr", "auc", "sensitivity", "specificity",
        "accuracy", "f1", "precision", "recall",
    ],
    "molecular": [
        "rmsd", "mae", "mse", "r2",
        "auc", "aupr", "mcc",
        "validity", "uniqueness", "novelty",
        "fcd", "nspdk",
    ],
    "drug": [
        "auc", "aupr", "mcc", "accuracy", "f1",
        "rmsd", "binding affinity", "enrichment factor",
    ],
    "biological": [
        "auc", "aupr", "mcc", "f1", "accuracy",
        "sensitivity", "specificity", "precision", "recall",
    ],
    # ── Speech and audio ───────────────────────────────────────────
    "speech": [
        "wer", "cer", "mer", "word error rate",
        "character error rate", "match error rate",
        "accuracy", "f1", "pesq", "stoi", "snr",
    ],
    "audio": [
        "wer", "cer", "accuracy", "f1",
        "pesq", "stoi", "snr", "si-sdr",
        "map", "auc",
    ],
    "recognition": [
        "accuracy", "f1", "precision", "recall",
        "wer", "cer", "top-1 accuracy", "top-5 accuracy",
        "map", "auc",
    ],
    # ── Graph and network ──────────────────────────────────────────
    "graph": [
        "accuracy", "f1", "auc", "aupr",
        "hits@k", "mrr", "mean rank", "map",
        "ndcg", "recall@k", "precision@k",
        "nmi", "ari",
    ],
    "link": [
        "hits@k", "mrr", "mean rank",
        "auc", "aupr", "map",
        "recall@k", "precision@k",
    ],
    "node": [
        "accuracy", "f1", "micro-f1", "macro-f1",
        "auc", "aupr",
    ],
    "knowledge": [
        "hits@k", "mrr", "mean rank",
        "accuracy", "f1", "map",
    ],
    # ── Time series and forecasting ────────────────────────────────
    "forecasting": [
        "mse", "mae", "rmse", "mape", "smape",
        "r2", "dtw", "crps",
    ],
    "time": [
        "mse", "mae", "rmse", "mape", "smape",
        "accuracy", "f1", "auc",
        "dtw", "r2",
    ],
    "anomaly": [
        "auc", "aupr", "f1", "precision", "recall",
        "average precision", "auc-roc",
        "accuracy", "mcc",
    ],
    # ── Recommendation systems ─────────────────────────────────────
    "recommendation": [
        "ndcg", "map", "mrr", "hit rate",
        "recall@k", "precision@k",
        "mse", "mae", "rmse",
        "coverage", "diversity",
    ],
    "collaborative": [
        "ndcg", "map", "mrr",
        "recall@k", "precision@k",
        "mse", "mae", "rmse",
    ],
    # ── NLP — extended ────────────────────────────────────────────
    "summarization": [
        "rouge", "rouge-1", "rouge-2", "rouge-l",
        "bleu", "meteor", "bertscore",
        "faithfulness", "factuality",
    ],
    "question": [
        "exact match", "f1", "accuracy",
        "bleu", "rouge", "bertscore",
    ],
    "dialogue": [
        "bleu", "rouge", "meteor", "bertscore",
        "perplexity", "distinct-1", "distinct-2",
        "f1", "accuracy",
    ],
    "machine": [
        "bleu", "sacrebleu", "chrf", "comet",
        "ter", "meteor", "bertscore",
        "wer",
    ],
    # ── Security and privacy ───────────────────────────────────────
    "privacy": [
        "accuracy", "f1", "auc",
        "attack success rate", "membership inference accuracy",
        "differential privacy epsilon", "privacy budget",
        "reconstruction error",
    ],
    "adversarial": [
        "accuracy", "robust accuracy", "attack success rate",
        "certified accuracy", "f1", "auc",
    ],
    "byzantine": [
        "accuracy", "f1", "attack success rate",
        "detection rate", "convergence rate",
        "robust accuracy",
    ],
    "poisoning": [
        "accuracy", "attack success rate",
        "detection rate", "f1",
        "backdoor success rate",
    ],
    # ── Robotics and RL extended ───────────────────────────────────
    "robot": [
        "success rate", "cumulative reward",
        "average return", "episode return",
        "task completion rate", "collision rate",
        "accuracy", "mse",
    ],
    "planning": [
        "success rate", "plan length",
        "optimality ratio", "accuracy",
        "cumulative reward",
    ],
}

# All valid metrics flattened — used as permissive fallback
_ALL_VALID_METRICS = list({m for metrics in VALID_METRIC_METHOD_PAIRS.values() for m in metrics})


# ── Phase B: New rule sets ───────────────────────────────────────────────────

# Task B.2 — Computational feasibility blacklist
# Phrases in proposed_method indicating the plan requires unrealistic compute
COMPUTE_BLACKLIST: list[str] = [
    "train from scratch on imagenet",
    "training gpt from scratch",
    "training a large language model from scratch",
    "pretraining on the entire web",
    "trillion parameter",
    "requires a supercomputer",
    "petabyte scale",
    "full fine-tuning of llama",
    "training on 1 billion",
    "10000 gpu hours",
    "requires hundreds of gpus",
    "train a foundation model from scratch",
]

# Minimal safety-net fallback — only used if the LLM is unreachable.
_METHOD_FALLBACK: list[str] = [
    "fine-tuning", "transfer learning", "transformer", "attention",
    "regularization", "gradient", "embedding", "encoder", "decoder",
    "generative", "contrastive", "federated", "reinforcement", "distillation",
]

# Groq client for LLM-based phrase extraction — imported lazily to avoid
# circular imports and only used in check_methodological_practicality()
_groq_client = None

def _get_groq_client():
    """Lazy initialisation of Groq client."""
    global _groq_client
    if _groq_client is None:
        try:
            from groq import Groq
            import config
            _groq_client = Groq(api_key=config.GROQ_API_KEY)
        except Exception:
            pass
    return _groq_client

# Task B.6 — Time feasibility scope flags
# Phrases indicating the plan is unrealistic for a research timeframe
SCOPE_FLAGS: list[str] = [
    "build a new large language model",
    "create a new dataset from scratch",
    "conduct human trials",
    "requires decades",
    "multi-year study",
    "clinical trial",
    "requires new hardware",
    "invent a new algorithm from scratch",
    "collect data from millions of users",
    "deploy to production at scale",
    "build a complete autonomous system",
    "replace existing infrastructure",
]


# ── Phase B: Three new check functions ───────────────────────────────────────

def check_computational_feasibility(plan: ResearchPlan) -> tuple[bool, str]:
    """
    Check whether the proposed method requires prohibitively expensive compute.

    Scans proposed_method for phrases from COMPUTE_BLACKLIST. Plans that
    require training trillion-parameter models from scratch, petabyte-scale
    data, or hundreds of GPUs are flagged as computationally infeasible for
    a standard research setting.

    Returns:
        Tuple of (passed: bool, notes: str).
    """
    method_lower = plan.proposed_method.lower()
    for phrase in COMPUTE_BLACKLIST:
        if phrase in method_lower:
            return (
                False,
                f"Computational feasibility: proposed method appears to require "
                f"prohibitive compute ('{phrase}'). Scope to single-GPU or "
                f"parameter-efficient approaches."
            )
    return (True, "Computational feasibility: OK")


def check_methodological_practicality(plan: ResearchPlan) -> tuple:
    """
    Check whether the proposed method uses a recognised ML/AI technique.

    Uses a single LLM call to both extract the core technique name from the
    method description AND verify whether it is a real, published ML/AI method.
    This is more reliable than keyword matching or external API search because:
    - The LLM understands context and ignores filler language naturally
    - The LLM has knowledge of the ML literature built in
    - No hardcoded whitelist needed — works for any domain

    Algorithm:
        1. LLM extracts the technique AND judges if it is a real published method.
        2. Returns is_recognised=true/false with a reason.
        3. Falls back to _METHOD_FALLBACK list if LLM is unreachable.
    """
    client = _get_groq_client()

    if client is None:
        method_lower = plan.proposed_method.lower()
        if any(term in method_lower for term in _METHOD_FALLBACK):
            return (True, "Methodological practicality: OK (fallback check)")
        return (
            False,
            "Methodological practicality: LLM unavailable for method verification. "
            "Use a named ML technique."
        )

    try:
        import json as _json
        import config as _config
        response = client.chat.completions.create(
            model=_config.LLM_MODEL,
            max_tokens=200,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an ML research expert. Given a proposed research method "
                        "description, determine whether it uses a real, recognised ML/AI "
                        "technique that appears in published research. "
                        "Return ONLY a JSON object with these keys: "
                        "technique (the core method name extracted, string), "
                        "is_recognised (true if it is a real published ML technique, false if vague or fabricated), "
                        "reason (one sentence explanation). "
                        "Examples of recognised techniques: fine-tuning, contrastive learning, "
                        "energy-based models, metalearning, variational autoencoder, "
                        "sparse routing networks, elastic weight consolidation, "
                        "contrastive divergence, generative replay, knowledge distillation. "
                        "Examples of unrecognised: a completely undefined approach, "
                        "a mysterious novel framework with no grounding in literature."
                    )
                },
                {
                    "role": "user",
                    "content": f"Method: {plan.proposed_method}"
                }
            ]
        )
        raw = response.choices[0].message.content or ""
        parsed = _json.loads(raw)

        technique     = str(parsed.get("technique", "unknown"))
        is_recognised = bool(parsed.get("is_recognised", False))
        reason        = str(parsed.get("reason", ""))

        if is_recognised:
            return (True, f"Methodological practicality: OK ({technique})")
        else:
            return (
                False,
                f"Methodological practicality: '{technique}' does not appear to be "
                f"a recognised ML/AI technique. {reason}"
            )

    except Exception as e:
        method_lower = plan.proposed_method.lower()
        if any(term in method_lower for term in _METHOD_FALLBACK):
            return (True, "Methodological practicality: OK (fallback check)")
        return (
            False,
            f"Methodological practicality: verification failed. "
            f"Use a named ML technique such as fine-tuning or contrastive learning."
        )

def check_time_feasibility(plan: ResearchPlan) -> tuple[bool, str]:
    """
    Check whether the proposed research scope is completable in a reasonable
    research timeframe (e.g. MSc or PhD project duration).

    Scans both proposed_method and research_question for phrases from
    SCOPE_FLAGS that indicate multi-year, clinical, or otherwise unrealistic
    scope.

    Returns:
        Tuple of (passed: bool, notes: str).
    """
    combined = (plan.proposed_method + " " + plan.research_question).lower()
    for phrase in SCOPE_FLAGS:
        if phrase in combined:
            return (
                False,
                f"Time feasibility: research scope appears unrealistic for a "
                f"standard research timeframe ('{phrase}'). Narrow the scope "
                f"to a focused experiment or pilot study."
            )
    return (True, "Time feasibility: OK")


# ── Main check function ───────────────────────────────────────────────────────

def check_feasibility(plan: ResearchPlan) -> tuple[bool, str, dict]:
    """
    Run all 5 feasibility components against a ResearchPlan.

    Components (Phase B upgrade):
        1. Data availability      — HF Hub API search
        2. Metric validity        — rule-based metric/method matching
        3. Computational feasibility — blacklist of expensive operations
        4. Methodological practicality — whitelist of known techniques
        5. Time feasibility       — scope flag detection

    Each component returns (passed: bool, note: str) independently.
    The overall result passes only if ALL 5 components pass.

    Args:
        plan: A ResearchPlan to evaluate.

    Returns:
        Tuple of (passed: bool, notes: str, components: dict).
        - passed     : True only if all 5 components pass
        - notes      : Concatenated failure notes (empty string if all pass)
        - components : Per-component dict of {name: (passed, note)}
    """
    method_lower = plan.proposed_method.lower()
    metric_lower = plan.evaluation_metric.lower()
    components   = {}

    # ── Component 1: Data availability (HF Hub) ───────────────────
    dataset_ok = _dataset_exists_online(plan.dataset)
    components["data_availability"] = (
        dataset_ok,
        "Data availability: OK" if dataset_ok else
        f"Dataset '{plan.dataset}' could not be verified on the Hugging Face Hub. "
        f"Use a publicly available, named benchmark dataset."
    )

    # ── Component 2: Metric validity ──────────────────────────────
    metric_valid        = False
    matched_method_type = None

    for method_type, valid_metrics in VALID_METRIC_METHOD_PAIRS.items():
        if method_type in method_lower:
            matched_method_type = method_type
            if any(m in metric_lower for m in valid_metrics):
                metric_valid = True
            break

    if matched_method_type is None:
        if any(m in metric_lower for m in _ALL_VALID_METRICS):
            metric_valid = True
        metric_note = (
            "Metric validity: OK" if metric_valid else
            f"Evaluation metric '{plan.evaluation_metric}' could not be matched "
            f"to a recognised metric type. Use accuracy, F1, BLEU, ROUGE, MSE, "
            f"IoU, PSNR, WER, etc."
        )
    else:
        metric_note = (
            "Metric validity: OK" if metric_valid else
            f"Evaluation metric '{plan.evaluation_metric}' does not appear "
            f"appropriate for a '{matched_method_type}' method. "
            f"Expected one of: {VALID_METRIC_METHOD_PAIRS[matched_method_type]}."
        )
    components["metric_validity"] = (metric_valid, metric_note)

    # ── Component 3: Computational feasibility ────────────────────
    components["computational_feasibility"] = check_computational_feasibility(plan)

    # ── Component 4: Methodological practicality ──────────────────
    components["methodological_practicality"] = check_methodological_practicality(plan)

    # ── Component 5: Time feasibility ─────────────────────────────
    components["time_feasibility"] = check_time_feasibility(plan)

    # ── Aggregate result ──────────────────────────────────────────
    failed_notes = [
        note for name, (ok, note) in components.items()
        if not ok and "OK" not in note
    ]
    passed    = all(ok for _, (ok, _) in components.items())
    notes_str = " | ".join(failed_notes) if failed_notes else "All feasibility checks passed."

    return passed, notes_str, components