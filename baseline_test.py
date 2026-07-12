"""
baseline_test.py
----------------
Phase C — Formal Baseline Evaluation.

Runs the direct LLM baseline (no pipeline) across all 5 evaluation topics
and scores each idea against a 4-criterion rubric. Saves structured results
to outputs/baseline_all_topics.json for comparison with pipeline results.

Baseline: direct LLM prompting with no literature retrieval, no gap
identification, no novelty scoring, and no feasibility validation.

Usage:
    python baseline_test.py
"""

import json
import os
import time
from datetime import datetime
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL

client = Groq(api_key=GROQ_API_KEY)

# ── Evaluation topics — same 5 as pipeline runs ───────────────────────────────
TOPICS = [
    "federated learning privacy",
    "large language models hallucination detection",
    "quantum computing error correction",
    "sign language recognition Indian regional languages",
    "deep learning",
]

# ── 4-criterion scoring rubric ────────────────────────────────────────────────
RUBRIC = [
    "Is the dataset a real, named, publicly available benchmark? (not a vague description)",
    "Is the evaluation metric a standard named metric (accuracy, F1, BLEU, WER, etc.)?",
    "Is the proposed method grounded in an existing, named ML/AI technique?",
    "Is the experimental setup specific and testable (not vague or hallucinated)?",
]


def generate_baseline_ideas(topic: str) -> list:
    """Generate 3 research ideas for a topic using direct LLM prompting."""
    prompt = (
        f"Generate 3 novel research ideas about: {topic}\n\n"
        "For each idea provide:\n"
        "- research_question: the core research question\n"
        "- proposed_method: the method or approach\n"
        "- dataset: the specific dataset to be used\n"
        "- evaluation_metric: how success will be measured\n\n"
        "Return ONLY a valid JSON object with a single key 'ideas' containing "
        "an array of 3 objects. Each object must have exactly these keys: "
        "research_question, proposed_method, dataset, evaluation_metric."
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=1024,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a research assistant. Return only valid JSON."},
            {"role": "user",   "content": prompt}
        ]
    )
    raw    = response.choices[0].message.content or ""
    parsed = json.loads(raw)
    return parsed.get("ideas", [])


def score_ideas_with_llm(ideas: list, topic: str) -> list:
    """
    Score each baseline idea against the 4-criterion rubric using the LLM.

    Returns a list of scored ideas, each with a per-criterion breakdown
    and total score (0–4).
    """
    rubric_text = "\n".join(f"{i+1}. {r}" for i, r in enumerate(RUBRIC))

    scored = []
    for idea in ideas:
        prompt = (
            f"Score this research idea on 4 criteria. "
            f"Return ONLY a JSON object with key 'scores' containing a list of "
            f"4 integers (0 or 1 each), one per criterion.\n\n"
            f"Criteria:\n{rubric_text}\n\n"
            f"Research idea:\n"
            f"  Dataset: {idea.get('dataset','')}\n"
            f"  Metric: {idea.get('evaluation_metric','')}\n"
            f"  Method: {idea.get('proposed_method','')}\n"
            f"  Question: {idea.get('research_question','')}"
        )

        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                max_tokens=100,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a research evaluator. Return only valid JSON."},
                    {"role": "user",   "content": prompt}
                ]
            )
            raw    = response.choices[0].message.content or ""
            parsed = json.loads(raw)
            scores = parsed.get("scores", [0, 0, 0, 0])
            # Ensure exactly 4 scores, clamped to 0 or 1
            scores = [min(1, max(0, int(s))) for s in (scores + [0, 0, 0, 0])[:4]]
        except Exception:
            scores = [0, 0, 0, 0]

        scored.append({
            **idea,
            "criterion_scores": scores,
            "total_score": sum(scores),
            "criteria": {RUBRIC[i]: bool(scores[i]) for i in range(4)},
        })

    return scored


def run_baseline_for_topic(topic: str) -> dict:
    """Run baseline generation and scoring for a single topic."""
    print(f"\n  Generating ideas for: '{topic}'...")
    ideas = generate_baseline_ideas(topic)
    print(f"  Generated {len(ideas)} idea(s). Scoring...")
    scored = score_ideas_with_llm(ideas, topic)

    avg_score = sum(i["total_score"] for i in scored) / len(scored) if scored else 0
    print(f"  Average rubric score: {avg_score:.2f}/4.00")

    return {
        "topic": topic,
        "timestamp": datetime.now().isoformat(),
        "model": LLM_MODEL,
        "prompt_type": "direct — no pipeline",
        "ideas": scored,
        "summary": {
            "total_ideas": len(scored),
            "avg_rubric_score": round(avg_score, 2),
            "max_possible": 4,
            "criterion_pass_rates": {
                RUBRIC[i]: sum(1 for idea in scored if idea["criterion_scores"][i]) / len(scored)
                for i in range(4)
            } if scored else {},
        }
    }


def main():
    print("=" * 65)
    print("Phase C — Formal Baseline Evaluation")
    print("Direct LLM prompting across all 5 evaluation topics")
    print("=" * 65)

    all_results = []

    for i, topic in enumerate(TOPICS, 1):
        print(f"\n[{i}/{len(TOPICS)}] Topic: {topic}")
        result = run_baseline_for_topic(topic)
        all_results.append(result)
        # Brief pause to respect rate limits
        if i < len(TOPICS):
            time.sleep(2)

    # ── Save all results ──────────────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/baseline_all_topics.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"runs": all_results}, f, indent=2, ensure_ascii=False)

    # ── Print summary table ───────────────────────────────────────
    print("\n" + "=" * 65)
    print("Baseline Summary — Average Rubric Scores by Topic")
    print("=" * 65)
    print(f"{'Topic':<45} {'Avg Score':>10} {'Max':>5}")
    print("-" * 65)
    for r in all_results:
        print(f"  {r['topic'][:43]:<43} {r['summary']['avg_rubric_score']:>10.2f} {r['summary']['max_possible']:>5}")

    print(f"\nResults saved to: {output_path}")
    print("Run compare_results.py to generate the full comparison table.")
    print("=" * 65)


if __name__ == "__main__":
    main()