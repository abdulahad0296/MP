"""
main.py
-------
Pipeline orchestrator for the Agentic Research Planning Framework.

Wires all three agents into a single end-to-end pipeline:
    1. Librarian Agent  — retrieve papers + extract concepts + identify gaps
    2. Planner Agent    — generate candidate research plans per gap
    3. Reviewer Agent   — score novelty + check feasibility + generate outputs
    4. Revision loop    — re-plan rejected proposals (up to MAX_REVISION_ATTEMPTS)
    5. Save + display   — write results.json and print a clean summary

Usage:
    python main.py --topic "federated learning privacy"
    python main.py --topic "vision transformers for medical imaging"
    python main.py --topic "graph neural networks drug discovery"
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List

import config
from agents import librarian_agent, planner_agent, reviewer_agent
from models.schemas import ReviewResult


# ── Save results ──────────────────────────────────────────────────────────────

def save_results(results: List[ReviewResult], topic: str, path: str = None) -> str:
    """
    Append this run's results to the cumulative JSON log file.

    Each run is stored as a separate entry in a top-level "runs" list.
    Existing runs are never overwritten — every run is preserved for
    evaluation and comparison across topics.

    File structure:
        {
            "runs": [
                { "topic": ..., "timestamp": ..., "summary": ..., "results": [...] },
                { "topic": ..., "timestamp": ..., "summary": ..., "results": [...] },
                ...
            ]
        }

    Args:
        results : List of ReviewResult objects from the Reviewer Agent.
        topic   : The research topic used for this run.
        path    : Output file path. Defaults to config.OUTPUT_PATH.

    Returns:
        The path where results were saved.
    """
    if path is None:
        path = config.OUTPUT_PATH

    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Load existing data if file already exists
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
                # Handle old format (single run object, no "runs" key)
                if "runs" not in existing:
                    existing = {"runs": [existing]}
            except json.JSONDecodeError:
                existing = {"runs": []}
    else:
        existing = {"runs": []}

    # Build this run's entry
    run_entry = {
        "topic": topic,
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_reviewed": len(results),
            "accepted": sum(1 for r in results if r.accepted),
            "rejected": sum(1 for r in results if not r.accepted),
        },
        "results": [
            {
                "accepted": r.accepted,
                "novelty_score": r.novelty_score,
                "feasibility_passed": r.feasibility_passed,
                "feasibility_notes": r.feasibility_notes,
                "suggested_title": r.suggested_title,
                "research_direction": r.research_direction,
                "experimental_blueprint": r.experimental_blueprint,
                "plan": {
                    "research_question": r.plan.research_question,
                    "proposed_method": r.plan.proposed_method,
                    "dataset": r.plan.dataset,
                    "evaluation_metric": r.plan.evaluation_metric,
                    "source_gap": r.plan.source_gap.description if r.plan.source_gap else "",
                },
            }
            for r in results
        ],
    }

    # Append and save
    existing["runs"].append(run_entry)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    run_number = len(existing["runs"])
    return path, run_number


# ── Display results ───────────────────────────────────────────────────────────

def display_results(results: List[ReviewResult], topic: str) -> None:
    """
    Print a clean, readable summary of accepted research proposals to the terminal.
    """
    accepted = [r for r in results if r.accepted]
    rejected = [r for r in results if not r.accepted]

    width = 68

    print("\n" + "=" * width)
    print("  AGENTIC RESEARCH PLANNING FRAMEWORK — RESULTS")
    print("=" * width)
    print(f"  Topic    : {topic}")
    print(f"  Reviewed : {len(results)} plans")
    print(f"  Accepted : {len(accepted)}  |  Rejected : {len(rejected)}")
    print("=" * width)

    if not accepted:
        print("\n  No plans accepted in this run.")
        print("  Try running again — gap identification varies between runs.")
        print("  You can also lower NOVELTY_THRESHOLD in config.py.")
    else:
        print(f"\n  {len(accepted)} ACCEPTED RESEARCH PROPOSAL(S)\n")

        for i, r in enumerate(accepted, 1):
            print(f"  {'─' * (width - 2)}")
            print(f"  Proposal {i} of {len(accepted)}")
            print(f"  {'─' * (width - 2)}")
            print(f"\n  Title     : {r.suggested_title}")
            print(f"\n  Direction : {r.research_direction}")
            print(f"\n  Question  : {r.plan.research_question}")
            print(f"\n  Method    : {r.plan.proposed_method}")
            print(f"\n  Dataset   : {r.plan.dataset}")
            print(f"  Metric    : {r.plan.evaluation_metric}")
            print(f"\n  Blueprint :")

            # Word-wrap blueprint at 62 chars
            words = r.experimental_blueprint.split()
            line, lines = [], []
            for word in words:
                if sum(len(w) + 1 for w in line) + len(word) > 62:
                    lines.append(" ".join(line))
                    line = [word]
                else:
                    line.append(word)
            if line:
                lines.append(" ".join(line))
            for bl in lines:
                print(f"    {bl}")

            print(f"\n  Novelty Score : {r.novelty_score:.2f} / 10.00")
            print(f"  Feasibility   : PASS")
            print(f"  Source Gap    : {r.plan.source_gap.description[:65] if r.plan.source_gap else 'N/A'}...")
            print()

    if rejected:
        print(f"  {'─' * (width - 2)}")
        print(f"\n  {len(rejected)} REJECTED PLAN(S)\n")
        for r in rejected:
            reasons = []
            if r.novelty_score < config.NOVELTY_THRESHOLD:
                reasons.append(f"novelty {r.novelty_score:.2f} < threshold {config.NOVELTY_THRESHOLD}")
            if not r.feasibility_passed:
                reasons.append(f"feasibility: {r.feasibility_notes[:60]}")
            print(f"  - {r.plan.research_question[:60]}...")
            print(f"    Reason: {' | '.join(reasons)}")
        print()

    print("=" * width)


# ── Pipeline orchestrator ─────────────────────────────────────────────────────

def run_pipeline(topic: str) -> List[ReviewResult]:
    """
    Run the full Agentic Research Planning pipeline for a given topic.

    Pipeline:
        Step 1+2 : Librarian Agent — retrieve papers, extract concepts, find gaps
        Step 3   : Planner Agent   — generate 3 candidate plans per gap
        Step 4+5 : Reviewer Agent  — score novelty + check feasibility
        Revision : Re-plan rejected proposals (up to MAX_REVISION_ATTEMPTS)
        Output   : Save to JSON + display clean summary

    Args:
        topic: Research topic string entered by the user.

    Returns:
        List of all ReviewResult objects (accepted and rejected).
    """
    print(f"\n[pipeline] Starting research planning for topic: '{topic}'")
    print(f"[pipeline] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ── Step 1 + 2: Librarian Agent ───────────────────────────────
    papers, gaps = librarian_agent.run(topic)

    if not gaps:
        print("[pipeline] ERROR: No research gaps identified. Cannot continue.")
        print("[pipeline] Try a more specific topic or re-run — gap detection varies.")
        return []

    # ── Step 3: Planner Agent ─────────────────────────────────────
    plans = planner_agent.run(gaps, papers)

    if not plans:
        print("[pipeline] ERROR: No research plans generated. Cannot continue.")
        return []

    # ── Step 4 + 5: Reviewer Agent ────────────────────────────────
    results = reviewer_agent.run(plans, papers)

    # ── Revision loop ─────────────────────────────────────────────
    # Re-submit rejected plans to the Planner for a second attempt.
    # IMPORTANT: gaps_to_retry is fixed from the ORIGINAL rejection list
    # only — it does not grow across attempts. This prevents exponential
    # explosion of API calls across revision rounds.
    rejected = [r for r in results if not r.accepted]

    if rejected:
        print(f"\n[pipeline] Revision loop: {len(rejected)} rejected plan(s) will be re-planned.")

    # Deduplicate gaps from original rejections — retry each gap once only
    seen_gap_descriptions = set()
    unique_gaps_to_retry = []
    for r in rejected:
        if r.plan.source_gap and r.plan.source_gap.description not in seen_gap_descriptions:
            seen_gap_descriptions.add(r.plan.source_gap.description)
            unique_gaps_to_retry.append(r.plan.source_gap)

    for attempt in range(config.MAX_REVISION_ATTEMPTS):
        if not unique_gaps_to_retry:
            break

        print(f"[pipeline] Revision attempt {attempt + 1} of {config.MAX_REVISION_ATTEMPTS} "
              f"({len(unique_gaps_to_retry)} unique gap(s))...")

        revised_plans   = planner_agent.run(unique_gaps_to_retry, papers)
        revised_results = reviewer_agent.run(revised_plans, papers)

        newly_accepted  = [r for r in revised_results if r.accepted]
        still_rejected  = [r for r in revised_results if not r.accepted]

        results += newly_accepted

        # Only keep gaps that are still producing rejected plans
        accepted_gap_descs = {r.plan.source_gap.description for r in newly_accepted if r.plan.source_gap}
        unique_gaps_to_retry = [
            g for g in unique_gaps_to_retry
            if g.description not in accepted_gap_descs
        ]

        print(f"[pipeline] Revision {attempt + 1}: {len(newly_accepted)} newly accepted, "
              f"{len(still_rejected)} still rejected.")

    # ── Save + display ────────────────────────────────────────────
    save_path, run_number = save_results(results, topic)
    print(f"\n[pipeline] Results saved to: {save_path} (run #{run_number})")

    display_results(results, topic)

    accepted_count = sum(1 for r in results if r.accepted)
    print(f"[pipeline] Done. {accepted_count} accepted proposal(s) for topic: '{topic}'\n")

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Agentic Research Planning Framework — MSc AI & ML Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --topic "federated learning privacy"
  python main.py --topic "vision transformers medical imaging"
  python main.py --topic "graph neural networks drug discovery"
  python main.py --topic "continual learning catastrophic forgetting"
        """
    )
    parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help="Research topic to generate proposals for (e.g. 'federated learning privacy')"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=f"Output JSON path (default: {config.OUTPUT_PATH})"
    )

    args = parser.parse_args()

    if not args.topic.strip():
        print("ERROR: --topic cannot be empty.")
        sys.exit(1)

    results = run_pipeline(args.topic.strip())

    if not any(r.accepted for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()