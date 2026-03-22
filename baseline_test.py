"""
baseline_test.py
----------------
B2 Baseline Test — Direct LLM prompting without the pipeline.

This simulates what a researcher would get if they simply asked an LLM
to generate research ideas directly, with no literature retrieval,
no concept extraction, no gap identification, and no feasibility checks.

Compare this output against:
    python main.py --topic "federated learning privacy"

Evaluation criteria (record manually):
    1. Are the datasets real and specifically named?
    2. Are the evaluation metrics appropriate?
    3. Are the ideas grounded in actual literature?
    4. Are there any hallucinated or vague experimental setups?

Usage:
    python baseline_test.py
"""

import json
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL

client = Groq(api_key=GROQ_API_KEY)

BASELINE_PROMPT = """Generate 3 novel research ideas about federated learning privacy.

For each idea provide:
- research_question: the core research question
- proposed_method: the method or approach
- dataset: the specific dataset to be used
- evaluation_metric: how success will be measured

Return ONLY a valid JSON object with a single key 'ideas' containing an array of 3 objects.
Each object must have exactly these keys: research_question, proposed_method, dataset, evaluation_metric."""

print("=" * 60)
print("B2 Baseline Test — Direct LLM Prompting")
print("Topic: federated learning privacy")
print("=" * 60)
print("\nSending prompt directly to LLM (no pipeline)...\n")

response = client.chat.completions.create(
    model=LLM_MODEL,
    max_tokens=1024,
    response_format={"type": "json_object"},
    messages=[
        {
            "role": "system",
            "content": "You are a research assistant. Return only valid JSON."
        },
        {
            "role": "user",
            "content": BASELINE_PROMPT
        }
    ]
)

raw = response.choices[0].message.content
parsed = json.loads(raw)
ideas = parsed.get("ideas", [])

print(f"Generated {len(ideas)} baseline idea(s)\n")
print("=" * 60)

for i, idea in enumerate(ideas, 1):
    print(f"\nBaseline Idea {i}:")
    print(f"  Research Question : {idea.get('research_question', 'N/A')}")
    print(f"  Proposed Method   : {idea.get('proposed_method', 'N/A')}")
    print(f"  Dataset           : {idea.get('dataset', 'N/A')}")
    print(f"  Evaluation Metric : {idea.get('evaluation_metric', 'N/A')}")

print("\n" + "=" * 60)
print("Evaluation Checklist (fill in manually):")
print("=" * 60)
print("""
Compare each idea against the pipeline output from:
    python main.py --topic "federated learning privacy"

For each baseline idea, answer:
  [ ] 1. Is the dataset a real, named, publicly available benchmark?
  [ ] 2. Is the evaluation metric standard and appropriate?
  [ ] 3. Is the method grounded in existing literature?
  [ ] 4. Does it avoid vague or hallucinated experimental setups?

Score each idea: 0-4 points (1 point per YES answer above)
Record scores for your evaluation chapter comparison table.
""")

# Save baseline output for comparison
import os
os.makedirs("outputs", exist_ok=True)
with open("outputs/baseline_b2.json", "w") as f:
    json.dump({
        "test": "B2 baseline",
        "topic": "federated learning privacy",
        "model": LLM_MODEL,
        "prompt_type": "direct — no pipeline",
        "ideas": ideas
    }, f, indent=2)

print("Baseline output saved to: outputs/baseline_b2.json")
print("=" * 60)
