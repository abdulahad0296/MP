"""
tools/topic_validator.py
------------------------
Validates that a research topic is within the scope of Computer Science
and Machine Learning before the pipeline runs.

Uses a single fast LLM call to classify the topic. Returns a structured
result so both main.py (CLI) and app.py (Gradio) can handle it uniformly.
"""

import json

from tools.llm import call_llm

_SYSTEM_PROMPT = """You are a research topic classifier for an AI/ML research planning tool.

Your job: decide whether a given topic is within scope for this tool.

IN SCOPE — topics in Computer Science, AI, or Machine Learning, including:
- Machine learning, deep learning, neural networks
- Natural language processing, computer vision, speech
- Federated learning, reinforcement learning, continual learning
- Quantum computing (as applied to ML/CS problems)
- Robotics, autonomous systems, human-computer interaction
- Data science, knowledge graphs, recommendation systems
- Software engineering, distributed systems, cybersecurity

OUT OF SCOPE — topics with no meaningful CS/ML angle, including:
- Pure physics, chemistry, biology, medicine (unless ML is applied)
- History, economics, law, philosophy, social sciences
- Pure mathematics with no CS application
- Creative arts, literature, sports, cooking
- Vague one-word topics that are not CS fields (e.g. "gravity", "love")

Return ONLY a JSON object with exactly these keys:
{
  "in_scope": true or false,
  "confidence": "high" or "medium" or "low",
  "reason": "one sentence explanation",
  "suggested_topic": "a corrected or more specific CS/ML topic if out of scope, else empty string"
}"""


def validate_topic(topic: str) -> dict:
    """
    Validate whether a topic is within CS/ML scope.

    Args:
        topic: The raw topic string from the user.

    Returns:
        Dict with keys:
            in_scope       : bool — True if topic is CS/ML relevant
            confidence     : str  — "high", "medium", or "low"
            reason         : str  — one-sentence explanation
            suggested_topic: str  — alternative suggestion if out of scope
            error          : str  — set only if the LLM call itself failed
    """
    topic = topic.strip()

    if not topic:
        return {
            "in_scope": False,
            "confidence": "high",
            "reason": "Topic is empty.",
            "suggested_topic": "",
        }

    try:
        raw = call_llm(_SYSTEM_PROMPT, f"Topic: {topic}", max_tokens=150) or "{}"
        result = json.loads(raw)

        return {
            "in_scope":        bool(result.get("in_scope", True)),
            "confidence":      str(result.get("confidence", "low")),
            "reason":          str(result.get("reason", "")),
            "suggested_topic": str(result.get("suggested_topic", "")),
        }

    except Exception as e:
        # If validation itself fails, allow the pipeline to proceed rather
        # than blocking the user on a non-critical guard.
        return {
            "in_scope":        True,
            "confidence":      "low",
            "reason":          "Topic validation unavailable — proceeding.",
            "suggested_topic": "",
            "error":           str(e),
        }
