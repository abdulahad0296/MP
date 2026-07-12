"""
tools/llm.py
------------
Shared Groq LLM client and JSON helpers for the Agentic Research
Planning Framework.

All agents and tools make their LLM calls through this module, so the
Groq client is initialised exactly once and JSON-mode calling / parsing
behaviour is consistent everywhere.

Usage:
    from tools.llm import call_llm, parse_json
    raw    = call_llm(system_prompt, user_prompt)
    parsed = parse_json(raw, "my_label", tag="planner_agent")
"""

import json
import re
import time
from typing import Any, Optional

from groq import Groq, RateLimitError

import config

# ── Groq client — initialised once, shared by all agents and tools ────────────
_client = Groq(api_key=config.GROQ_API_KEY)

# Transient rate-limit retry policy. The Groq free tier has BOTH a
# per-minute token limit (TPM, ~12k — recoverable by waiting seconds)
# and a daily token limit (not recoverable). We retry short waits and
# re-raise immediately for long ones so main.py's partial-results
# recovery still triggers on daily exhaustion.
_MAX_RETRIES        = 4
_RETRY_CAP_SECONDS  = 90.0


def _suggested_wait(exc: RateLimitError, attempt: int) -> float:
    """
    Extract Groq's 'Please try again in ...' hint, else exponential backoff.
    Handles simple ('240ms', '7.66s') and compound ('14m59.5s', '1h2m')
    duration formats.
    """
    m = re.search(r"try again in ([\dhms.]+)", str(exc))
    if m:
        total = 0.0
        for value, unit in re.findall(r"([\d.]+)(ms|h|m|s)", m.group(1)):
            total += float(value) * {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}[unit]
        if total > 0:
            return total + 1.0  # small pad past the window boundary
    return 5.0 * (attempt + 1)


def call_llm(system_prompt: str, user_prompt: str,
             max_tokens: Optional[int] = None) -> str:
    """
    Make a single Groq API call in JSON mode and return the raw response text.

    Transient per-minute rate limits are retried with backoff (up to
    _MAX_RETRIES). Daily-limit errors (long suggested waits) are raised
    immediately.

    Args:
        system_prompt : System message enforcing the output structure.
        user_prompt   : User message with the task content.
        max_tokens    : Per-call token cap. Defaults to config.LLM_MAX_TOKENS.

    Returns:
        Raw response text (may be empty string if the model returned nothing).
    """
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = _client.chat.completions.create(
                model=config.LLM_MODEL,
                max_tokens=max_tokens or config.LLM_MAX_TOKENS,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ]
            )
            return response.choices[0].message.content or ""
        except RateLimitError as e:
            wait = _suggested_wait(e, attempt)
            if attempt >= _MAX_RETRIES or wait > _RETRY_CAP_SECONDS:
                raise  # daily limit or retries exhausted — let callers recover
            print(f"[llm] Rate limit (per-minute) — retrying in {wait:.1f}s "
                  f"(attempt {attempt + 1}/{_MAX_RETRIES})...")
            time.sleep(wait)
    return ""  # unreachable — loop either returns or raises


def parse_json(raw: str, label: str, tag: str = "llm") -> Optional[Any]:
    """
    Parse a JSON string returned by the LLM.
    Logs a warning and returns None on failure — never crashes the pipeline.

    Args:
        raw   : Raw LLM response text.
        label : Short description of the call, used in the warning message.
        tag   : Log prefix identifying the calling module.

    Returns:
        Parsed JSON value, or None if parsing failed.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[{tag}] WARNING: JSON parse failed for '{label}'. Error: {e}")
        print(f"[{tag}] Raw response: {raw[:200]}")
        return None
