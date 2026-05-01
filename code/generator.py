from __future__ import annotations

import json
import re

import requests

from config import (
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_SEED,
    OLLAMA_TEMPERATURE,
    OLLAMA_TOP_K,
    OLLAMA_TOP_P,
    VALID_REQUEST_TYPES,
    VALID_STATUSES,
)
from utils import cache_get, cache_set

GENERATE_PROMPT = """You are a support triage agent. Answer the ticket using ONLY the support articles below.
Do NOT use any information outside these articles.
If the articles do not contain enough information to answer confidently, set "grounded" to false.

## Support Articles
{articles}

## Ticket
Issue: {issue}
Subject: {subject}
Company: {company}

## Instructions
Output ONLY a valid JSON object — no explanation, no markdown, no extra text.

Output schema:
{{
  "response": string (user-facing reply grounded in the articles, or a polite escalation message if you cannot answer),
  "justification": string (1-2 sentences: what article(s) you used and why, or why escalation is needed),
  "status": "replied" | "escalated",
  "grounded": true | false
}}

Rules:
- If escalating, response must say a human support agent will follow up shortly.
- justification must reference the article source or explain the gap.
- grounded=false triggers an automatic escalation check.
"""


def generate(
    issue: str,
    subject: str,
    company: str,
    chunks: list[dict],
    cache_store: dict,
) -> dict:
    """
    Returns {response, justification, status, grounded}.
    """
    articles_text = _format_chunks(chunks)
    prompt = GENERATE_PROMPT.format(
        articles=articles_text,
        issue=issue,
        subject=subject,
        company=company or "None",
    )

    cached = cache_get(cache_store, prompt)
    if cached:
        return cached

    result = _call_ollama(prompt)
    cache_set(cache_store, prompt, result)
    return result


def _format_chunks(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        source = c.get("source", "unknown")
        parts.append(f"[Article {i}] (source: {source})\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def _call_ollama(prompt: str) -> dict:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": OLLAMA_TEMPERATURE,
            "top_k": OLLAMA_TOP_K,
            "top_p": OLLAMA_TOP_P,
            "seed": OLLAMA_SEED,
        },
    }
    resp = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=180)
    resp.raise_for_status()
    raw = resp.json().get("response", "{}")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = _extract_json(raw)

    return {
        "response": str(data.get("response", "We were unable to process your request. A human agent will follow up shortly.")),
        "justification": str(data.get("justification", "Unable to parse LLM output.")),
        "status": _coerce(data.get("status"), VALID_STATUSES, "escalated"),
        "grounded": bool(data.get("grounded", False)),
    }


def _coerce(value, allowed: list, default: str) -> str:
    if value in allowed:
        return value
    return default


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}
