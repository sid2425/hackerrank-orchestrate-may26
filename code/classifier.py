import json
import re

import requests

from config import (
    COMPANIES,
    HIGH_RISK_KEYWORDS,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_SEED,
    OLLAMA_TEMPERATURE,
    OLLAMA_TOP_K,
    OLLAMA_TOP_P,
    VALID_REQUEST_TYPES,
)
from utils import cache_get, cache_set

CLASSIFY_PROMPT = """You are a support ticket classifier. Classify the ticket below and output ONLY a valid JSON object — no explanation, no markdown, no extra text.

Output schema:
{{
  "request_type": one of {request_types},
  "product_area": string (specific support category, e.g. "screen", "privacy", "billing", "account_management"),
  "risk_level": one of "low" | "medium" | "high",
  "inferred_company": one of "HackerRank" | "Claude" | "Visa" | "Unknown"
}}

Rules:
- risk_level "high" = involves billing disputes, account access loss, security, fraud, legal issues, card problems
- request_type "invalid" = nonsensical, off-topic, clearly not a support request, malicious content
- inferred_company "Unknown" = cannot determine company with high confidence from the ticket text
- If company is explicitly provided below, use it as inferred_company (do not override with Unknown)

Ticket:
Issue: {issue}
Subject: {subject}
Company: {company}
"""


def rule_based_escalation(issue: str, subject: str) -> bool:
    """True if ticket matches high-risk keyword list."""
    text = f"{issue} {subject}".lower()
    return any(kw in text for kw in HIGH_RISK_KEYWORDS)


def classify(
    issue: str,
    subject: str,
    company: str,
    cache_store: dict,
) -> dict:
    """
    Returns {request_type, product_area, risk_level, inferred_company, rule_triggered}.
    rule_triggered=True means pre-filter fired; LLM was not called.
    """
    if rule_based_escalation(issue, subject):
        return {
            "request_type": "product_issue",
            "product_area": "security",
            "risk_level": "high",
            "inferred_company": company if company and company != "None" else "Unknown",
            "rule_triggered": True,
        }

    prompt = CLASSIFY_PROMPT.format(
        request_types=json.dumps(VALID_REQUEST_TYPES),
        issue=issue,
        subject=subject,
        company=company or "None",
    )

    cached = cache_get(cache_store, prompt)
    if cached:
        cached["rule_triggered"] = False
        return cached

    result = _call_ollama(prompt)
    result["rule_triggered"] = False
    cache_set(cache_store, prompt, result)
    return result


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
    resp = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=120)
    resp.raise_for_status()
    raw = resp.json().get("response", "{}")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = _extract_json(raw)

    return {
        "request_type": _coerce(data.get("request_type"), VALID_REQUEST_TYPES, "product_issue"),
        "product_area": str(data.get("product_area", "general")),
        "risk_level": _coerce(data.get("risk_level"), ["low", "medium", "high"], "low"),
        "inferred_company": _coerce(data.get("inferred_company"), COMPANIES + ["Unknown"], "Unknown"),
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
