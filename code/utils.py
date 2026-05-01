from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Optional

from config import CACHE_FILE, CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS, PROMPT_VERSION


def approx_token_count(text: str) -> int:
    return len(text.split())


def chunk_document(text: str, source: str) -> list[dict]:
    """Split document into chunks. Heading-split first, paragraph fallback."""
    chunks = []

    # Try heading split on ##
    sections = re.split(r"\n(?=## )", text)
    if len(sections) > 1:
        raw_chunks = sections
    else:
        # Fallback: paragraph split
        raw_chunks = [p.strip() for p in text.split("\n\n") if p.strip()]

    for raw in raw_chunks:
        if not raw.strip():
            continue
        if approx_token_count(raw) <= CHUNK_MAX_TOKENS:
            chunks.append({"text": raw.strip(), "source": source})
        else:
            # Split long chunk by paragraph with overlap
            paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
            window: list[str] = []
            window_tokens = 0
            for para in paragraphs:
                para_tokens = approx_token_count(para)
                if window_tokens + para_tokens > CHUNK_MAX_TOKENS and window:
                    chunks.append({"text": "\n\n".join(window), "source": source})
                    # Keep last overlap worth of paragraphs
                    overlap: list[str] = []
                    overlap_tokens = 0
                    for p in reversed(window):
                        if overlap_tokens + approx_token_count(p) <= CHUNK_OVERLAP_TOKENS:
                            overlap.insert(0, p)
                            overlap_tokens += approx_token_count(p)
                        else:
                            break
                    window = overlap
                    window_tokens = overlap_tokens
                window.append(para)
                window_tokens += para_tokens
            if window:
                chunks.append({"text": "\n\n".join(window), "source": source})

    return chunks


def company_from_path(path: Path) -> str:
    """Infer company from data/ subdirectory name."""
    parts = path.parts
    for i, part in enumerate(parts):
        if part == "data" and i + 1 < len(parts):
            domain = parts[i + 1].lower()
            if domain == "hackerrank":
                return "HackerRank"
            elif domain == "claude":
                return "Claude"
            elif domain == "visa":
                return "Visa"
    return "Unknown"


def product_area_from_path(path: Path) -> str:
    """Derive product_area from path segments under data/<company>/."""
    parts = path.parts
    for i, part in enumerate(parts):
        if part == "data" and i + 2 < len(parts):
            return parts[i + 2]
    return path.stem


# LLM response cache keyed by hash(ticket + prompt_version)

def _cache_key(prompt: str) -> str:
    payload = f"{PROMPT_VERSION}:{prompt}"
    return hashlib.sha256(payload.encode()).hexdigest()


def cache_load() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def cache_save(store: dict) -> None:
    CACHE_FILE.write_text(json.dumps(store, indent=2))


def cache_get(store: dict, prompt: str) -> Optional[dict]:
    return store.get(_cache_key(prompt))


def cache_set(store: dict, prompt: str, response: dict) -> None:
    store[_cache_key(prompt)] = response
