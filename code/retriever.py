from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBED_MODEL, RETRIEVAL_THRESHOLD, RETRIEVAL_TOP_K


def retrieve(
    query: str,
    matrix: np.ndarray,
    metadata: list[dict],
    model: SentenceTransformer,
    company_filter: str | None = None,
    top_k: int = RETRIEVAL_TOP_K,
    threshold: float = RETRIEVAL_THRESHOLD,
) -> tuple[list[dict], bool]:
    """
    Returns (top_chunks, no_corpus_match).
    no_corpus_match=True when best score is below threshold.
    """
    query_vec = model.encode([query], normalize_embeddings=True)[0].astype(np.float32)

    # Filter by company if provided
    if company_filter and company_filter != "Unknown":
        indices = [i for i, m in enumerate(metadata) if m["company"] == company_filter]
        if not indices:
            indices = list(range(len(metadata)))
    else:
        indices = list(range(len(metadata)))

    sub_matrix = matrix[indices]
    scores = sub_matrix @ query_vec  # cosine sim (vectors normalized)

    top_local = np.argsort(scores)[::-1][:top_k]
    top_global = [indices[i] for i in top_local]
    top_scores = scores[top_local]

    if len(top_scores) == 0 or float(top_scores[0]) < threshold:
        return [], True

    results = []
    for idx, score in zip(top_global, top_scores):
        if float(score) < threshold:
            break
        chunk = dict(metadata[idx])
        chunk["score"] = float(score)
        results.append(chunk)

    return results, False
