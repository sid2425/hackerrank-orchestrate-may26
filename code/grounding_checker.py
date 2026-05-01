from __future__ import annotations

from sentence_transformers import CrossEncoder

from config import NLI_ENTAILMENT_THRESHOLD, NLI_MODEL

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(NLI_MODEL)
    return _model


def is_grounded(response: str, chunks: list[dict], threshold: float = NLI_ENTAILMENT_THRESHOLD) -> bool:
    """
    Returns True if the response is entailed by at least one retrieved chunk.
    Uses NLI cross-encoder: premise=chunk text, hypothesis=response.
    Label order for nli-MiniLM2-L6-H768: [contradiction, entailment, neutral]
    """
    if not chunks:
        return False

    model = _get_model()
    pairs = [(c["text"], response) for c in chunks]
    scores = model.predict(pairs)

    # scores shape: (N, 3) — columns: contradiction, entailment, neutral
    import numpy as np
    scores_arr = np.array(scores)
    if scores_arr.ndim == 1:
        # Single pair edge case
        scores_arr = scores_arr.reshape(1, -1)

    entailment_scores = scores_arr[:, 1]
    return float(entailment_scores.max()) >= threshold
