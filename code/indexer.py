from __future__ import annotations

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from config import DATA_DIR, EMBED_MODEL
from utils import chunk_document, company_from_path, product_area_from_path


def load_corpus(data_dir: Path = DATA_DIR) -> tuple[np.ndarray, list[dict]]:
    """
    Walk data/, chunk every .md file, embed chunks.
    Returns (embedding_matrix, metadata_list).
    embedding_matrix: (N, 384) float32
    metadata_list: list of {text, source, company, product_area}
    """
    model = SentenceTransformer(EMBED_MODEL)

    all_chunks: list[dict] = []
    for md_file in sorted(data_dir.rglob("*.md")):
        try:
            text = md_file.read_text(encoding="utf-8", errors="ignore").strip()
        except OSError:
            continue
        if not text:
            continue

        company = company_from_path(md_file)
        product_area = product_area_from_path(md_file)

        for chunk in chunk_document(text, str(md_file)):
            chunk["company"] = company
            chunk["product_area"] = product_area
            all_chunks.append(chunk)

    texts = [c["text"] for c in all_chunks]
    print(f"Indexing {len(texts)} chunks from {data_dir} ...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    matrix = np.array(embeddings, dtype=np.float32)

    return matrix, all_chunks
