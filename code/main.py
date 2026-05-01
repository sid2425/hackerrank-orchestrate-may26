#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer

import classifier
import escalator
import generator
import grounding_checker
import output_writer
from config import EMBED_MODEL, INPUT_CSV, OUTPUT_CSV
from indexer import load_corpus
from retriever import retrieve
from utils import cache_load, cache_save


def process_ticket(
    issue: str,
    subject: str,
    company: str,
    matrix,
    metadata: list[dict],
    embed_model: SentenceTransformer,
    cache_store: dict,
) -> dict:
    # Normalize company
    company = company.strip() if company else ""
    if company.lower() in ("none", ""):
        company = ""

    # 1. Classify (includes rule-based pre-filter)
    cls = classifier.classify(issue, subject, company, cache_store)
    rule_triggered = cls["rule_triggered"]
    request_type = cls["request_type"]
    product_area = cls["product_area"]
    risk_level = cls["risk_level"]
    inferred_company = cls["inferred_company"] if not company else company
    company_unknown = (not company) and (inferred_company == "Unknown")

    # Short-circuit: skip retrieval and generation for rule-based escalations
    if rule_triggered or company_unknown:
        status, reason = escalator.decide(
            rule_triggered=rule_triggered,
            company_unknown=company_unknown,
            no_corpus_match=False,
            risk_level=risk_level,
            generator_status="escalated",
            grounded=False,
        )
        return {
            "status": status,
            "product_area": product_area,
            "response": escalator.escalation_response(),
            "justification": reason,
            "request_type": request_type,
        }

    # 2. Retrieve
    query = f"{issue} {subject}".strip()
    chunks, no_corpus_match = retrieve(query, matrix, metadata, embed_model, company_filter=inferred_company)

    # 3. Generate
    if no_corpus_match:
        gen_result = {
            "response": escalator.escalation_response(),
            "justification": "No relevant corpus article found.",
            "status": "escalated",
            "grounded": False,
        }
    else:
        gen_result = generator.generate(issue, subject, inferred_company, chunks, cache_store)

    # 4. NLI grounding check
    if gen_result["grounded"] and chunks:
        nli_grounded = grounding_checker.is_grounded(gen_result["response"], chunks)
    else:
        nli_grounded = gen_result["grounded"]

    # 5. Escalator — final decision
    status, reason = escalator.decide(
        rule_triggered=False,
        company_unknown=False,
        no_corpus_match=no_corpus_match,
        risk_level=risk_level,
        generator_status=gen_result["status"],
        grounded=nli_grounded,
    )

    response = gen_result["response"]
    justification = gen_result["justification"]

    if status == "escalated" and gen_result["status"] == "replied":
        # Escalator overrode generator — replace response with escalation message
        response = escalator.escalation_response()
        if reason:
            justification = reason

    return {
        "status": status,
        "product_area": product_area,
        "response": response,
        "justification": justification,
        "request_type": request_type,
    }


def main() -> None:
    print("Loading corpus and building index ...")
    matrix, metadata = load_corpus()
    embed_model = SentenceTransformer(EMBED_MODEL)
    print(f"Index ready: {len(metadata)} chunks.")

    cache_store = cache_load()

    output_writer.init_output(OUTPUT_CSV)

    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Processing {len(rows)} tickets ...")
    for i, row in enumerate(rows, 1):
        issue = row.get("Issue", "").strip()
        subject = row.get("Subject", "").strip()
        company = row.get("Company", "").strip()

        print(f"  [{i}/{len(rows)}] {issue[:60]!r} ...", end=" ", flush=True)
        t0 = time.time()

        result = process_ticket(issue, subject, company, matrix, metadata, embed_model, cache_store)
        output_writer.write_row(result)

        cache_save(cache_store)
        print(f"{result['status']} ({time.time() - t0:.1f}s)")

    print(f"\nDone. Output written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
