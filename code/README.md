# Support Triage Agent

Terminal-based support triage agent for HackerRank Orchestrate (May 2026).
Reads `support_tickets/support_tickets.csv`, processes each ticket through a RAG pipeline, writes results to `support_tickets/output.csv`.

---

## Approach

### Problem
Triage support tickets across three product domains (HackerRank, Claude, Visa) — classify each ticket, retrieve relevant documentation, generate a grounded response, and decide whether to reply or escalate to a human.

### Solution: RAG pipeline with explicit escalation logic

Each ticket passes through a linear, stateless pipeline:

```
ticket
  │
  ├─ 1. Rule-based pre-filter
  │       High-risk keywords (fraud, identity theft, security vulnerability, etc.)
  │       → immediate escalation, no LLM call
  │
  ├─ 2. Company inference (when company=None)
  │       LLM infers company from ticket text
  │       → escalate if "Unknown" (wrong-domain grounding is dangerous)
  │
  ├─ 3. LLM classify (Ollama)
  │       → request_type, product_area, risk_level, inferred_company
  │
  ├─ 4. Corpus retrieval
  │       Embed query with all-MiniLM-L6-v2
  │       Cosine-sim search against 8324 chunks (722 corpus docs, in-memory numpy)
  │       → escalate if no chunk scores above threshold (cosine sim < 0.4)
  │
  ├─ 5. LLM generate (Ollama)
  │       Prompt includes retrieved articles only — no outside knowledge
  │       → response, justification, status, grounded flag
  │
  ├─ 6. NLI grounding check
  │       cross-encoder/nli-MiniLM2-L6-H768 checks entailment
  │       Premise = retrieved chunks, Hypothesis = generated response
  │       No extra LLM call — runs on CPU in ~50ms
  │
  └─ 7. Escalator (pure function)
          Aggregates all signals → final status
          Priority: rule_triggered > company_unknown > no_corpus_match > LLM_escalated
```

### Key design decisions

| Decision | Choice | Why |
|---|---|---|
| LLM backend | Ollama (local) | No API key dependency; fully reproducible in Docker |
| Vector store | In-memory numpy | 722 docs × 384-dim = ~1 MB; no database needed |
| Two LLM calls | classify then generate | product_area from classify narrows retrieval scope |
| Grounding check | NLI model | Verifiable without extra LLM call; CPU-friendly |
| Escalation | Centralised escalator.py | Single source of truth; pure function — easy to test |
| Determinism | seed=42 + response cache | Byte-for-byte identical output.csv across runs |

---

## Setup & Run

### Option A — Docker (recommended)

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop).

```bash
# Step 1: pull the LLM model once (~2 GB)
bash setup.sh

# Step 2: run the agent
docker compose up agent
```

Output written to `support_tickets/output.csv`.

### Option B — Local

```bash
# 1. Install Ollama: https://ollama.com/download
ollama pull llama3.2

# 2. Install Python deps (Python 3.9+)
cd code
pip install -r requirements.txt

# 3. Configure (optional)
cp ../.env.example ../.env
# edit .env if Ollama runs on a different host/port

# 4. Run
python main.py
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `phi3` | Model name (`llama3.2`, `phi3`, `llama3.1:8b`) |
| `DATA_DIR` | `../data` | Path to corpus directory |
| `SUPPORT_DIR` | `../support_tickets` | Path to CSV input/output directory |

---

## Models

| Model | Purpose | Size | License |
|---|---|---|---|
| `llama3.2` via Ollama | Classification + generation | ~2 GB | Llama 3 Community |
| `all-MiniLM-L6-v2` | Corpus embeddings (384-dim) | 80 MB | Apache 2.0 |
| `cross-encoder/nli-MiniLM2-L6-H768` | NLI grounding check | 65 MB | Apache 2.0 |

No external API keys required.

---

## Determinism

- Ollama: `temperature=0`, `top_k=1`, `top_p=0`, `seed=42`
- LLM responses cached in `code/.llm_cache.json` keyed by `sha256(prompt + prompt_version)`
- Re-runs serve from cache — output.csv is byte-for-byte identical
- Docker pins Python 3.11, all package versions, and Ollama version

---

## Module reference

```
main.py              CLI entry point — orchestrates pipeline per ticket
config.py            Constants, env vars, path resolution
indexer.py           Walk data/, chunk docs, embed → numpy matrix
retriever.py         Cosine-sim search with company-domain filter
classifier.py        Rule-based pre-filter + LLM classify
generator.py         LLM response generation with grounded flag
grounding_checker.py NLI entailment check
escalator.py         Aggregate signals → final (status, reason)
output_writer.py     Append rows to output.csv
utils.py             Chunking (heading → paragraph fallback), LLM cache
```
