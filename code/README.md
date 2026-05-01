# Support Triage Agent

Terminal-based support triage agent for HackerRank Orchestrate (May 2026).
Processes `support_tickets/support_tickets.csv` and writes `support_tickets/output.csv`.

## Architecture

```
main.py          orchestrates the pipeline per ticket
config.py        constants, env vars, path resolution
indexer.py       chunk + embed 722 corpus docs → numpy matrix
retriever.py     cosine-sim search against the matrix
classifier.py    rule-based pre-filter + Ollama LLM classify
generator.py     Ollama LLM response generation
grounding_checker.py  NLI entailment check (no extra LLM call)
escalator.py     aggregate pipeline signals → final status
output_writer.py write rows to output.csv
utils.py         chunking, text cleaning, LLM response cache
```

### Per-ticket pipeline

```
ticket
  │
  ├─ rule-based pre-filter (keyword match) ──► escalated
  │
  ├─ company inference (company=None) ────────► escalated if Unknown
  │
  ├─ LLM classify → request_type, product_area, risk_level
  │
  ├─ retrieve top-5 corpus chunks (numpy cosine sim)
  │     └─ no match → escalated
  │
  ├─ LLM generate → response, justification, grounded
  │
  ├─ NLI grounding check → override grounded if entailment < 0.5
  │
  └─ escalator → final status
```

## Running with Docker (recommended)

Requires Docker and Docker Compose.

```bash
# Step 1: pull the Ollama model (once)
bash setup.sh

# Step 2: run the agent
docker compose up agent
```

Output is written to `support_tickets/output.csv`.

## Running locally

```bash
# Install Ollama: https://ollama.com/download
ollama pull phi3

cd code
pip install -r requirements.txt

# Optional: copy and edit .env
cp ../.env.example ../.env

python main.py
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `phi3` | Model to use (`phi3` or `llama3.1:8b`) |
| `DATA_DIR` | `../data` | Path to corpus directory |
| `SUPPORT_DIR` | `../support_tickets` | Path to CSV directory |

## Determinism

- Ollama called with `temperature=0, top_k=1, top_p=0, seed=42`
- LLM responses cached in `code/.llm_cache.json` keyed by `sha256(prompt)`
- Second run is fully deterministic: cache serves all LLM calls
- Docker pins Ollama version, Python version, and all package versions

## Models used

| Model | Purpose | License |
|---|---|---|
| `phi3` (Ollama) | Classification + generation | MIT |
| `all-MiniLM-L6-v2` | Embeddings (384-dim) | Apache 2.0 |
| `cross-encoder/nli-MiniLM2-L6-H768` | NLI grounding check | Apache 2.0 |

No API keys required.
