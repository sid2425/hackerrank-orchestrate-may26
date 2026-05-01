#!/usr/bin/env bash
set -euo pipefail

MODEL="${OLLAMA_MODEL:-phi3}"

echo "==> Starting Ollama to pull model: $MODEL"
docker compose up -d ollama

echo "==> Waiting for Ollama to be ready ..."
until docker compose exec ollama curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
  sleep 2
done

echo "==> Pulling model: $MODEL (this may take a few minutes on first run)"
docker compose exec ollama ollama pull "$MODEL"

echo "==> Model ready. Run: docker compose up agent"
