FROM python:3.11-slim

WORKDIR /app

# Install deps first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models during build so first run is instant
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('all-MiniLM-L6-v2'); \
    CrossEncoder('cross-encoder/nli-MiniLM2-L6-H768')"

COPY . .

# Mount the repo root at /repo so the agent can reach data/ and support_tickets/
CMD ["python", "main.py"]
