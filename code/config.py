import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3")
OLLAMA_SEED = 42
OLLAMA_TEMPERATURE = 0.0
OLLAMA_TOP_K = 1
OLLAMA_TOP_P = 0.0

EMBED_MODEL = "all-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"

# Allow Docker mounts to override paths via env vars
_code_dir = Path(__file__).parent
ROOT_DIR = Path(os.getenv("ROOT_DIR", str(_code_dir.parent)))
DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT_DIR / "data")))
SUPPORT_DIR = Path(os.getenv("SUPPORT_DIR", str(ROOT_DIR / "support_tickets")))
INPUT_CSV = SUPPORT_DIR / "support_tickets.csv"
OUTPUT_CSV = SUPPORT_DIR / "output.csv"
CACHE_FILE = _code_dir / ".llm_cache.json"

RETRIEVAL_TOP_K = 5
RETRIEVAL_THRESHOLD = 0.4
NLI_ENTAILMENT_THRESHOLD = 0.1

CHUNK_MAX_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 50

HIGH_RISK_KEYWORDS = [
    "fraud", "identity theft",
    "delete my account", "delete account",
    "security vulnerability",
    "blocked card", "card blocked",
    "unauthorized transaction", "unauthorised transaction",
    "chargeback", "lawsuit", "legal action",
    "police report", "data breach",
    "account hacked", "account has been hacked",
    "right to erasure", "right to be forgotten",
    "phishing",
]

COMPANIES = ["HackerRank", "Claude", "Visa"]
VALID_REQUEST_TYPES = ["product_issue", "feature_request", "bug", "invalid"]
VALID_STATUSES = ["replied", "escalated"]

PROMPT_VERSION = "v1"
