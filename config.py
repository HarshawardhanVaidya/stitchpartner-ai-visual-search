"""Central configuration. Everything comes from environment variables."""

from __future__ import annotations

from dotenv import load_dotenv
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name) or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name) or default)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------
# WooCommerce — the source of truth for products
# --------------------------------------------------------------------------
WOOCOMMERCE_URL = _env("WOOCOMMERCE_URL").rstrip("/")
WOOCOMMERCE_CONSUMER_KEY = _env("WOOCOMMERCE_CONSUMER_KEY")
WOOCOMMERCE_CONSUMER_SECRET = _env("WOOCOMMERCE_CONSUMER_SECRET")
WOOCOMMERCE_TIMEOUT = _env_float("WOOCOMMERCE_TIMEOUT", 20.0)
WOOCOMMERCE_PAGE_SIZE = _env_int("WOOCOMMERCE_PAGE_SIZE", 50)

# Shared secret configured on the WooCommerce webhook.
WEBHOOK_SECRET = _env("WEBHOOK_SECRET")

# Bearer token protecting /sync, /sync/status and other admin routes.
ADMIN_API_KEY = _env("ADMIN_API_KEY")


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------
# On Railway, point INDEX_PATH at a mounted volume so the index survives deploys.
INDEX_PATH = Path(_env("INDEX_PATH") or (BASE_DIR / "data" / "index"))
TEMP_DIR = Path(_env("TEMP_DIR") or (BASE_DIR / "data" / "tmp"))

# Keep the HuggingFace cache on the same volume so the model is downloaded once.
HF_CACHE_DIR = _env("HF_HOME") or str(INDEX_PATH.parent / "hf-cache")
os.environ.setdefault("HF_HOME", HF_CACHE_DIR)


# --------------------------------------------------------------------------
# Model / embeddings
# --------------------------------------------------------------------------
MODEL_NAME = _env("MODEL_NAME", "facebook/dinov2-base")

# Bumping this invalidates every stored embedding and forces a rebuild.
EMBEDDING_VERSION = _env("EMBEDDING_VERSION", "v2")

# Longest edge the image is reduced to before the model's own preprocessing.
# DINOv2's processor crops to 224, so anything above ~768 is wasted work.
PREPROCESS_MAX_DIM = _env_int("PREPROCESS_MAX_DIM", 768)

# Embedding pooling: "cls" (current, accuracy-tested) or "cls_patch" (wider
# vector, must be A/B measured with scripts/evaluate_accuracy.py before use).
POOLING = _env("POOLING", "cls").lower()

# torch intra-op threads. Leave at 0 to use every core Railway gives us.
TORCH_THREADS = _env_int("TORCH_THREADS", 0)

# Run one throwaway inference at startup so the first customer doesn't pay for
# lazy kernel initialisation.
WARMUP_ON_STARTUP = _env_bool("WARMUP_ON_STARTUP", True)


# --------------------------------------------------------------------------
# Upload limits
# --------------------------------------------------------------------------
MAX_UPLOAD_SIZE = _env_int("MAX_UPLOAD_SIZE", 10 * 1024 * 1024)
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_PIXELS = _env_int("MAX_PIXELS", 50_000_000)  # decompression-bomb guard


# --------------------------------------------------------------------------
# Search behaviour
# --------------------------------------------------------------------------
TOP_K_IMAGES = _env_int("TOP_K_IMAGES", 40)
TOP_K_PRODUCTS = _env_int("TOP_K_PRODUCTS", 12)

# Below this cosine similarity we tell the customer we found nothing,
# rather than showing unrelated products.
MIN_MATCH_SCORE = _env_float("MIN_MATCH_SCORE", 0.45)
STRONG_MATCH_SCORE = _env_float("STRONG_MATCH_SCORE", 0.65)


# --------------------------------------------------------------------------
# Sync
# --------------------------------------------------------------------------
SYNC_INTERVAL_MINUTES = _env_int("SYNC_INTERVAL_MINUTES", 60)
SYNC_ON_STARTUP = _env_bool("SYNC_ON_STARTUP", False)
MAX_IMAGES_PER_PRODUCT = _env_int("MAX_IMAGES_PER_PRODUCT", 6)
IMAGE_DOWNLOAD_TIMEOUT = _env_float("IMAGE_DOWNLOAD_TIMEOUT", 20.0)


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
CORS_ORIGINS = [o.strip() for o in _env("CORS_ORIGINS", "*").split(",") if o.strip()]
SEARCH_RATE_LIMIT = _env_int("SEARCH_RATE_LIMIT", 20)       # requests
SEARCH_RATE_WINDOW = _env_int("SEARCH_RATE_WINDOW", 60)     # seconds

LOG_LEVEL = _env("LOG_LEVEL", "INFO").upper()


def woocommerce_configured() -> bool:
    return bool(
        WOOCOMMERCE_URL
        and WOOCOMMERCE_CONSUMER_KEY
        and WOOCOMMERCE_CONSUMER_SECRET
    )


def ensure_directories() -> None:
    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    Path(HF_CACHE_DIR).mkdir(parents=True, exist_ok=True)
