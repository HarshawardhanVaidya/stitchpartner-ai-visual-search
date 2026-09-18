"""
StitchPartnerAI — visual product search API.

Startup does four cheap things and nothing else:

    load model -> load index from disk -> warm up -> serve

It never downloads the catalogue or generates embeddings at boot. Catalogue
work happens on the webhook, on the periodic sync, or when an administrator
asks for it.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from base64 import b64encode
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import (
    BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request, Response, UploadFile, File,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

import config
from PIL import UnidentifiedImageError

from embedder import ImageDecodeError, embedder
from search import SearchService
from sync import SyncService
from vector_store import VectorStore
from woocommerce import WooCommerceClient

# --------------------------------------------------------------------------
# Logging — never log image bytes, credentials or full request bodies.
# --------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
logger = logging.getLogger("stitchpartner")

config.ensure_directories()

store = VectorStore(config.INDEX_PATH)
woo_client = WooCommerceClient()
search_service = SearchService(store, embedder)
sync_service = SyncService(store, embedder, woo_client)

STATIC_DIR = Path(__file__).resolve().parent / "static"


# --------------------------------------------------------------------------
# Lifespan
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    started = time.perf_counter()

    store.load()

    try:
        embedder.load()
        if config.WARMUP_ON_STARTUP:
            embedder.warmup()
    except Exception:
        # A model failure must not stop the process: /health will report it and
        # Railway can restart us, rather than crash-looping silently.
        logger.exception("Model failed to load at startup.")

    dropped = store.prune_stale_embeddings(config.EMBEDDING_VERSION, embedder.model_name)
    if dropped:
        logger.warning(
            "Dropped %d embeddings from an older model/version. "
            "Run a sync to regenerate them.", dropped,
        )
        store.save()

    if not config.woocommerce_configured():
        logger.warning(
            "WooCommerce is not configured; search will serve the existing "
            "index but cannot sync new products."
        )
    else:
        sync_service.start_scheduler()

    logger.info("Startup complete in %.1fs.", time.perf_counter() - started)
    yield
    sync_service.stop_scheduler()


app = FastAPI(title="StitchPartnerAI Visual Search", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Auth helpers
# --------------------------------------------------------------------------
def require_admin(authorization: str = Header(default="")) -> None:
    """Bearer-token gate for anything that can mutate or expose the index."""
    if not config.ADMIN_API_KEY:
        raise HTTPException(503, "Admin API is disabled: ADMIN_API_KEY is not set.")
    expected = f"Bearer {config.ADMIN_API_KEY}"
    if not hmac.compare_digest(authorization.strip(), expected):
        raise HTTPException(401, "Unauthorised.")


def verify_webhook(body: bytes, signature: str) -> bool:
    """WooCommerce signs webhooks with base64(HMAC-SHA256(body, secret))."""
    if not config.WEBHOOK_SECRET:
        return False
    digest = hmac.new(
        config.WEBHOOK_SECRET.encode("utf-8"), body, hashlib.sha256
    ).digest()
    return hmac.compare_digest(b64encode(digest).decode(), (signature or "").strip())


# --------------------------------------------------------------------------
# Rate limiting — in-process, which is right for a single Railway instance.
# --------------------------------------------------------------------------
_hits: dict[str, deque] = defaultdict(deque)


def rate_limit(request: Request) -> None:
    if config.SEARCH_RATE_LIMIT <= 0:
        return
    client = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    client = client or (request.client.host if request.client else "unknown")

    now = time.monotonic()
    window = _hits[client]
    while window and now - window[0] > config.SEARCH_RATE_WINDOW:
        window.popleft()
    if len(window) >= config.SEARCH_RATE_LIMIT:
        raise HTTPException(429, "Too many searches. Please wait a moment and try again.")
    window.append(now)


# --------------------------------------------------------------------------
# Frontend
# --------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


# --------------------------------------------------------------------------
# Health — cheap by design. No inference, no WooCommerce call.
# --------------------------------------------------------------------------
@app.get("/health")
def health() -> JSONResponse:
    manifest = store.manifest
    model_ok = embedder.is_loaded
    index_ok = store.image_count > 0

    payload = {
        "status": "ok" if model_ok and index_ok else "degraded",
        "model": embedder.health(),
        "index": {
            "available": index_ok,
            "images": store.image_count,
            "products": store.product_count,
            "last_sync_at": manifest.get("last_sync_at"),
            "embedding_version": config.EMBEDDING_VERSION,
        },
        "woocommerce_configured": config.woocommerce_configured(),
        "sync_running": sync_service.is_running,
    }
    # 200 even when degraded: a loaded model with an empty index should not
    # make Railway kill a container that can still be synced back to health.
    return JSONResponse(payload, status_code=200 if model_ok else 503)


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------
@app.post("/search")
async def search(
    request: Request,
    file: UploadFile = File(...),
    _: None = Depends(rate_limit),
) -> JSONResponse:
    """
    Accept one customer photo, return matching products.

    The uploaded image is held in memory for the duration of the request and
    never written to disk, so there is nothing to clean up, nothing to leak
    through a public path, and no orphaned file when the request fails.
    """
    if not embedder.is_loaded:
        return JSONResponse(
            {"success": False, "message": "Search is starting up. Please try again in a moment."},
            status_code=503,
        )

    data = b""
    try:
        # Read in chunks so an oversized upload is rejected before it is
        # fully buffered.
        while chunk := await file.read(1024 * 256):
            data += chunk
            if len(data) > config.MAX_UPLOAD_SIZE:
                limit_mb = config.MAX_UPLOAD_SIZE // (1024 * 1024)
                return JSONResponse(
                    {"success": False, "message": f"Please choose an image smaller than {limit_mb} MB."},
                    status_code=413,
                )

        if not data:
            return JSONResponse(
                {"success": False, "message": "No image was received. Please try again."},
                status_code=400,
            )

        logger.info("Search request received (%.0f KB).", len(data) / 1024)

        try:
            result = search_service.search(data)
        except (ImageDecodeError, UnidentifiedImageError):
            return JSONResponse(
                {"success": False, "message": "Please upload a JPG, PNG or WebP image."},
                status_code=400,
            )

        logger.info("Returned %d products.", len(result.get("results", [])))
        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception:
        # Full detail to the logs, a plain sentence to the customer.
        logger.exception("Search failed.")
        return JSONResponse(
            {"success": False, "message": "We're having trouble analysing the image. Please try again."},
            status_code=500,
        )
    finally:
        # Release the buffer and close the SpooledTemporaryFile that Starlette
        # may have spilled to disk for a large upload.
        data = b""
        try:
            await file.close()
        except Exception:
            logger.debug("Upload handle was already closed.")


@app.get("/similar/{product_id}")
def similar(product_id: str, limit: int = 8) -> JSONResponse:
    """'Find similar products' for a WooCommerce product page."""
    results = search_service.similar_to_product(product_id, min(limit, 24))
    return JSONResponse({"success": True, "product_id": product_id, "results": results})


# --------------------------------------------------------------------------
# Sync — administrative
# --------------------------------------------------------------------------
@app.post("/sync", dependencies=[Depends(require_admin)])
def trigger_sync(background: BackgroundTasks, product_id: str | None = None, wait: bool = False):
    """
    Reconcile the index with WooCommerce.

    `wait=true` returns the full report and is the useful mode for debugging;
    the default returns immediately so a long catalogue sync can't time out.
    """
    if not config.woocommerce_configured():
        raise HTTPException(503, "WooCommerce credentials are not configured.")

    if wait:
        if product_id:
            from sync import SyncReport
            report = SyncReport()
            outcome = sync_service.sync_product(product_id, report)
            store.save()
            return {"success": True, "product_id": product_id, "outcome": outcome,
                    "report": report.as_dict()}
        return {"success": True, "report": sync_service.sync_all().as_dict()}

    sync_service.run_in_background(product_id)
    return {"success": True, "status": "started",
            "message": "Sync running in the background. Poll /sync/status for the result."}


@app.get("/sync/status", dependencies=[Depends(require_admin)])
def sync_status() -> dict[str, Any]:
    manifest = store.manifest
    return {
        "running": sync_service.is_running,
        "last_sync_at": manifest.get("last_sync_at"),
        "indexed_products": store.product_count,
        "indexed_images": store.image_count,
        "last_report": sync_service.last_report,
    }


# --------------------------------------------------------------------------
# WooCommerce webhook
# --------------------------------------------------------------------------
@app.post("/webhook/woocommerce")
async def woocommerce_webhook(
    request: Request,
    x_wc_webhook_signature: str = Header(default=""),
    x_wc_webhook_topic: str = Header(default=""),
) -> Response:
    """
    Fires on product.created / product.updated / product.deleted.

    Returns 200 as soon as the signature checks out and does the indexing in
    the background — WooCommerce disables webhooks that respond slowly.
    """
    body = await request.body()

    if not verify_webhook(body, x_wc_webhook_signature):
        logger.warning("Rejected a webhook with an invalid signature.")
        raise HTTPException(401, "Invalid webhook signature.")

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Webhook body was not valid JSON.")

    product_id = str(payload.get("id") or "")
    if not product_id:
        # WooCommerce sends a ping with no id when the webhook is first saved.
        return JSONResponse({"success": True, "status": "ignored"})

    logger.info("Webhook received: topic=%s product=%s", x_wc_webhook_topic, product_id)

    if x_wc_webhook_topic.endswith("deleted"):
        removed = store.remove_product(product_id)
        store.save()
        return JSONResponse({"success": True, "status": "removed" if removed else "unknown"})

    sync_service.run_in_background(product_id)
    return JSONResponse({"success": True, "status": "queued"})
