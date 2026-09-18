"""
API tests. The embedder is stubbed, so these run without model weights.

    python -m pytest tests/test_api.py -v
"""

from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as app_module
import config
from vector_store import ImageRecord


def make_image(size=(400, 300), colour=(180, 60, 60), fmt="JPEG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format=fmt)
    return buffer.getvalue()


class StubEmbedder:
    model_name = "stub"
    is_loaded = True
    dim = 8

    def load(self):
        return None

    def warmup(self):
        return 0.0

    def decode(self, data):
        from embedder import ImageDecodeError
        try:
            return Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as exc:
            raise ImageDecodeError("not an image") from exc

    def embed_image(self, image):
        return self._vector(image.tobytes()[:2048])

    def embed_bytes(self, data):
        from embedder import EmbeddingResult
        image = self.decode(data)
        return EmbeddingResult(self.embed_image(image), 5.0, 30.0)

    @staticmethod
    def _vector(payload: bytes) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
        vector = np.random.default_rng(seed).standard_normal(8).astype(np.float32)
        return vector / np.linalg.norm(vector)

    def health(self):
        return {"loaded": True, "model": "stub", "dim": 8, "load_seconds": 0.0}


@pytest.fixture
def client(tmp_path, monkeypatch):
    stub = StubEmbedder()
    monkeypatch.setattr(app_module, "embedder", stub)
    monkeypatch.setattr(app_module.search_service, "embedder", stub)
    monkeypatch.setattr(config, "ADMIN_API_KEY", "test-admin-key")
    monkeypatch.setattr(app_module.config, "ADMIN_API_KEY", "test-admin-key")

    # Run the lifespan first (it calls store.load(), which would otherwise
    # overwrite whatever we seed), then seed the index.
    with TestClient(app_module.app):
        pass

    store = app_module.store
    store.replace_all(
        np.array([[1, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32),
        [ImageRecord("141", "141-0", "https://img/141.jpg", "h", config.EMBEDDING_VERSION, "stub")],
        {
            "141": {
                "id": "141",
                "name": "Thread Guide Set for Jack W4",
                "url": "https://shop.test/product/141",
                "sku": "30313019",
                "primary_image": "https://img/141.jpg",
                "price_html": "<span>₹250</span>",
                "categories": ["Thread guides"],
                "brand": "Jack",
                "stock_status": "instock",
            }
        },
    )
    return TestClient(app_module.app, raise_server_exceptions=False)


# ----------------------------------------------------------------------
def test_health_is_cheap_and_informative(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["model"]["loaded"] is True
    assert body["index"]["products"] >= 1


def test_search_returns_products(client):
    response = client.post("/search", files={"file": ("part.jpg", make_image(), "image/jpeg")})
    assert response.status_code == 200
    assert response.json()["success"] is True


def test_search_rejects_a_non_image(client):
    response = client.post(
        "/search", files={"file": ("notes.txt", b"this is not an image", "image/jpeg")}
    )
    assert response.status_code == 400
    assert "JPG" in response.json()["message"]


def test_oversized_upload_is_rejected(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_SIZE", 1024)
    monkeypatch.setattr(app_module.config, "MAX_UPLOAD_SIZE", 1024)
    response = client.post(
        "/search",
        files={"file": ("big.jpg", make_image(size=(2000, 2000), colour=(9, 200, 77)), "image/jpeg")},
    )
    assert response.status_code == 413
    assert "smaller than" in response.json()["message"]


def test_errors_never_leak_internals(client, monkeypatch):
    def explode(_data):
        raise RuntimeError("psycopg2.OperationalError at /srv/secret/path")

    monkeypatch.setattr(app_module.search_service, "search", explode)
    response = client.post("/search", files={"file": ("p.jpg", make_image(), "image/jpeg")})
    assert response.status_code == 500
    body = response.text
    assert "psycopg2" not in body and "/srv/secret" not in body
    assert "trouble analysing" in response.json()["message"]


def test_admin_routes_require_a_token(client):
    assert client.get("/sync/status").status_code == 401
    assert client.post("/sync").status_code == 401

    ok = client.get("/sync/status", headers={"Authorization": "Bearer test-admin-key"})
    assert ok.status_code == 200
    assert ok.json()["indexed_products"] >= 1


def test_webhook_rejects_a_bad_signature(client):
    response = client.post(
        "/webhook/woocommerce",
        json={"id": 103},
        headers={"x-wc-webhook-signature": "not-the-real-signature",
                 "x-wc-webhook-topic": "product.updated"},
    )
    assert response.status_code == 401


def test_webhook_accepts_a_valid_signature_and_removes_on_delete(client, monkeypatch):
    import hmac, json
    from base64 import b64encode

    secret = "webhook-secret"
    monkeypatch.setattr(config, "WEBHOOK_SECRET", secret)
    monkeypatch.setattr(app_module.config, "WEBHOOK_SECRET", secret)

    body = json.dumps({"id": 141}).encode()
    signature = b64encode(
        hmac.new(secret.encode(), body, hashlib.sha256).digest()
    ).decode()

    response = client.post(
        "/webhook/woocommerce",
        content=body,
        headers={
            "content-type": "application/json",
            "x-wc-webhook-signature": signature,
            "x-wc-webhook-topic": "product.deleted",
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "removed"
    assert app_module.store.get_product("141") is None


def test_rate_limit_kicks_in(client, monkeypatch):
    monkeypatch.setattr(config, "SEARCH_RATE_LIMIT", 3)
    monkeypatch.setattr(app_module.config, "SEARCH_RATE_LIMIT", 3)
    app_module._hits.clear()

    image = make_image()
    codes = [
        client.post("/search", files={"file": ("p.jpg", image, "image/jpeg")}).status_code
        for _ in range(5)
    ]
    assert 429 in codes
