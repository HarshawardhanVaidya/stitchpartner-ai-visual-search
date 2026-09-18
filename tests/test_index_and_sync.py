"""
Tests for the parts that broke before: index persistence, idempotency, and the
add / update / remove lifecycle that new WooCommerce products depend on.

These use a fake WooCommerce and a fake embedder so they run in a second and
need no network, no credentials and no model weights.

    python -m pytest tests/ -v
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from sync import SyncReport, SyncService
from vector_store import ImageRecord, VectorStore


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------
class FakeEmbedder:
    """Deterministic 8-dimensional embeddings derived from the image bytes."""

    model_name = "fake/dinov2"
    dim = 8
    inference_calls = 0

    def decode(self, data: bytes):
        return data

    def embed_image(self, image) -> np.ndarray:
        FakeEmbedder.inference_calls += 1
        seed = int.from_bytes(hashlib.sha256(image).digest()[:4], "big")
        rng = np.random.default_rng(seed)
        vector = rng.standard_normal(8).astype(np.float32)
        return vector / np.linalg.norm(vector)


class FakeWoo:
    """A tiny in-memory WooCommerce whose catalogue the tests can edit."""

    def __init__(self, products: dict[str, dict]):
        self.products = products
        self.image_bytes: dict[str, bytes] = {}

    def get_product(self, product_id):
        return self.products.get(str(product_id))

    def iter_products(self, status="publish"):
        yield from list(self.products.values())

    def download_image(self, url: str) -> bytes:
        return self.image_bytes.get(url, url.encode("utf-8"))


def raw_product(pid: str, name: str, images: list[str], status: str = "publish") -> dict:
    return {
        "id": pid,
        "name": name,
        "permalink": f"https://shop.test/product/{pid}",
        "sku": f"SKU-{pid}",
        "status": status,
        "catalog_visibility": "visible",
        "price": "100",
        "price_html": "<span>₹100</span>",
        "stock_status": "instock",
        "categories": [{"name": "Overlock parts"}],
        "images": [{"id": f"{pid}-{i}", "src": src} for i, src in enumerate(images)],
    }


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_VERSION", "test")
    store = VectorStore(tmp_path / "index", dim=8)
    store.load()
    woo = FakeWoo({})
    FakeEmbedder.inference_calls = 0
    return SyncService(store, FakeEmbedder(), woo), store, woo


# ----------------------------------------------------------------------
# VectorStore
# ----------------------------------------------------------------------
def test_index_survives_a_restart(tmp_path):
    store = VectorStore(tmp_path / "index", dim=4)
    store.load()
    vectors = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    records = [
        ImageRecord("7", "7-0", "https://img/a.jpg", "hash-a", "v", "m"),
        ImageRecord("7", "7-1", "https://img/b.jpg", "hash-b", "v", "m"),
    ]
    store.upsert_product({"id": "7", "name": "Looper"}, vectors, records)
    store.save()

    reopened = VectorStore(tmp_path / "index", dim=4)
    reopened.load()
    assert reopened.image_count == 2
    assert reopened.product_count == 1
    assert reopened.get_product("7")["name"] == "Looper"


def test_upsert_is_idempotent(tmp_path):
    """Re-processing a product must replace its rows, never duplicate them."""
    store = VectorStore(tmp_path / "index", dim=4)
    store.load()
    vectors = np.array([[1, 0, 0, 0]], dtype=np.float32)
    records = [ImageRecord("7", "7-0", "https://img/a.jpg", "hash-a", "v", "m")]

    for _ in range(5):
        store.upsert_product({"id": "7", "name": "Looper"}, vectors, records)

    assert store.image_count == 1


def test_search_ranks_by_similarity(tmp_path):
    store = VectorStore(tmp_path / "index", dim=4)
    store.load()
    store.upsert_product(
        {"id": "1", "name": "A"},
        np.array([[1, 0, 0, 0]], dtype=np.float32),
        [ImageRecord("1", "1-0", "a", "ha", "v", "m")],
    )
    store.upsert_product(
        {"id": "2", "name": "B"},
        np.array([[0, 1, 0, 0]], dtype=np.float32),
        [ImageRecord("2", "2-0", "b", "hb", "v", "m")],
    )

    results = store.search(np.array([0.9, 0.1, 0, 0], dtype=np.float32), top_k=2)
    assert [r.product_id for r, _ in results] == ["1", "2"]
    assert results[0][1] > results[1][1]


def test_stale_embeddings_are_pruned(tmp_path):
    store = VectorStore(tmp_path / "index", dim=4)
    store.load()
    store.upsert_product(
        {"id": "1", "name": "A"},
        np.array([[1, 0, 0, 0]], dtype=np.float32),
        [ImageRecord("1", "1-0", "a", "ha", "v1", "old-model")],
    )
    assert store.prune_stale_embeddings("v2", "new-model") == 1
    assert store.image_count == 0


# ----------------------------------------------------------------------
# Sync lifecycle — the acceptance test from the brief
# ----------------------------------------------------------------------
def test_new_product_becomes_searchable(service):
    sync, store, woo = service
    woo.products["103"] = raw_product("103", "Needle Bar", ["https://img/103-a.jpg"])

    assert sync.sync_product("103") == "new"
    assert store.product_count == 1
    assert store.get_product("103")["sku"] == "SKU-103"


def test_only_the_new_product_is_embedded(service):
    """Adding product 103 to a catalogue of 102 must cost one embedding."""
    sync, store, woo = service
    for i in range(102):
        woo.products[str(i)] = raw_product(str(i), f"Part {i}", [f"https://img/{i}.jpg"])

    first = sync.sync_all()
    assert first.products_found == 102
    assert first.new_products == 102
    baseline = FakeEmbedder.inference_calls
    assert baseline == 102

    woo.products["103"] = raw_product("103", "Needle Bar", ["https://img/103.jpg"])
    second = sync.sync_all()

    assert second.new_products == 1
    assert second.already_indexed == 102
    assert second.embeddings_generated == 1
    assert FakeEmbedder.inference_calls == baseline + 1


def test_changed_image_is_reembedded(service):
    sync, store, woo = service
    woo.products["103"] = raw_product("103", "Needle Bar", ["https://img/103-a.jpg"])
    sync.sync_product("103")
    before = store.iter_product_vectors("103")[0][1].copy()

    woo.products["103"]["images"] = [{"id": "103-0", "src": "https://img/103-b.jpg"}]
    assert sync.sync_product("103") == "updated"

    after = store.iter_product_vectors("103")[0][1]
    assert not np.allclose(before, after)
    assert store.image_count == 1     # replaced, not appended


def test_unpublished_product_disappears(service):
    sync, store, woo = service
    woo.products["103"] = raw_product("103", "Needle Bar", ["https://img/103-a.jpg"])
    sync.sync_product("103")
    assert store.product_count == 1

    woo.products["103"]["status"] = "draft"
    assert sync.sync_product("103") == "removed"
    assert store.product_count == 0
    assert store.image_count == 0


def test_deleted_product_disappears(service):
    sync, store, woo = service
    woo.products["103"] = raw_product("103", "Needle Bar", ["https://img/103-a.jpg"])
    sync.sync_product("103")

    del woo.products["103"]
    assert sync.sync_product("103") == "removed"
    assert store.product_count == 0


def test_sync_all_removes_products_woocommerce_no_longer_returns(service):
    sync, store, woo = service
    woo.products["1"] = raw_product("1", "A", ["https://img/1.jpg"])
    woo.products["2"] = raw_product("2", "B", ["https://img/2.jpg"])
    sync.sync_all()
    assert store.product_count == 2

    del woo.products["2"]
    report = sync.sync_all()
    assert report.removed_products == 1
    assert store.product_count == 1


def test_corrupt_image_is_skipped_not_fatal(service):
    """One bad image must not stop the rest of the catalogue indexing."""
    sync, store, woo = service

    class Flaky(FakeEmbedder):
        def embed_image(self, image):
            if b"bad" in image:
                raise ValueError("corrupt image")
            return super().embed_image(image)

    sync.embedder = Flaky()
    woo.products["1"] = raw_product("1", "A", ["https://img/bad.jpg", "https://img/good.jpg"])

    report = sync.sync_all()
    assert store.product_count == 1
    assert store.image_count == 1          # only the good image
    assert report.images_skipped == 1


def test_product_with_no_images_is_not_indexed(service):
    sync, store, woo = service
    woo.products["1"] = raw_product("1", "A", [])
    sync.sync_all()
    assert store.product_count == 0
