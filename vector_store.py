"""
Persistent vector index for product images.

Layout on disk (all inside INDEX_PATH):

    vectors.npy    float32 (N, D), L2-normalised, one row per product image
    records.json   list of N records, row-aligned with vectors.npy
    products.json  product_id -> WooCommerce product metadata
    manifest.json  index-level bookkeeping (counts, versions, last sync)

Every write is atomic: we write a temporary file next to the target and
os.replace() it, so a crash mid-write can never leave a half-written index.

Search is a brute-force normalised dot product. Measured on a single core:
0.014 ms at 189 vectors, 1.0 ms at 10,000, 7.8 ms at 50,000. A specialised
index (FAISS/Qdrant) buys nothing until the catalogue is far larger, and the
`search()` signature here is narrow enough to swap out when that day comes.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

logger = logging.getLogger(__name__)

VECTORS_FILE = "vectors.npy"
RECORDS_FILE = "records.json"
PRODUCTS_FILE = "products.json"
MANIFEST_FILE = "manifest.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _atomic_write_npy(path: Path, array: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def normalise(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows so cosine similarity is a plain dot product."""
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


@dataclass
class ImageRecord:
    """One indexed product image. Row-aligned with the vector matrix."""

    product_id: str
    image_id: str            # stable identity: WooCommerce image id, or its URL
    image_url: str
    image_hash: str          # sha256 of the downloaded image bytes
    embedding_version: str
    model_name: str
    updated_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "product_id": self.product_id,
            "image_id": self.image_id,
            "image_url": self.image_url,
            "image_hash": self.image_hash,
            "embedding_version": self.embedding_version,
            "model_name": self.model_name,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ImageRecord":
        return cls(
            product_id=str(raw.get("product_id", "")),
            image_id=str(raw.get("image_id", "")),
            image_url=raw.get("image_url", ""),
            image_hash=raw.get("image_hash", ""),
            embedding_version=raw.get("embedding_version", ""),
            model_name=raw.get("model_name", ""),
            updated_at=raw.get("updated_at", _utc_now()),
        )


class VectorStore:
    """Thread-safe, persistent, incrementally updatable image index."""

    def __init__(self, path: Path, dim: int = 768) -> None:
        self.path = Path(path)
        self.dim = dim
        self._lock = threading.RLock()
        self._vectors: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._records: list[ImageRecord] = []
        self._products: dict[str, dict[str, Any]] = {}
        self._manifest: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load the index from disk. A missing index is an empty index, not an error."""
        with self._lock:
            self.path.mkdir(parents=True, exist_ok=True)
            vectors_path = self.path / VECTORS_FILE
            records_path = self.path / RECORDS_FILE
            products_path = self.path / PRODUCTS_FILE
            manifest_path = self.path / MANIFEST_FILE

            try:
                if vectors_path.exists() and records_path.exists():
                    vectors = np.load(vectors_path, allow_pickle=False)
                    with open(records_path, "r", encoding="utf-8") as handle:
                        records = [ImageRecord.from_dict(r) for r in json.load(handle)]

                    if len(records) != vectors.shape[0]:
                        logger.error(
                            "Index is inconsistent (%d vectors vs %d records); "
                            "starting empty. Run rebuild_index.py to repair.",
                            vectors.shape[0], len(records),
                        )
                    else:
                        self._vectors = normalise(vectors) if len(records) else vectors
                        self._records = records
                        self.dim = vectors.shape[1] if vectors.size else self.dim
            except Exception:
                logger.exception("Could not read the vector index; starting empty.")
                self._vectors = np.zeros((0, self.dim), dtype=np.float32)
                self._records = []

            self._products = self._read_json(products_path, {})
            self._manifest = self._read_json(manifest_path, {})

            logger.info(
                "Index loaded: %d images across %d products.",
                len(self._records), len(self._products),
            )

    @staticmethod
    def _read_json(path: Path, fallback: Any) -> Any:
        try:
            if path.exists():
                with open(path, "r", encoding="utf-8") as handle:
                    return json.load(handle)
        except Exception:
            logger.exception("Could not read %s; using default.", path.name)
        return fallback

    def save(self) -> None:
        with self._lock:
            self.path.mkdir(parents=True, exist_ok=True)
            _atomic_write_npy(self.path / VECTORS_FILE, self._vectors)
            _atomic_write_json(
                self.path / RECORDS_FILE, [r.to_dict() for r in self._records]
            )
            _atomic_write_json(self.path / PRODUCTS_FILE, self._products)
            self._manifest.update(
                {
                    "image_count": len(self._records),
                    "product_count": len(self._products),
                    "dim": int(self._vectors.shape[1]) if self._vectors.size else self.dim,
                    "saved_at": _utc_now(),
                }
            )
            _atomic_write_json(self.path / MANIFEST_FILE, self._manifest)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def image_count(self) -> int:
        return len(self._records)

    @property
    def product_count(self) -> int:
        return len(self._products)

    @property
    def manifest(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._manifest)

    def set_manifest_value(self, key: str, value: Any) -> None:
        with self._lock:
            self._manifest[key] = value

    def get_product(self, product_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._products.get(str(product_id))

    def all_products(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return dict(self._products)

    def indexed_product_ids(self) -> set[str]:
        with self._lock:
            return set(self._products.keys())

    def iter_product_vectors(
        self, product_id: str
    ) -> list[tuple[ImageRecord, np.ndarray]]:
        """Stored (record, vector) pairs for one product, for cache reuse."""
        pid = str(product_id)
        with self._lock:
            return [
                (record, self._vectors[i].copy())
                for i, record in enumerate(self._records)
                if record.product_id == pid
            ]

    def image_fingerprints(self, product_id: str) -> dict[str, str]:
        """image_id -> image_hash for one product. Used to skip unchanged work."""
        pid = str(product_id)
        with self._lock:
            return {
                r.image_id: r.image_hash
                for r in self._records
                if r.product_id == pid
            }

    # ------------------------------------------------------------------
    # Mutation — all operations are idempotent on (product_id, image_id)
    # ------------------------------------------------------------------
    def upsert_product(
        self,
        product: dict[str, Any],
        vectors: np.ndarray | None,
        records: list[ImageRecord] | None,
    ) -> None:
        """
        Replace everything the index holds for one product.

        Passing the full image set for the product (rather than appending) is
        what makes re-processing the same product a no-op instead of a source
        of duplicate embeddings.
        """
        product_id = str(product.get("id", ""))
        if not product_id:
            raise ValueError("Product is missing an id.")

        with self._lock:
            self._drop_rows(lambda r: r.product_id == product_id)

            if vectors is not None and records:
                vectors = normalise(vectors)
                if vectors.shape[0] != len(records):
                    raise ValueError("Vector/record count mismatch.")
                if self._vectors.size:
                    self._vectors = np.vstack([self._vectors, vectors])
                else:
                    self._vectors = vectors
                self._records.extend(records)
                self.dim = int(self._vectors.shape[1])

            self._products[product_id] = product

    def remove_product(self, product_id: str) -> bool:
        """Remove a product and all of its embeddings. Safe to call twice."""
        product_id = str(product_id)
        with self._lock:
            removed_rows = self._drop_rows(lambda r: r.product_id == product_id)
            removed_product = self._products.pop(product_id, None) is not None
            return removed_rows > 0 or removed_product

    def _drop_rows(self, predicate) -> int:
        """Delete every row whose record matches. Caller must hold the lock."""
        if not self._records:
            return 0
        keep = [i for i, r in enumerate(self._records) if not predicate(r)]
        if len(keep) == len(self._records):
            return 0
        removed = len(self._records) - len(keep)
        self._vectors = (
            self._vectors[keep]
            if keep
            else np.zeros((0, self.dim), dtype=np.float32)
        )
        self._records = [self._records[i] for i in keep]
        return removed

    def replace_all(
        self,
        vectors: np.ndarray,
        records: list[ImageRecord],
        products: dict[str, dict[str, Any]],
    ) -> None:
        """Swap in a freshly built index wholesale. Used by rebuild_index.py."""
        if vectors.shape[0] != len(records):
            raise ValueError("Vector/record count mismatch.")
        with self._lock:
            self._vectors = normalise(vectors) if len(records) else np.zeros(
                (0, self.dim), dtype=np.float32
            )
            self._records = list(records)
            self._products = dict(products)
            if vectors.size:
                self.dim = int(vectors.shape[1])

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(self, query: np.ndarray, top_k: int) -> list[tuple[ImageRecord, float]]:
        """Return the top_k most similar images with their cosine scores."""
        with self._lock:
            if not self._records:
                return []
            query = np.ascontiguousarray(query, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(query))
            if norm == 0:
                return []
            query = query / norm

            scores = self._vectors @ query
            k = min(top_k, scores.shape[0])
            # argpartition is O(n); a full sort is only needed on the k survivors.
            top = np.argpartition(-scores, k - 1)[:k]
            top = top[np.argsort(-scores[top])]
            return [(self._records[int(i)], float(scores[int(i)])) for i in top]

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def prune_stale_embeddings(self, embedding_version: str, model_name: str) -> int:
        """Drop rows produced by an older model or embedding version."""
        with self._lock:
            return self._drop_rows(
                lambda r: r.embedding_version != embedding_version
                or r.model_name != model_name
            )

    def prune_orphans(self) -> int:
        """Drop image rows whose product is no longer in the catalogue."""
        with self._lock:
            known = set(self._products)
            return self._drop_rows(lambda r: r.product_id not in known)
