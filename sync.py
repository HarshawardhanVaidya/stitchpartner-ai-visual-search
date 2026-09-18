"""
Incremental WooCommerce -> index synchronisation.

The rule that makes this cheap: a product image is only downloaded and embedded
when its content hash is not already in the index. Adding one product to a
catalogue of a thousand therefore costs one embedding, not a thousand.

    WooCommerce -> normalise -> diff against index -> embed only what changed

Because the unit of work is "replace everything for product X", processing the
same product twice is a no-op rather than a source of duplicate embeddings.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

import config
from embedder import Embedder, ImageDecodeError
from vector_store import ImageRecord, VectorStore
from woocommerce import WooCommerceClient, WooCommerceError, is_searchable, normalise_product

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class SyncReport:
    """The numbers an administrator needs to debug a sync."""

    started_at: str = field(default_factory=_utc_now)
    finished_at: str = ""
    duration_seconds: float = 0.0
    products_found: int = 0
    already_indexed: int = 0
    new_products: int = 0
    updated_products: int = 0
    removed_products: int = 0
    embeddings_generated: int = 0
    images_skipped: int = 0
    errors: list[str] = field(default_factory=list)
    status: str = "running"

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["error_count"] = len(self.errors)
        data["errors"] = self.errors[:20]   # keep the payload small
        return data


class SyncService:
    def __init__(self, store: VectorStore, embedder: Embedder, client: WooCommerceClient) -> None:
        self.store = store
        self.embedder = embedder
        self.client = client
        self._lock = threading.Lock()        # only one sync at a time
        self._running = False
        self._last_report: SyncReport | None = None
        self._stop_event = threading.Event()
        self._scheduler: threading.Thread | None = None

    # ------------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def last_report(self) -> dict[str, Any] | None:
        return self._last_report.as_dict() if self._last_report else None

    # ------------------------------------------------------------------
    # Single product
    # ------------------------------------------------------------------
    def sync_product(self, product_id: str | int, report: SyncReport | None = None) -> str:
        """
        Bring one product into line with WooCommerce.

        Returns one of: "new", "updated", "unchanged", "removed", "error".
        """
        product_id = str(product_id)
        report = report or SyncReport()

        try:
            raw = self.client.get_product(product_id)
        except WooCommerceError as exc:
            report.errors.append(f"{product_id}: {exc}")
            return "error"

        if raw is None:
            # Deleted in WooCommerce.
            return "removed" if self.store.remove_product(product_id) else "unchanged"

        product = normalise_product(raw)
        if not is_searchable(product):
            # Unpublished, hidden, or has no image — it must stop appearing.
            return "removed" if self.store.remove_product(product_id) else "unchanged"

        return self._index_product(product, report)

    # ------------------------------------------------------------------
    def _index_product(self, product: dict[str, Any], report: SyncReport) -> str:
        product_id = product["id"]
        existing = self.store.get_product(product_id)
        known_hashes = self.store.image_fingerprints(product_id)

        vectors: list[np.ndarray] = []
        records: list[ImageRecord] = []
        reused = 0

        for image in product["images"]:
            image_id, src = image["id"], image["src"]
            try:
                data = self.client.download_image(src)
            except WooCommerceError as exc:
                report.errors.append(f"{product_id} image {image_id}: {exc}")
                report.images_skipped += 1
                continue

            image_hash = hashlib.sha256(data).hexdigest()

            # Identical bytes already embedded under this product: reuse the
            # stored vector rather than paying for inference again.
            cached = self._cached_vector(product_id, image_id, image_hash, known_hashes)
            if cached is not None:
                vectors.append(cached)
                records.append(
                    ImageRecord(
                        product_id=product_id,
                        image_id=image_id,
                        image_url=src,
                        image_hash=image_hash,
                        embedding_version=config.EMBEDDING_VERSION,
                        model_name=self.embedder.model_name,
                    )
                )
                reused += 1
                continue

            try:
                decoded = self.embedder.decode(data)
                vector = self.embedder.embed_image(decoded)
            except Exception as exc:   # corrupt or undecodable image: skip, keep going
                report.errors.append(f"{product_id} image {image_id}: {exc}")
                report.images_skipped += 1
                continue

            vectors.append(vector)
            records.append(
                ImageRecord(
                    product_id=product_id,
                    image_id=image_id,
                    image_url=src,
                    image_hash=image_hash,
                    embedding_version=config.EMBEDDING_VERSION,
                    model_name=self.embedder.model_name,
                )
            )
            report.embeddings_generated += 1

        if not records:
            report.errors.append(f"{product_id}: no usable images.")
            # Never leave a product half-indexed.
            self.store.remove_product(product_id)
            return "error"

        matrix = np.vstack([v.reshape(1, -1) for v in vectors])
        self.store.upsert_product(product, matrix, records)

        if existing is None:
            return "new"
        if reused == len(records) and self._metadata_equal(existing, product):
            return "unchanged"
        return "updated"

    def _cached_vector(
        self,
        product_id: str,
        image_id: str,
        image_hash: str,
        known_hashes: dict[str, str],
    ) -> np.ndarray | None:
        """Return the stored vector if this exact image is already indexed."""
        if known_hashes.get(image_id) != image_hash:
            return None
        for record, vector in self.store.iter_product_vectors(product_id):
            if record.image_id == image_id and record.image_hash == image_hash:
                if (
                    record.embedding_version == config.EMBEDDING_VERSION
                    and record.model_name == self.embedder.model_name
                ):
                    return vector
        return None

    @staticmethod
    def _metadata_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
        fields = ("name", "url", "sku", "price", "stock_status", "categories", "brand")
        return all(a.get(f) == b.get(f) for f in fields)

    # ------------------------------------------------------------------
    # Full catalogue sync
    # ------------------------------------------------------------------
    def sync_all(self) -> SyncReport:
        """
        Reconcile the whole catalogue: add new products, re-embed changed
        images, remove products that disappeared or were unpublished.

        This is the safety net behind the webhook, not a rebuild — products
        whose images are unchanged cost one HTTP request and no inference.
        """
        if not self._lock.acquire(blocking=False):
            logger.info("Sync already in progress; skipping this run.")
            return self._last_report or SyncReport(status="skipped")

        report = SyncReport()
        self._running = True
        started = time.perf_counter()
        logger.info("WooCommerce sync started.")

        try:
            seen: set[str] = set()
            for raw in self.client.iter_products():
                product = normalise_product(raw)
                if not product["id"]:
                    continue
                report.products_found += 1
                seen.add(product["id"])

                if not is_searchable(product):
                    if self.store.remove_product(product["id"]):
                        report.removed_products += 1
                    continue

                try:
                    outcome = self._index_product(product, report)
                except Exception as exc:
                    logger.exception("Failed to index product %s", product["id"])
                    report.errors.append(f"{product['id']}: {exc}")
                    continue

                if outcome == "new":
                    report.new_products += 1
                elif outcome == "updated":
                    report.updated_products += 1
                elif outcome == "unchanged":
                    report.already_indexed += 1

            # Anything in the index that WooCommerce no longer returns is gone.
            for stale_id in self.store.indexed_product_ids() - seen:
                if self.store.remove_product(stale_id):
                    report.removed_products += 1

            self.store.set_manifest_value("last_sync_at", _utc_now())
            self.store.save()
            report.status = "completed"

        except WooCommerceError as exc:
            # WooCommerce being down must not damage the working index.
            logger.error("Sync aborted: %s", exc)
            report.errors.append(str(exc))
            report.status = "failed"
        except Exception as exc:
            logger.exception("Sync aborted with an unexpected error.")
            report.errors.append(str(exc))
            report.status = "failed"
        finally:
            report.duration_seconds = round(time.perf_counter() - started, 2)
            report.finished_at = _utc_now()
            self._last_report = report
            self._running = False
            self._lock.release()

        logger.info(
            "WooCommerce sync %s in %.1fs | found=%d new=%d updated=%d "
            "unchanged=%d removed=%d embeddings=%d skipped=%d errors=%d",
            report.status, report.duration_seconds, report.products_found,
            report.new_products, report.updated_products, report.already_indexed,
            report.removed_products, report.embeddings_generated,
            report.images_skipped, len(report.errors),
        )
        return report

    # ------------------------------------------------------------------
    # Background scheduler — the safety net if a webhook is missed
    # ------------------------------------------------------------------
    def start_scheduler(self) -> None:
        if config.SYNC_INTERVAL_MINUTES <= 0:
            logger.info("Periodic sync disabled (SYNC_INTERVAL_MINUTES=0).")
            return
        if self._scheduler and self._scheduler.is_alive():
            return

        def loop() -> None:
            interval = config.SYNC_INTERVAL_MINUTES * 60
            if not config.SYNC_ON_STARTUP:
                # Never make a deploy wait on a catalogue sync.
                self._stop_event.wait(interval)
            while not self._stop_event.is_set():
                try:
                    self.sync_all()
                except Exception:
                    logger.exception("Scheduled sync failed.")
                self._stop_event.wait(interval)

        self._scheduler = threading.Thread(target=loop, daemon=True, name="woo-sync")
        self._scheduler.start()
        logger.info("Periodic sync every %d minutes.", config.SYNC_INTERVAL_MINUTES)

    def stop_scheduler(self) -> None:
        self._stop_event.set()

    def run_in_background(self, product_id: str | int | None = None) -> None:
        """Fire-and-forget sync so webhook and admin calls return immediately."""

        def task() -> None:
            try:
                if product_id is None:
                    self.sync_all()
                else:
                    report = SyncReport()
                    outcome = self.sync_product(product_id, report)
                    self.store.save()
                    logger.info("Webhook sync for product %s: %s", product_id, outcome)
            except Exception:
                logger.exception("Background sync failed.")

        threading.Thread(target=task, daemon=True).start()
