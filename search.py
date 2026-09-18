"""
Search pipeline: query embedding -> nearest images -> products -> response.

Two deliberate carry-overs from the original implementation, because they were
working and there is no evidence against them:

  * multi-view scoring — a product matching on several of its photos is a
    better answer than one matching on a single lucky angle;
  * a similarity floor — below it we say we found nothing rather than showing
    unrelated machinery.

The change from the original: results are keyed on the WooCommerce product ID,
not the product name. Name-keyed lookup silently broke whenever a product was
renamed, and collided whenever two products shared a name.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import config
from embedder import Embedder
from vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class SearchTimings:
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    vector_search_ms: float = 0.0
    total_ms: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "preprocess_ms": round(self.preprocess_ms, 1),
            "inference_ms": round(self.inference_ms, 1),
            "vector_search_ms": round(self.vector_search_ms, 2),
            "total_ms": round(self.total_ms, 1),
        }


class SearchService:
    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    # ------------------------------------------------------------------
    def search(self, image_bytes: bytes) -> dict[str, Any]:
        started = time.perf_counter()
        timings = SearchTimings()

        embedding = self.embedder.embed_bytes(image_bytes)
        timings.preprocess_ms = embedding.preprocess_ms
        timings.inference_ms = embedding.inference_ms

        search_started = time.perf_counter()
        matches = self.store.search(embedding.vector, config.TOP_K_IMAGES)
        timings.vector_search_ms = (time.perf_counter() - search_started) * 1000

        products = self._group_by_product(matches)
        timings.total_ms = (time.perf_counter() - started) * 1000

        logger.info(
            "Search complete | preprocess=%.0fms inference=%.0fms "
            "vector_search=%.2fms total=%.0fms candidates=%d products=%d",
            timings.preprocess_ms, timings.inference_ms, timings.vector_search_ms,
            timings.total_ms, len(matches), len(products),
        )

        best_score = products[0]["_score"] if products else 0.0

        if not products or best_score < config.MIN_MATCH_SCORE:
            return {
                "success": True,
                "matched": False,
                "message": "We couldn't find a close match.",
                "suggestions": [
                    "Take a clearer photo",
                    "Use better lighting",
                    "Photograph one part at a time",
                ],
                "results": [],
                "timings": timings.as_dict(),
            }

        return {
            "success": True,
            "matched": True,
            "message": (
                "Here are the closest matches."
                if best_score >= config.STRONG_MATCH_SCORE
                else "These look similar. Check the photos to confirm."
            ),
            "confident": best_score >= config.STRONG_MATCH_SCORE,
            "results": [self._to_public(p) for p in products[: config.TOP_K_PRODUCTS]],
            "timings": timings.as_dict(),
        }

    # ------------------------------------------------------------------
    def _group_by_product(self, matches) -> list[dict[str, Any]]:
        grouped: dict[str, list[float]] = {}
        best_image: dict[str, str] = {}

        for record, score in matches:
            grouped.setdefault(record.product_id, []).append(score)
            best_image.setdefault(record.product_id, record.image_url)

        results: list[dict[str, Any]] = []
        for product_id, scores in grouped.items():
            product = self.store.get_product(product_id)
            if not product:
                # Index and metadata drifted apart; the next sync repairs it.
                continue

            scores.sort(reverse=True)
            best = scores[0]
            second = scores[1] if len(scores) > 1 else 0.0

            # Agreement across several views of the same product is real
            # evidence; cap the bonus so it can never outweigh a strong match.
            view_bonus = min(len(scores) - 1, 3) * 0.035
            consistency_bonus = (second / best) * 0.10 if len(scores) > 1 and best > 0 else 0.0
            score = best * 0.75 + second * 0.15 + view_bonus + consistency_bonus

            if len(scores) == 1 and best < 0.60:
                score *= 0.90

            results.append(
                {
                    "_score": score,
                    "_best_image_score": best,
                    "_views": len(scores),
                    "product": product,
                    "matched_image": best_image.get(product_id, ""),
                }
            )

        results.sort(key=lambda r: r["_score"], reverse=True)
        return results

    # ------------------------------------------------------------------
    @staticmethod
    def _to_public(entry: dict[str, Any]) -> dict[str, Any]:
        """
        Shape one result for the customer.

        Raw cosine scores are not exposed: they are uncalibrated, and a
        customer reading "0.61" learns nothing useful. A coarse band is enough
        to drive the UI, and the exact figure stays in the server logs.
        """
        product = entry["product"]
        score = entry["_score"]
        if score >= config.STRONG_MATCH_SCORE:
            band = "strong"
        elif score >= (config.MIN_MATCH_SCORE + config.STRONG_MATCH_SCORE) / 2:
            band = "close"
        else:
            band = "possible"

        return {
            "product_id": product.get("id", ""),
            "name": product.get("name", ""),
            "sku": product.get("sku", ""),
            "url": product.get("url", ""),
            "image": entry.get("matched_image") or product.get("primary_image", ""),
            "price_html": product.get("price_html", ""),
            "categories": product.get("categories", []),
            "brand": product.get("brand", ""),
            "stock_status": product.get("stock_status", ""),
            "match": band,
        }

    # ------------------------------------------------------------------
    def similar_to_product(self, product_id: str, limit: int = 8) -> list[dict[str, Any]]:
        """
        'You may also like', powered by the same index.

        Uses the product's own stored embeddings as the query, so it costs a
        dot product and no inference at all.
        """
        stored = self.store.iter_product_vectors(product_id)
        if not stored:
            return []

        _, vector = stored[0]
        matches = [
            (record, score)
            for record, score in self.store.search(vector, config.TOP_K_IMAGES)
            if record.product_id != str(product_id)
        ]
        return [self._to_public(p) for p in self._group_by_product(matches)[:limit]]
