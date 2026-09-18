"""
WooCommerce REST API client.

WooCommerce is the source of truth. We pull product metadata and image URLs
from it and store only what the search feature needs: id, name, url, sku,
categories, price and image references.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import config

logger = logging.getLogger(__name__)


class WooCommerceError(Exception):
    """WooCommerce could not be reached or returned an unusable response."""


def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=8)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "StitchPartnerAI/2.0"})
    return session


class WooCommerceClient:
    def __init__(
        self,
        base_url: str = config.WOOCOMMERCE_URL,
        consumer_key: str = config.WOOCOMMERCE_CONSUMER_KEY,
        consumer_secret: str = config.WOOCOMMERCE_CONSUMER_SECRET,
        timeout: float = config.WOOCOMMERCE_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth = (consumer_key, consumer_secret)
        self.timeout = timeout
        self.session = _build_session()

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.auth[0] and self.auth[1])

    # ------------------------------------------------------------------
    def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> requests.Response:
        if not self.configured:
            raise WooCommerceError("WooCommerce credentials are not configured.")
        url = f"{self.base_url}/wp-json/wc/v3/{endpoint.lstrip('/')}"
        try:
            response = self.session.get(
                url, params=params or {}, auth=self.auth, timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise WooCommerceError(f"WooCommerce request failed: {exc}") from exc

        if response.status_code == 401:
            raise WooCommerceError("WooCommerce rejected the API credentials (401).")
        if response.status_code >= 400:
            raise WooCommerceError(
                f"WooCommerce returned HTTP {response.status_code} for {endpoint}."
            )
        return response

    # ------------------------------------------------------------------
    def iter_products(self, status: str = "publish") -> Iterator[dict[str, Any]]:
        """
        Yield every published product, one page at a time.

        Uses orderby=id so pagination stays stable even if products are edited
        while the sync is running.
        """
        page = 1
        while True:
            response = self._get(
                "products",
                {
                    "per_page": config.WOOCOMMERCE_PAGE_SIZE,
                    "page": page,
                    "status": status,
                    "orderby": "id",
                    "order": "asc",
                },
            )
            batch = response.json()
            if not isinstance(batch, list) or not batch:
                return
            for product in batch:
                yield product

            total_pages = int(response.headers.get("X-WP-TotalPages", page) or page)
            if page >= total_pages:
                return
            page += 1

    def get_product(self, product_id: str | int) -> dict[str, Any] | None:
        try:
            response = self._get(f"products/{product_id}")
        except WooCommerceError as exc:
            if "404" in str(exc):
                return None
            raise
        product = response.json()
        return product if isinstance(product, dict) else None

    def download_image(self, url: str) -> bytes:
        try:
            response = self.session.get(
                url, timeout=config.IMAGE_DOWNLOAD_TIMEOUT, stream=True
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise WooCommerceError(f"Image download failed: {exc}") from exc

        chunks, total = [], 0
        for chunk in response.iter_content(chunk_size=65536):
            total += len(chunk)
            if total > config.MAX_UPLOAD_SIZE * 4:
                raise WooCommerceError("Product image is unreasonably large.")
            chunks.append(chunk)
        return b"".join(chunks)


# ----------------------------------------------------------------------
# Normalisation: keep only the fields the search feature actually needs.
# ----------------------------------------------------------------------
def normalise_product(raw: dict[str, Any]) -> dict[str, Any]:
    images = []
    seen: set[str] = set()
    for image in raw.get("images") or []:
        src = (image.get("src") or "").strip()
        if not src or src in seen:
            continue
        seen.add(src)
        images.append({"id": str(image.get("id") or src), "src": src})
        if len(images) >= config.MAX_IMAGES_PER_PRODUCT:
            break

    return {
        "id": str(raw.get("id", "")),
        "name": (raw.get("name") or "").strip(),
        "url": raw.get("permalink") or "",
        "sku": (raw.get("sku") or "").strip(),
        "status": raw.get("status") or "",
        "catalog_visibility": raw.get("catalog_visibility") or "visible",
        "price": raw.get("price") or "",
        "price_html": raw.get("price_html") or "",
        "stock_status": raw.get("stock_status") or "",
        "categories": [c.get("name") for c in (raw.get("categories") or []) if c.get("name")],
        "brand": _extract_brand(raw),
        "images": images,
        "primary_image": images[0]["src"] if images else "",
        "date_modified": raw.get("date_modified_gmt") or raw.get("date_modified") or "",
    }


def _extract_brand(raw: dict[str, Any]) -> str:
    """Brand lives in different places depending on the plugins installed."""
    brands = raw.get("brands")
    if isinstance(brands, list) and brands:
        first = brands[0]
        if isinstance(first, dict) and first.get("name"):
            return str(first["name"])
    for attribute in raw.get("attributes") or []:
        name = (attribute.get("name") or "").strip().lower()
        if name in {"brand", "make", "manufacturer"}:
            options = attribute.get("options") or []
            if options:
                return str(options[0])
    return ""


def is_searchable(product: dict[str, Any]) -> bool:
    """Only published, visible products with at least one image belong in the index."""
    return (
        product.get("status") == "publish"
        and product.get("catalog_visibility") in {"visible", "catalog"}
        and bool(product.get("images"))
    )
