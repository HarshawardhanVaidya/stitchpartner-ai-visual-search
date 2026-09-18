#!/usr/bin/env python3
"""
Import the existing committed index into the new store — without re-embedding.

The old system shipped `product_database/woocommerce_embeddings.npy` (189
vectors) alongside `woocommerce_image_records.json` and
`woocommerce_product_groups.json`. Those embeddings are still valid: they came
from the same model and the same CLS pooling. This script carries them across
so the search feature keeps working from the first boot, instead of going dark
until the first sync finishes.

    python scripts/migrate_legacy_index.py --source product_database

Afterwards, run a normal sync. Products whose WooCommerce images are unchanged
will be recognised by hash and cost no inference; only genuinely new or changed
products are embedded.

Note: the legacy records carry no image hashes, so the first sync re-embeds each
migrated image once to establish a fingerprint. That is a one-off cost and it is
still far cheaper than an outage.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from vector_store import ImageRecord, VectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("migrate")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="product_database",
                        help="Directory holding the legacy .npy / .json files.")
    parser.add_argument("--dest", default=None,
                        help="Index directory to write to (defaults to INDEX_PATH).")
    args = parser.parse_args()

    source = Path(args.source)
    embeddings_file = source / "woocommerce_embeddings.npy"
    records_file = source / "woocommerce_image_records.json"
    groups_file = source / "woocommerce_product_groups.json"

    for path in (embeddings_file, records_file, groups_file):
        if not path.exists():
            logger.error("Missing %s", path)
            return 2

    vectors = np.load(embeddings_file, allow_pickle=False).astype(np.float32)
    legacy_records = json.loads(records_file.read_text(encoding="utf-8"))
    legacy_groups = json.loads(groups_file.read_text(encoding="utf-8"))

    if len(legacy_records) != vectors.shape[0]:
        logger.error(
            "Legacy files disagree: %d vectors vs %d records. Aborting.",
            vectors.shape[0], len(legacy_records),
        )
        return 3

    records: list[ImageRecord] = []
    for entry in legacy_records:
        image_url = entry.get("image_url", "")
        records.append(
            ImageRecord(
                product_id=str(entry.get("product_id", "")),
                image_id=image_url or entry.get("local_path", ""),
                image_url=image_url,
                image_hash="",                       # unknown; first sync fills it in
                embedding_version=config.EMBEDDING_VERSION,
                model_name=config.MODEL_NAME,
            )
        )

    products: dict[str, dict] = {}
    for product_id, group in legacy_groups.items():
        products[str(product_id)] = {
            "id": str(product_id),
            "name": group.get("name", ""),
            "url": group.get("url", ""),
            "sku": "",
            "status": "publish",
            "catalog_visibility": "visible",
            "price": "",
            "price_html": "",
            "stock_status": "",
            "categories": [],
            "brand": "",
            "images": [],
            "primary_image": "",
            "date_modified": "",
        }

    # Fill primary_image from the records we do have.
    for record in records:
        product = products.get(record.product_id)
        if product is not None and not product["primary_image"] and record.image_url:
            product["primary_image"] = record.image_url

    dest = Path(args.dest) if args.dest else config.INDEX_PATH
    store = VectorStore(dest, dim=vectors.shape[1])
    store.replace_all(vectors, records, products)
    store.set_manifest_value("migrated_from", str(source))
    store.save()

    logger.info(
        "Migrated %d images across %d products into %s",
        store.image_count, store.product_count, dest,
    )
    logger.info("Next: run a sync so WooCommerce metadata and image hashes are filled in.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
