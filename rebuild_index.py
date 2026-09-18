#!/usr/bin/env python3
"""
Full index rebuild. Run deliberately — never from the Railway start command.

    python rebuild_index.py            # rebuild into a staging directory, then swap
    python rebuild_index.py --dry-run  # build and validate, but keep the live index

The new index is built in a separate directory and only swapped into place once
it has been validated. If the rebuild fails at any point, the existing working
index is left exactly as it was.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

import config
from embedder import embedder
from sync import SyncService
from vector_store import VectorStore
from woocommerce import WooCommerceClient, WooCommerceError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("rebuild")


def validate(store: VectorStore) -> list[str]:
    """Refuse to promote an index that is worse than useless."""
    problems: list[str] = []
    if store.image_count == 0:
        problems.append("The rebuilt index contains no images.")
    if store.product_count == 0:
        problems.append("The rebuilt index contains no products.")
    orphans = store.prune_orphans()
    if orphans:
        logger.warning("Removed %d image rows with no matching product.", orphans)
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild the StitchPartnerAI index.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build and validate without replacing the live index.")
    parser.add_argument("--min-products", type=int, default=1,
                        help="Refuse to promote an index with fewer products than this.")
    args = parser.parse_args()

    config.ensure_directories()

    if not config.woocommerce_configured():
        logger.error("WooCommerce credentials are not configured. Nothing to rebuild from.")
        return 2

    live_path = config.INDEX_PATH
    staging_path = live_path.parent / (live_path.name + ".rebuilding")
    backup_path = live_path.parent / (live_path.name + ".previous")

    if staging_path.exists():
        shutil.rmtree(staging_path)
    staging_path.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    logger.info("Rebuilding into %s", staging_path)

    try:
        embedder.load()
    except Exception:
        logger.exception("Could not load the embedding model.")
        shutil.rmtree(staging_path, ignore_errors=True)
        return 3

    staging = VectorStore(staging_path, dim=embedder.dim)
    staging.load()   # empty, but sets up the directory

    service = SyncService(staging, embedder, WooCommerceClient())

    try:
        report = service.sync_all()
    except WooCommerceError:
        logger.exception("WooCommerce was unreachable. The live index is untouched.")
        shutil.rmtree(staging_path, ignore_errors=True)
        return 4

    if report.status != "completed":
        logger.error("Rebuild did not complete (%s). The live index is untouched.", report.status)
        for error in report.errors[:10]:
            logger.error("  %s", error)
        shutil.rmtree(staging_path, ignore_errors=True)
        return 5

    problems = validate(staging)
    if staging.product_count < args.min_products:
        problems.append(
            f"Only {staging.product_count} products indexed; "
            f"--min-products is {args.min_products}."
        )

    if problems:
        logger.error("Validation failed. The live index is untouched:")
        for problem in problems:
            logger.error("  %s", problem)
        shutil.rmtree(staging_path, ignore_errors=True)
        return 6

    staging.save()

    logger.info(
        "Rebuilt %d products / %d images in %.1fs.",
        staging.product_count, staging.image_count, time.perf_counter() - started,
    )

    if args.dry_run:
        logger.info("--dry-run: staging index left at %s, live index untouched.", staging_path)
        return 0

    # Atomic-ish swap: keep the old index as .previous until the new one is in place.
    if backup_path.exists():
        shutil.rmtree(backup_path)
    if live_path.exists():
        live_path.rename(backup_path)
    try:
        staging_path.rename(live_path)
    except Exception:
        logger.exception("Swap failed; restoring the previous index.")
        if backup_path.exists() and not live_path.exists():
            backup_path.rename(live_path)
        return 7

    logger.info("New index is live. Previous index kept at %s.", backup_path)
    logger.info("Restart the application (or redeploy) to load it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
