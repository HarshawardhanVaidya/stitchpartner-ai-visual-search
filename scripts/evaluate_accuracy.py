#!/usr/bin/env python3
"""
Measure top-1 / top-3 / top-5 accuracy against a labelled test set.

Nothing in this project should claim an accuracy improvement without a run of
this script before and after the change.

Layout — one folder per expected WooCommerce product ID:

    tests/dataset/
        141/            <- WooCommerce product ID
            exact.jpg
            angle.jpg
            phone_photo.jpg
            blurry.jpg
        135/
            ...
        _no_match/      <- images that must NOT match anything
            keyboard.jpg
            coffee_cup.jpg

Run:

    python scripts/evaluate_accuracy.py --dataset tests/dataset
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from embedder import embedder
from search import SearchService
from vector_store import VectorStore

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
NO_MATCH_DIR = "_no_match"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tests/dataset")
    parser.add_argument("--verbose", action="store_true", help="List every failure.")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    if not dataset.exists():
        print(f"No dataset at {dataset}")
        return 2

    embedder.load()
    store = VectorStore(config.INDEX_PATH)
    store.load()
    if store.image_count == 0:
        print("The index is empty. Run a sync first.")
        return 3

    service = SearchService(store, embedder)

    hits = Counter()
    total = 0
    failures: list[str] = []

    no_match_total = 0
    no_match_correct = 0

    for folder in sorted(dataset.iterdir()):
        if not folder.is_dir():
            continue
        expected = folder.name

        for image_path in sorted(folder.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            response = service.search(image_path.read_bytes())
            ranked = [r["product_id"] for r in response.get("results", [])]

            if expected == NO_MATCH_DIR:
                no_match_total += 1
                if not response.get("matched"):
                    no_match_correct += 1
                elif args.verbose:
                    failures.append(
                        f"  false positive: {image_path.name} -> "
                        f"{response['results'][0]['name']}"
                    )
                continue

            total += 1
            for k in (1, 3, 5):
                if expected in ranked[:k]:
                    hits[k] += 1
            if expected not in ranked[:5]:
                failures.append(f"  miss: {folder.name}/{image_path.name} -> {ranked[:3]}")

    print(f"\nEvaluated {total} labelled images across the catalogue.")
    if total:
        for k in (1, 3, 5):
            print(f"  top-{k} accuracy: {hits[k]/total:6.1%}  ({hits[k]}/{total})")

    if no_match_total:
        print(
            f"\n  correct 'no match' rate: {no_match_correct/no_match_total:6.1%}  "
            f"({no_match_correct}/{no_match_total})"
        )

    if failures:
        print(f"\n{len(failures)} failure(s):")
        for line in failures[:40]:
            print(line)

    print(
        f"\nSettings: MIN_MATCH_SCORE={config.MIN_MATCH_SCORE} "
        f"POOLING={config.POOLING} PREPROCESS_MAX_DIM={config.PREPROCESS_MAX_DIM}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
