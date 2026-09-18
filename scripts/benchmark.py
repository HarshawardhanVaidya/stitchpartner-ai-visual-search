#!/usr/bin/env python3
"""
Measure where the time actually goes. Run this on Railway, not just locally —
the whole point is that the container's CPU allocation is the variable.

    python scripts/benchmark.py --image sample.jpg --runs 5
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from embedder import embedder
from vector_store import VectorStore


def summarise(label: str, samples: list[float]) -> None:
    print(
        f"  {label:<26} min {min(samples):7.1f} ms | "
        f"median {statistics.median(samples):7.1f} ms | max {max(samples):7.1f} ms"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="A representative customer photo.")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"No such image: {image_path}")
        return 2
    data = image_path.read_bytes()
    print(f"Test image: {image_path.name}, {len(data)/1024:.0f} KB")

    import os
    print(f"CPU cores visible: {os.cpu_count()}")

    started = time.perf_counter()
    embedder.load()
    print(f"\nModel load: {time.perf_counter() - started:.1f}s")

    started = time.perf_counter()
    embedder.warmup()
    print(f"Warmup inference: {(time.perf_counter() - started)*1000:.0f} ms")

    store = VectorStore(config.INDEX_PATH)
    store.load()
    print(f"Index: {store.image_count} images, {store.product_count} products\n")

    preprocess, inference, vector_search = [], [], []
    for _ in range(args.runs):
        result = embedder.embed_bytes(data)
        preprocess.append(result.preprocess_ms)
        inference.append(result.inference_ms)

        started = time.perf_counter()
        store.search(result.vector, config.TOP_K_IMAGES)
        vector_search.append((time.perf_counter() - started) * 1000)

    print(f"Over {args.runs} runs (steady state, model already warm):")
    summarise("preprocess", preprocess)
    summarise("model inference", inference)
    summarise("vector search", vector_search)

    totals = [p + i + v for p, i, v in zip(preprocess, inference, vector_search)]
    summarise("total per search", totals)

    median = statistics.median(totals)
    verdict = "ideal" if median <= 3000 else ("acceptable" if median <= 5000 else "TOO SLOW")
    print(f"\nMedian end-to-end: {median:.0f} ms — {verdict}")
    print("Note: this excludes network upload time. Measure that from a phone on 4G.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
