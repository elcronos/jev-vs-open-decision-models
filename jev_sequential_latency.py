"""Supplementary: Jev end-to-end latency measured SEQUENTIALLY (concurrency 1) on validation rows.

The primary run measured Jev latency under 8 concurrent in-flight requests (PROTOCOL.md §3.1), which
can inflate per-request wall-clock through client-side queuing / server-side contention. This script
measures the same request shape one-at-a-time on the first N rows of the *validation* split (never the
test split) so the number is not confounded by our own load. Results -> results/jev_sequential_latency.json.

Usage: OPENROUTER_API_KEY=... python jev_sequential_latency.py --n 100
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from datasets import load_dataset

from models.common import DATASET_CONFIG, DATASET_ID, DATASET_REVISION
from models.jev import JevBackend


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    ds = load_dataset(DATASET_ID, DATASET_CONFIG, split="validation", revision=DATASET_REVISION)
    texts = [ds[i]["text"] for i in range(args.n + args.warmup)]

    # cache_dir under the scratch smoke folder so nothing touches the test cache
    backend = JevBackend(cache_dir=Path("results/cache/seq_latency_validation"), concurrency=1, show_progress=False)
    backend.load()
    for i in range(args.warmup):
        backend.predict_one(-1000 - i, texts[i], "plain")

    lat = []
    errors = 0
    t_wall0 = time.perf_counter()
    for i in range(args.warmup, args.warmup + args.n):
        p = backend.predict_one(-i, texts[i], "plain")
        if p.error:
            errors += 1
        else:
            lat.append(p.latency_ms)
    wall = time.perf_counter() - t_wall0

    lat_sorted = sorted(lat)
    q = lambda f: lat_sorted[min(len(lat_sorted) - 1, int(round(f * (len(lat_sorted) - 1))))]
    out = {
        "kind": "remote API end-to-end, sequential (concurrency=1), validation rows",
        "n": len(lat),
        "errors": errors,
        "p50_ms": q(0.5),
        "p95_ms": q(0.95),
        "mean_ms": statistics.fmean(lat),
        "min_ms": lat_sorted[0],
        "max_ms": lat_sorted[-1],
        "wall_s": wall,
        "model": backend.load().revision,
        "client_location": "Perth, Australia (cf-ray suffix PER observed)",
    }
    Path("results/jev_sequential_latency.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
