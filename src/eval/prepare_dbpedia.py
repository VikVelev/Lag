# Converts the dbpedia-openai-1M parquet shards into flat binary files the Rust eval can read,
# and computes the exact nearest neighbours to measure recall against.
#
# Writes into the dataset folder (all files: u32 rows, u32 cols, then the data row by row):
#   base.fbin  - f32 reference vectors, everything except the held-out queries
#   query.fbin - f32 query vectors, randomly held out from the dataset
#   gt.ibin    - u32 ids into base.fbin of the exact top-k for every query, best first
#
# Usage: python prepare_dbpedia.py <dataset_dir> [--queries 10000] [--k 100] [--seed 42]

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

DIM = 1536


def write_header(f, rows, cols):
    f.write(np.array([rows, cols], dtype="<u4").tobytes())


def load_vectors(shards):
    total = sum(pq.ParquetFile(shard).metadata.num_rows for shard in shards)
    vectors = np.empty((total, DIM), dtype=np.float32)

    row = 0
    for shard in shards:
        # the embeddings are stored as a list<double> column called "openai"
        column = pq.read_table(shard, columns=["openai"]).column("openai")
        block = pc.list_flatten(column).to_numpy().reshape(-1, DIM)
        vectors[row : row + len(block)] = block
        row += len(block)
        print(f"  loaded {shard.name}: {row}/{total}", flush=True)

    return vectors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--queries", type=int, default=10_000)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    shards = sorted((args.dataset_dir / "data").glob("*.parquet"))
    if not shards:
        sys.exit(f"no parquet shards in {args.dataset_dir / 'data'} - download the dataset first")

    print(f"Loading {len(shards)} shards...", flush=True)
    vectors = load_vectors(shards)
    n = len(vectors)

    # ada-002 embeddings should already be unit length, normalize anyway so dot product == cosine
    norms = np.linalg.norm(vectors, axis=1)
    print(f"Norms before normalizing: min={norms.min():.4f} max={norms.max():.4f}")
    vectors /= norms[:, None]

    # Hold out random vectors as queries, so a query never finds itself in the index
    rng = np.random.default_rng(args.seed)
    query_rows = np.sort(rng.choice(n, size=args.queries, replace=False))
    is_query = np.zeros(n, dtype=bool)
    is_query[query_rows] = True
    queries = vectors[query_rows]
    # where each original row ends up in base.fbin once the queries are taken out
    base_id = np.cumsum(~is_query) - 1
    num_base = n - args.queries

    out = args.dataset_dir
    print(f"Writing base.fbin ({num_base} x {DIM}) and query.fbin ({args.queries} x {DIM})...")
    with open(out / "base.fbin", "wb") as f:
        write_header(f, num_base, DIM)
        # write in chunks so we never hold a second copy of the whole dataset
        for start in range(0, n, 50_000):
            chunk = slice(start, min(start + 50_000, n))
            f.write(vectors[chunk][~is_query[chunk]].tobytes())
    with open(out / "query.fbin", "wb") as f:
        write_header(f, args.queries, DIM)
        f.write(queries.tobytes())

    # Exact ground truth: score every query against every base vector
    print(f"Computing exact top-{args.k} ground truth...", flush=True)
    started = time.time()
    gt = np.empty((args.queries, args.k), dtype=np.uint32)
    batch = 256
    for start in range(0, args.queries, batch):
        scores = queries[start : start + batch] @ vectors.T
        scores[:, is_query] = -np.inf
        top = np.argpartition(-scores, args.k, axis=1)[:, : args.k]
        top_scores = np.take_along_axis(scores, top, axis=1)
        best_first = np.take_along_axis(top, np.argsort(-top_scores, axis=1), axis=1)
        gt[start : start + len(best_first)] = base_id[best_first]
        print(f"  {start + len(best_first)}/{args.queries} queries ({time.time() - started:.0f}s)", flush=True)

    with open(out / "gt.ibin", "wb") as f:
        write_header(f, args.queries, args.k)
        f.write(gt.astype("<u4").tobytes())

    print("Done.")


if __name__ == "__main__":
    main()
