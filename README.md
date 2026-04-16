# Lag

Low-Performance Vector Search DB

Zero AI - Maximum Understanding

## Multi-core Benchmark Results

_1,000,000 reference vectors | 64 dimensions | 10,000 Batch Queries_

Asymmetric Hashing uses k-means for centroid computation, 255 centroids, 4-dimensional subspaces.

| Engine             | Memory Usage | Time for 10,000 Lookups |
| ------------------ | ------------ | ----------------------- |
| Brute Force        | 267.03 MB    | 111.774s                |
| Asymmetric Hashing | 39.26 MB     | 29.019s                 |
| HNSW               | TODO         | TODO                    |

At some points AH was about 10x faster than brute force, however I suspect after parallelizing the brute force search, it proved to benefit much more from cache coherency and ended up being only 4x faster - there is probably much more room for improvement in AH implementation

Quality is very comparable, but not fully evaluated.
