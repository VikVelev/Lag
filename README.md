# Lag

Low-Performance Vector Search DB

Zero AI - Maximum Understanding

## Multi-core Benchmark Results

_1,000,000 reference vectors | 64 dimensions | 10,000 Batch Queries_

Asymmetric Hashing uses k-means for centroid computation, 255 centroids, 4-dimensional subspaces.

| Engine             | Index Memory Usage | Time for 10,000 Lookups |
| ------------------ | ------------------ | ----------------------- |
| Brute Force        | 267.03 MB          | 111.774s                |
| Asymmetric Hashing | 39.26 MB           | 29.019s                 |
| HNSW               | TODO               | TODO                    |

At some points AH was about 10x faster than brute force, however I suspect after parallelizing the brute force search, it proved to benefit much more from cache coherency and ended up being only 4x faster - there is probably much more room for improvement in AH implementation.

Quality is very comparable, but not fully evaluated - see the dbpedia benchmark below for actual recall numbers.

Hardware: Ryzen 7 7700X, 16 threads, 32 GB RAM

## dbpedia-openai-1M-1536-angular Benchmark

_990,000 reference vectors | 1536 dimensions | 1,000 Batch Queries | top 100 per query_

Real embeddings this time - [DBpedia entities embedded with OpenAI's `text-embedding-ada-002`](https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M). The vectors are unit length, so dot product is the same as cosine. 10,000 random vectors are held out as queries (only the first 1,000 are used here), and the ground truth is the exact top 100 for each query. Recall@k is how many of the true top k neighbours an engine found in its own top k.

Asymmetric Hashing uses 255 centroids per subspace, with k-means trained for 25 iterations on an evenly spaced sample of 100,000 vectors.

| Engine                                 | Index Memory Usage | Build Time | Time for 1,000 Lookups | QPS  | Recall@1 | Recall@10 | Recall@100 |
| -------------------------------------- | ------------------ | ---------- | ---------------------- | ---- | -------- | --------- | ---------- |
| Brute Force                            | 5823.44 MB         | -          | 183.698s               | 5.4  | 1.0000   | 1.0000    | 1.0000     |
| Asymmetric Hashing (4-d, 384 B/vector) | 386.70 MB          | 105.9s     | 24.175s                | 41.4 | 0.8170   | 0.8419    | 0.8610     |
| Asymmetric Hashing (4-d, 384 B/vector, mean-centered) | 386.71 MB | 106.5s  | 24.378s                | 41.0 | 0.7990   | 0.8398    | 0.8607     |
| Asymmetric Hashing (8-d, 192 B/vector) | 205.43 MB          | 67.4s      | 12.093s                | 82.7 | 0.6880   | 0.7211    | 0.7364     |
| Asymmetric Hashing (8-d, 192 B/vector, mean-centered) | 205.43 MB | 67.9s   | 12.464s                | 80.2 | 0.6850   | 0.7224    | 0.7353     |
| Asymmetric Hashing (4-d, 384 B/vector, rerank top 1000) | 386.70 MB* | 105.0s | 24.879s               | 40.2 | 1.0000   | 1.0000    | 1.0000     |
| Asymmetric Hashing (8-d, 192 B/vector, rerank top 1000) | 205.43 MB* | 68.2s  | 13.829s               | 72.3 | 1.0000   | 1.0000    | 0.9986     |
| Asymmetric Hashing (8-d, 192 B/vector, rerank top 200)  | 205.43 MB* | 67.7s  | 12.801s               | 78.1 | 1.0000   | 0.9991    | 0.9242     |
| HNSW                                   | TODO               | TODO       | TODO                   | TODO | TODO     | TODO      | TODO       |

AH is 8-15x faster and 15-28x smaller, but recall takes a big hit. ada-002 embeddings are packed really tightly - the median gap between the 10th and 11th neighbour is ~0.001, while AH's scoring error is around 0.01, so neighbours near the cut-off get shuffled around.

Mean centering (hashing `x - mean` instead of `x`) was an attempt to fix that, since ada-002 vectors all point roughly the same way (a random pair has ~0.69 similarity). It didn't help at all - in hindsight k-means doesn't care about a constant shift, the centroids just move along with the data and the quantization error stays exactly the same.

What does work is reranking: use the hashed scores only to pick a shortlist of the best M candidates, then score just those M again with the full vectors. The true neighbours are almost always somewhere in the shortlist, so recall goes back to ~1.0, and scoring 1,000 extra vectors per query is nothing compared to scanning all the hashed ones.

\* Reranking needs the full vectors as well. Right now they're kept in RAM (5823.44 MB here, same as brute force) - the memory column only counts the hashed index. Since a query only ever looks at M of them they could live on disk instead, but that's not implemented.

To reproduce:

```bash
src/eval/download_eval_data.sh
cargo run --release -- eval --engines bf,ah4,ah4c,ah8,ah8c,ah4r1000,ah8r1000,ah8r200 --queries 1000
```
