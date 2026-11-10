// Benchmarks the engines on dbpedia-openai-1M-1536-angular and measures recall
// against the exact neighbours computed by prepare_dbpedia.py.
//
// cargo run --release -- eval --engines bf,ah8 --queries 1000
//
// Engines:
//   bf  - brute force
//   ahN - asymmetric hashing with N-dimensional subvectors (1536 / N bytes per vector)
//   add c for mean centering, e.g. ah8c
//
// Other flags: --train-sample 100000, --kmeans-iters 25, --data datasets/dbpedia-openai-1M

use std::{
    collections::HashSet,
    path::PathBuf,
    time::{Duration, Instant},
};

use crate::{
    engine::{
        asymmetric::{AsymmetricConfig, AsymmetricHashingEngine},
        bruteforce::BruteForceEngine,
        engine::{CandidateScore, CentroidComputerType, Distance, VSEngine},
    },
    eval::dataset::{read_ids, read_vectors},
    vector::Vector,
};

// every engine is asked for the top 100, recall is reported for these cut-offs
const RECALL_AT: [usize; 3] = [1, 10, 100];

struct EvalOptions {
    data_dir: PathBuf,
    engines: Vec<String>,
    num_queries: usize,
    training_sample: usize,
    kmeans_iterations: usize,
}

struct EvalResult {
    name: String,
    memory_bytes: usize,
    build_time: Duration,
    search_time: Duration,
    num_queries: usize,
    recalls: Vec<f64>,
}

fn parse_options(args: &[String]) -> EvalOptions {
    let mut options = EvalOptions {
        data_dir: PathBuf::from("datasets/dbpedia-openai-1M"),
        engines: vec![String::from("bf"), String::from("ah8")],
        num_queries: 1000,
        training_sample: 100_000,
        kmeans_iterations: 25,
    };

    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        let value = args
            .get(i + 1)
            .unwrap_or_else(|| panic!("{} needs a value", flag));

        match flag {
            "--data" => options.data_dir = PathBuf::from(value),
            "--engines" => options.engines = value.split(',').map(String::from).collect(),
            "--queries" => options.num_queries = value.parse().expect("--queries must be a number"),
            "--train-sample" => {
                options.training_sample = value.parse().expect("--train-sample must be a number")
            }
            "--kmeans-iters" => {
                options.kmeans_iterations = value.parse().expect("--kmeans-iters must be a number")
            }
            _ => panic!("Unknown eval flag: {}", flag),
        }
        i += 2;
    }

    return options;
}

// What fraction of the true top k ended up in the engine's top k, averaged over all queries
fn recall_at(results: &[Vec<CandidateScore>], ground_truth: &[Vec<u32>], k: usize) -> f64 {
    let mut total = 0.0;

    for (result, truth) in results.iter().zip(ground_truth.iter()) {
        let true_neighbours: HashSet<usize> = truth[..k].iter().map(|&id| id as usize).collect();
        let found = result
            .iter()
            .take(k)
            .filter(|candidate| true_neighbours.contains(&candidate.index))
            .count();
        total += found as f64 / k as f64;
    }

    return total / results.len() as f64;
}

fn evaluate(
    name: &str,
    engine: &mut dyn VSEngine,
    queries: &[Vector],
    ground_truth: &[Vec<u32>],
) -> EvalResult {
    println!("\n=== {} ===", name);

    let build_start = Instant::now();
    engine.build();
    let build_time = build_start.elapsed();
    println!("Index building time taken: {:?}", build_time);

    let top_k = RECALL_AT[RECALL_AT.len() - 1];
    println!("Searching {} queries...", queries.len());
    let search_start = Instant::now();
    let results = engine.search_batch(queries, top_k);
    let search_time = search_start.elapsed();

    let recalls: Vec<f64> = RECALL_AT
        .iter()
        .map(|&k| recall_at(&results, ground_truth, k))
        .collect();

    println!(
        "Search time taken: {:?} ({:.1} QPS)",
        search_time,
        queries.len() as f64 / search_time.as_secs_f64()
    );
    println!(
        "Recall@1: {:.4}, Recall@10: {:.4}, Recall@100: {:.4}",
        recalls[0], recalls[1], recalls[2]
    );

    return EvalResult {
        name: String::from(name),
        memory_bytes: engine.index_memory_bytes(),
        build_time,
        search_time,
        num_queries: queries.len(),
        recalls,
    };
}

// Prints the results as a markdown table, ready to paste into the README
fn print_table(results: &[EvalResult]) {
    println!();
    println!(
        "| Engine | Index Memory Usage | Build Time | Time for {} Lookups | QPS | Recall@1 | Recall@10 | Recall@100 |",
        results[0].num_queries
    );
    println!("| ------ | ------------------ | ---------- | ------------------- | --- | -------- | --------- | ---------- |");

    for result in results {
        println!(
            "| {} | {:.2} MB | {:.1}s | {:.3}s | {:.1} | {:.4} | {:.4} | {:.4} |",
            result.name,
            result.memory_bytes as f64 / 1_048_576.0,
            result.build_time.as_secs_f64(),
            result.search_time.as_secs_f64(),
            result.num_queries as f64 / result.search_time.as_secs_f64(),
            result.recalls[0],
            result.recalls[1],
            result.recalls[2]
        );
    }
}

pub fn run(args: &[String]) {
    let options = parse_options(args);

    println!("Loading dataset from {}...", options.data_dir.display());
    let load_start = Instant::now();
    let references = read_vectors(&options.data_dir.join("base.fbin"), None);
    let queries = read_vectors(&options.data_dir.join("query.fbin"), Some(options.num_queries));
    let ground_truth = read_ids(&options.data_dir.join("gt.ibin"), Some(options.num_queries));
    let vector_dims = references[0].len() as i32;
    println!(
        "Loaded {} references and {} queries ({} dims) in {:?}",
        references.len(),
        queries.len(),
        vector_dims,
        load_start.elapsed()
    );

    // All vectors are unit length, so the raw dot product is the same as cosine
    let mut results = Vec::new();
    for engine_name in options.engines.iter() {
        if engine_name == "bf" {
            let mut engine = BruteForceEngine::new(&references, Distance::DotProductRaw);
            results.push(evaluate("Brute Force", &mut engine, &queries, &ground_truth));
        } else if let Some(ah_options) = engine_name.strip_prefix("ah") {
            let mean_center = ah_options.ends_with('c');
            let subvector_size: i32 = ah_options
                .trim_end_matches('c')
                .parse()
                .expect("Asymmetric hashing engines look like ah8 or ah8c");
            assert!(
                vector_dims % subvector_size == 0,
                "subvector size has to divide the vector size"
            );

            let assymetric_config = AsymmetricConfig {
                distance: Distance::DotProductRaw,
                num_centroids: 255,
                centroid_computer: CentroidComputerType::KMeans,
                vector_size: vector_dims,
                subvector_size,
                training_sample: options.training_sample,
                kmeans_iterations: options.kmeans_iterations,
                mean_center,
            };
            let mut engine = AsymmetricHashingEngine::new(&references, assymetric_config);
            let mut name = format!(
                "Asymmetric Hashing ({}-d, {} B/vector",
                subvector_size,
                vector_dims / subvector_size
            );
            if mean_center {
                name.push_str(", mean-centered");
            }
            name.push(')');
            results.push(evaluate(&name, &mut engine, &queries, &ground_truth));
        } else {
            panic!("Unknown engine {} - use bf, ahN or ahNc", engine_name);
        }
    }

    print_table(&results);
}
