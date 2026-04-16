mod engine;
mod vector;

use std::{
    env, iter::Zip, time::{SystemTime, UNIX_EPOCH}
};

use rand::{
    rng,
    rngs::{self},
};
use rand_distr::{Distribution, StandardNormal};

use engine::bruteforce::BruteForceEngine;
use engine::engine::{CandidateScore, Distance, VSEngine};

use crate::{
    engine::asymmetric::{AsymmetricConfig, AsymmetricHashingEngine},
    vector::Vector,
};

fn uniformly_random_vector(dim: i32, mut rng: rngs::ThreadRng) -> Vector {
    let mut vec: Vector = (0..dim).map(|_| StandardNormal.sample(&mut rng)).collect();
    let magnitude: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();

    if magnitude > 0.0 {
        for x in vec.iter_mut() {
            *x /= magnitude;
        }
    }

    return vec;
}

fn uniformly_random_index(dim: i32, num_vectors: usize) -> Vec<Vector> {
    let rng: rngs::ThreadRng = rng();
    let mut index: Vec<Vector> = Vec::with_capacity(num_vectors);

    for _ in 0..num_vectors {
        index.push(uniformly_random_vector(dim, rng.clone()));
    }

    return index;
}

fn time_now() -> std::time::Duration {
    return SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
}

fn main_asymmetric() {
    let vector_dims: i32 = 64;
    let references = uniformly_random_index(vector_dims, 1_000_00);
    let query = uniformly_random_vector(vector_dims, rng());

    // let engine: BruteForceEngine = BruteForceEngine::new(&references, Distance::Cosine);

    let assymetric_config = AsymmetricConfig {
        distance: Distance::Cosine,
        num_centroids: 255,
        centroid_computer: engine::engine::CentroidComputerType::KMeans,
        vector_size: vector_dims,
        subvector_size: 4i32,
    };
    let mut engine: AsymmetricHashingEngine =
        AsymmetricHashingEngine::new(&references, assymetric_config);

    println!("Building Index...");
    let index_build_start_ts = time_now();
    engine.build();
    let index_build_end_ts = time_now();
    println!(
        "Index building time taken: {:?}",
        index_build_end_ts - index_build_start_ts
    );

    // println!("{:?}", references);
    println!("Searching...");
    let search_start = time_now();

    for i in 0..10000 {
        let query = uniformly_random_vector(vector_dims, rng());
        let results = engine.search(&query, 5);
    }

    let search_end = time_now();
    // println!("{:?}", results);
    println!("Search time taken: {:?}", search_end - search_start);
    // println!(
    //     "Top K scores: {:?}",
    //     results.iter().map(|x| x.score).collect::<Vec<f32>>()
    // );
    // println!("Best Score {:?}", results[0].score);
}

fn main_bruteforce() {
    let vector_dims: i32 = 64;
    let references = uniformly_random_index(vector_dims, 1_000_00);
    let query = uniformly_random_vector(vector_dims, rng());

    let mut engine: BruteForceEngine = BruteForceEngine::new(&references, Distance::Cosine);

    println!("Building Index...");
    let index_build_start_ts = time_now();
    engine.build();
    let index_build_end_ts = time_now();
    println!(
        "Index building time taken: {:?}",
        index_build_end_ts - index_build_start_ts
    );

    // println!("{:?}", references);
    println!("Searching...");
    let search_start = time_now();

    for _ in 0..10000 {
        let query = uniformly_random_vector(vector_dims, rng());
        engine.search(&query, 5);
    }

    let search_end = time_now();
    // println!("{:?}", results);
    println!("Search time taken: {:?}", search_end - search_start);
    // println!(
    //     "Top K scores: {:?}",
    //     results.iter().map(|x| x.score).collect::<Vec<f32>>()
    // );
    // println!("Best Score {:?}", results[0].score);
}

fn quality_eval() {
    let vector_dims: i32 = 64;
    let top_k = 5;
    let references = uniformly_random_index(vector_dims, 1_000_00);
    let query = uniformly_random_vector(vector_dims, rng());

    // let engine: BruteForceEngine = BruteForceEngine::new(&references, Distance::Cosine);

    let assymetric_config = AsymmetricConfig {
        distance: Distance::DotProductRaw,
        num_centroids: 255,
        centroid_computer: engine::engine::CentroidComputerType::KMeans,
        vector_size: vector_dims,
        subvector_size: 4,
    };
    let mut engine: AsymmetricHashingEngine =
        AsymmetricHashingEngine::new(&references, assymetric_config);
    let mut brute_force_engine: BruteForceEngine =
        BruteForceEngine::new(&references, Distance::DotProductRaw);

    println!("Building asymmetric index...");
    let index_build_start_ts = time_now();
    engine.build();
    let index_build_end_ts = time_now();
    println!(
        "Index building time taken: {:?}",
        index_build_end_ts - index_build_start_ts
    );
    println!("Building bruteforce index...");
    brute_force_engine.build();
    println!(
        "Index building time taken: {:?}",
        index_build_end_ts - index_build_start_ts
    );

    println!("Searching...");
    let search_start = time_now();

    for _ in 0..100 {
        let query = uniformly_random_vector(vector_dims, rng());
        let ah_results = engine.search(&query, top_k);
        let bf_results = brute_force_engine.search(&query, top_k);

        for (top_i, (a, b)) in ah_results.into_iter().zip(bf_results).enumerate() {
            println!("#{} AH result: {:?} \n#{} BF result: {:?}\n", top_i, a, top_i, b);
        }
    }

    let search_end = time_now();
    // println!("{:?}", results);
    println!("Search time taken: {:?}", search_end - search_start);
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.contains(&String::from("-a")) {
        main_asymmetric();
    } else if args.contains(&String::from("-b")) {
        main_bruteforce();
    } else if args.contains(&String::from("-ab")) {
        quality_eval();
    }
}
