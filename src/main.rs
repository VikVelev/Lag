mod engine;
mod vector;

use rand::{
    rng,
    rngs::{self},
};
use rand_distr::{Distribution, StandardNormal};

use engine::bruteforce::BruteForceEngine;
use engine::engine::{CandidateScore, Distance, VSEngine};

use crate::vector::Vector;

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

fn main() {
    let references = uniformly_random_index(32, 10000);
    let query = uniformly_random_vector(32, rng());

    let engine: BruteForceEngine = BruteForceEngine::new(&references, Distance::Cosine);
    engine.build();

    let results = engine.search(&query, 5);

    println!("{:?}", references);
    println!("Seaching");
    println!("{:?}", results);
    println!(
        "Top K scores: {:?}",
        results.iter().map(|x| x.score).collect::<Vec<f32>>()
    );
    println!("Best Score {:?}", results[0].score);
}
