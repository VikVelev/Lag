use rand::{
    rng,
    rngs::{self},
};
use rand_distr::{Distribution, StandardNormal};

fn uniformly_random_vector(dim: i32, mut rng: rngs::ThreadRng) -> Vec<f64> {
    let mut vec: Vec<f64> = (0..dim).map(|_| StandardNormal.sample(&mut rng)).collect();
    let magnitude: f64 = vec.iter().map(|x| x * x).sum::<f64>().sqrt();

    if magnitude > 0.0 {
        for x in vec.iter_mut() {
            *x /= magnitude;
        }
    }

    return vec;
}

fn uniformly_random_index(dim: i32, num_vectors: usize) -> Vec<Vec<f64>> {
    let rng: rngs::ThreadRng = rng();
    let mut index: Vec<Vec<f64>> = Vec::with_capacity(num_vectors);

    for _ in 0..num_vectors {
        index.push(uniformly_random_vector(dim, rng.clone()));
    }

    return index;
}

fn main() {
    let index = uniformly_random_index(8, 5);
    let query = uniformly_random_vector(8, rng());

    
    println!("Hello, world!");
    println!("{:?}", index)
}
