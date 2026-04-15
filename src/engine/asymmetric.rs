use std::collections::HashMap;

use crate::{
    engine::engine::{CandidateScore, CentroidComputerType, Distance, VSEngine},
    vector::{HashedVector, Vector},
};

pub struct AssymetricConfig {
    // type of distance
    pub distance: Distance,

    // number of centroids
    pub num_centroids: u8,
    // how do we compute centroids in the codebook
    pub centroid_computer: CentroidComputerType,

    // size of db vectors
    pub vector_size: i32,
    // size of chunked vectors - multiple of vector_size
    pub subvector_size: i32,
}

pub struct AssymetricHashingEngine<'a> {
    references: &'a Vec<Vector>,
    config: AssymetricConfig,
    // for every sub-vector, all learned centroids
    codebooks: Vec<HashMap<u8, Vector>>,
    hashed_references: Vec<HashedVector>,
    // for every sub-vector, distances to all centroids
    lut: Vec<Vec<f32>>,
}

impl<'a> AssymetricHashingEngine<'a> {
    pub fn new(
        references: &'a Vec<Vector>,
        config: AssymetricConfig,
    ) -> AssymetricHashingEngine<'a> {
        let codebooks = Vec::<HashMap<u8, Vector>>::new();
        let lut = Vec::<Vec<f32>>::new();
        let hashed_references = Vec::<HashedVector>::new();

        return AssymetricHashingEngine {
            references,
            config,
            codebooks,
            lut,
            hashed_references,
        };
    }
}

impl<'a> VSEngine for AssymetricHashingEngine<'a> {
    fn build(&self) {
        // Split database into subvectors
        // Create a codebook for each
    }

    fn amend(&self, vec: Vector) {
        // TODO: Implement
        panic!();
    }

    // Assymetric Search
    // Compute a Look-up table
    // Use it to go over the compressed references
    fn search(&self, query: &Vector, top_k: usize) -> Vec<CandidateScore> {
        let mut scores: Vec<CandidateScore> = self
            .references
            .iter()
            .map(|reference| CandidateScore {
                candidate: reference.to_vec(),
                score: self.config.distance.compute(query, reference),
            })
            .collect();

        scores.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        return scores.into_iter().take(top_k).collect();
    }
}
