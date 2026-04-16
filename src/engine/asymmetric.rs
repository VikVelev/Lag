use std::collections::HashMap;

use kmeans::{EuclideanDistance, KMeans, KMeansConfig};

use crate::{
    engine::engine::{CandidateScore, CentroidComputerType, Distance, VSEngine},
    vector::{HashedVector, Vector},
};

pub struct AsymmetricConfig {
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

pub struct AsymmetricHashingEngine<'a> {
    references: &'a Vec<Vector>,
    config: AsymmetricConfig,
    // for every sub-vector, all learned centroids
    codebooks: Vec<Vec<Vector>>,
    hashed_references: Vec<HashedVector>,
    // for every sub-vector, distances to all centroids
    lut: Vec<Vec<f32>>,
}

impl<'a> AsymmetricHashingEngine<'a> {
    pub fn new(
        references: &'a Vec<Vector>,
        config: AsymmetricConfig,
    ) -> AsymmetricHashingEngine<'a> {
        let codebooks = Vec::<Vec<Vector>>::new();
        let lut = Vec::<Vec<f32>>::new();
        let hashed_references = Vec::<HashedVector>::new();

        return AsymmetricHashingEngine {
            references,
            config,
            codebooks,
            lut,
            hashed_references,
        };
    }
}

impl<'a> VSEngine for AsymmetricHashingEngine<'a> {
    fn build(&mut self) {
        println!("Starting build process...");

        // A storage for each # subvector - an array of subvectors
        let mut temp_subvector_split = Vec::<Vec<Vec<f32>>>::new();
        let num_subvectors = self.config.vector_size / self.config.subvector_size;

        for i in 0..num_subvectors {
            temp_subvector_split.push(Vec::<Vec<f32>>::new());
        }

        println!(
            "Splitting {} reference vectors into {} subvector groups...",
            self.references.len(),
            num_subvectors
        );

        // Split reference vectors into subvectors
        for (idx, vec) in self.references.iter().enumerate() {
            if idx % 100000 == 0 && idx > 0 {
                println!("  Processed {}/{} vectors", idx, self.references.len());
            }
            for i in 0..num_subvectors {
                let start = (i * self.config.subvector_size) as usize;
                let end = ((i + 1) * self.config.subvector_size) as usize;
                temp_subvector_split
                    .get_mut(i as usize)
                    .unwrap()
                    .push(vec[start..end].to_vec());
            }
        }

        println!("Entering KMeans training phase for each subvector group.");

        for (i, split) in temp_subvector_split.into_iter().enumerate() {
            println!("  Training codebook {}/{}...", i + 1, num_subvectors);

            // Run KMeans for each set of subvectors
            let kmeans: KMeans<f32, 8, _> = KMeans::new(
                &split.iter().flatten().copied().collect::<Vec<f32>>(),
                self.references.len(),
                self.config.subvector_size.try_into().unwrap(),
                EuclideanDistance,
            );

            let max_iter = 1;
            let result = kmeans.kmeans_lloyd(
                self.config.num_centroids.into(),
                max_iter,
                KMeans::init_kmeanplusplus,
                &KMeansConfig::default(),
            );

            let num_centroids = self.config.num_centroids as usize;

            let centroids: Vec<Vec<f32>> = (0..num_centroids)
                .map(|i| {
                    let centroid_slice = &result.centroids[i];
                    centroid_slice.to_vec()
                })
                .collect();

            // Create a codebook for each
            self.codebooks.push(centroids);

            // Get the current split -> find the closest centroid and compress it into a hash
            // todo!()
        }

        println!(
            "Successfully built {} codebooks. Proceeding to compression...",
            self.codebooks.len()
        );
    }

    // TODO: Implement
    fn amend(&self, vec: Vector) {
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
