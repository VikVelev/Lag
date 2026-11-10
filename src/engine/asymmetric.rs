use crate::{
    engine::engine::{CandidateScore, CentroidComputerType, Distance, VSEngine},
    vector::{HashedVector, Vector},
};
use rayon::prelude::*;

pub struct AsymmetricConfig {
    // type of distance
    pub distance: Distance,

    // number of centroids
    pub num_centroids: usize,
    // how do we compute centroids in the codebook
    pub centroid_computer: CentroidComputerType,

    // size of db vectors
    pub vector_size: i32,
    // size of chunked vectors - multiple of vector_size
    pub subvector_size: i32,

    // how many reference vectors to train k-means on, 0 means all of them
    pub training_sample: usize,
    // k-means iterations for each codebook
    pub kmeans_iterations: usize,

    // subtract the mean of all references before hashing - only for DotProductRaw,
    // since q·x = q·(x - mean) + q·mean and q·mean is the same for every reference
    pub mean_center: bool,

    // how many of the best hashed candidates to score again with the full vectors, 0 means no reranking
    pub rerank: usize,
}

// Keeps only the best k scores, sorted from best to worst
fn keep_top_k(scores: &mut Vec<(usize, f32)>, k: usize, is_similarity: bool) {
    let compare = |a: &(usize, f32), b: &(usize, f32)| {
        if is_similarity {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        } else {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        }
    };

    // select_nth leaves the k-th best at index k - 1, so the top k is everything up to and including it
    scores.select_nth_unstable_by(k - 1, compare);
    scores.truncate(k);
    scores.sort_by(compare);
}

pub struct AsymmetricHashingEngine<'a> {
    references: &'a Vec<Vector>,
    config: AsymmetricConfig,
    // for every sub-vector, all learned centroids
    codebooks: Vec<Vec<Vector>>,
    hashed_references: Vec<HashedVector>,
    // only set when mean_center is on
    mean: Option<Vector>,
}

impl<'a> AsymmetricHashingEngine<'a> {
    pub fn new(
        references: &'a Vec<Vector>,
        config: AsymmetricConfig,
    ) -> AsymmetricHashingEngine<'a> {
        let codebooks = Vec::<Vec<Vector>>::new();
        let hashed_references = Vec::<HashedVector>::new();

        return AsymmetricHashingEngine {
            references,
            config,
            codebooks,
            hashed_references,
            mean: None,
        };
    }

    fn compute_mean(&self) -> Vector {
        // sum in f64, a million f32 additions lose too much precision
        let mut sum = vec![0f64; self.config.vector_size as usize];
        for vec in self.references.iter() {
            for (total, x) in sum.iter_mut().zip(vec.iter()) {
                *total += *x as f64;
            }
        }

        let num_references = self.references.len() as f64;
        return sum.iter().map(|total| (total / num_references) as f32).collect();
    }

    // The vector that actually gets hashed - centered if mean_center is on
    fn centered(&self, vec: &[f32]) -> Vector {
        match &self.mean {
            Some(mean) => vec.iter().zip(mean.iter()).map(|(x, m)| x - m).collect(),
            None => vec.to_vec(),
        }
    }

    fn build_codebook(&mut self) {
        // Other codebook centroid computers are not implemented yet
        if !matches!(self.config.centroid_computer, CentroidComputerType::KMeans) {
            panic!();
        }

        let num_subvectors = (self.config.vector_size / self.config.subvector_size) as usize;
        let subvector_size = self.config.subvector_size as usize;

        // Training k-means on every vector gets way too slow (and memory hungry) for big indexes,
        // so train on an evenly spaced sample instead
        let num_references = self.references.len();
        let mut sample_size = num_references;
        if self.config.training_sample > 0 && self.config.training_sample < num_references {
            sample_size = self.config.training_sample;
        }
        let sample_rows: Vec<usize> = (0..sample_size)
            .map(|i| i * num_references / sample_size)
            .collect();

        println!(
            "Training {} codebooks on {}/{} reference vectors...",
            num_subvectors, sample_size, num_references
        );

        for i in 0..num_subvectors {
            println!("  Training codebook {}/{}...", i + 1, num_subvectors);

            let start = i * subvector_size;
            let end = (i + 1) * subvector_size;

            let mut flat_data: Vec<f32> = Vec::with_capacity(sample_size * subvector_size);
            for &row in sample_rows.iter() {
                let subvector = &self.references[row][start..end];
                match &self.mean {
                    Some(mean) => {
                        for (x, m) in subvector.iter().zip(mean[start..end].iter()) {
                            flat_data.push(x - m);
                        }
                    }
                    None => flat_data.extend_from_slice(subvector),
                }
            }

            let data = ndarray::Array2::from_shape_vec((sample_size, subvector_size), flat_data)
                .expect("Failed to create ndarray from subvectors");

            // Always train with euclidean because if we normalize subvectors (needed for spherical)
            // we would lose the ability to compute dot product and the relative strength to the overall vector
            let mut kmeans = kentro::KMeans::new(self.config.num_centroids)
                .with_iterations(self.config.kmeans_iterations)
                .with_euclidean(true);

            kmeans
                .train(data.view(), None)
                .expect("KMeans training failed");

            let centroids: Vec<Vec<f32>> = kmeans
                .centroids()
                .expect("Centroids not found after training")
                .outer_iter()
                .map(|row| row.to_vec())
                .collect();

            self.codebooks.push(centroids);
        }

        println!(
            "Successfully built {} codebooks. Proceeding to compression...",
            self.codebooks.len()
        );
    }

    fn find_codebook_index(&self, vec: &[f32], subspace: usize) -> usize {
        let mut idx: usize = std::usize::MAX;
        let mut best_score = std::f32::MAX;

        for (eid, centroid) in self.codebooks[subspace].iter().enumerate() {
            // FORCE squared Euclidean distance for vector quantization
            // This guarantees we pick the centroid that accurately matches
            // both the direction AND the magnitude of the subvector.
            let mut current_score = 0f32;
            for (v_val, c_val) in vec.iter().zip(centroid.iter()) {
                let diff = v_val - c_val;
                current_score += diff * diff;
            }

            if current_score < best_score {
                best_score = current_score;
                idx = eid;
            }
        }

        return idx;
    }

    fn create_lut(&self, query: &Vector) -> Vec<f32> {
        let num_subvectors = (self.config.vector_size / self.config.subvector_size) as usize;
        // Even though the lut is a matrix, keep it flattened so memory slices fit nicely within cachelines
        let mut lut = Vec::with_capacity(num_subvectors * self.config.num_centroids);

        for i in 0..num_subvectors {
            let start = i * self.config.subvector_size as usize;
            let end = (i + 1) * self.config.subvector_size as usize;
            let subvec = &query[start..end];

            for centroid in self.codebooks[i].iter() {
                lut.push(self.config.distance.compute(subvec, centroid));
            }
        }

        return lut;
    }
}

impl<'a> VSEngine for AsymmetricHashingEngine<'a> {
    fn build(&mut self) {
        println!("Starting build process...");

        let ref_bytes = (self.references.capacity() * std::mem::size_of::<Vec<f32>>())
            + self
                .references
                .iter()
                .map(|v| v.capacity() * std::mem::size_of::<f32>())
                .sum::<usize>();

        println!(
            "Original references size: ~{:.2} MB ({} vectors)",
            ref_bytes as f64 / 1_048_576.0,
            self.references.len()
        );

        if self.config.mean_center {
            if !matches!(self.config.distance, Distance::DotProductRaw) {
                panic!("Mean centering only keeps the ranking the same for DotProductRaw");
            }
            println!("Computing the mean of the references...");
            self.mean = Some(self.compute_mean());
        }

        self.build_codebook();

        let num_subvectors = (self.config.vector_size / self.config.subvector_size) as usize;
        let subvector_size = self.config.subvector_size as usize;

        println!("Hashing Vectors...");
        // Every vector is hashed on its own, so this is trivially parallel
        self.hashed_references = self
            .references
            .par_iter()
            .map(|vec| {
                let vec = self.centered(vec);
                let mut hashed_vector = vec![0u8; num_subvectors];
                for i in 0..num_subvectors {
                    let start = i * subvector_size;
                    let end = (i + 1) * subvector_size;
                    hashed_vector[i] = self.find_codebook_index(&vec[start..end], i) as u8;
                }
                hashed_vector
            })
            .collect();

        let hashed_bytes = (self.hashed_references.capacity() * std::mem::size_of::<Vec<u8>>())
            + self
                .hashed_references
                .iter()
                .map(|v| v.capacity() * std::mem::size_of::<u8>())
                .sum::<usize>();

        println!(
            "Full references size: ~{:.2} MB",
            ref_bytes as f64 / 1_048_576.0
        );

        println!(
            "Hashed references size: ~{:.2} MB",
            hashed_bytes as f64 / 1_048_576.0
        );
        println!(
            "Memory reduction: {:.2}x smaller",
            ref_bytes as f64 / hashed_bytes as f64
        );
    }

    fn index_memory_bytes(&self) -> usize {
        let hashed_bytes = (self.hashed_references.capacity() * std::mem::size_of::<Vec<u8>>())
            + self
                .hashed_references
                .iter()
                .map(|v| v.capacity() * std::mem::size_of::<u8>())
                .sum::<usize>();
        // the codebooks are needed at query time too, for the look-up table
        let codebook_bytes = self
            .codebooks
            .iter()
            .flatten()
            .map(|centroid| centroid.capacity() * std::mem::size_of::<f32>())
            .sum::<usize>();

        let mean_bytes = match &self.mean {
            Some(mean) => mean.capacity() * std::mem::size_of::<f32>(),
            None => 0,
        };

        return hashed_bytes + codebook_bytes + mean_bytes;
    }

    // TODO: Implement
    fn amend(&self, _vec: Vector) {
        panic!();
    }

    // Assymetric Search
    // Compute a Look-up table
    // Use it to go over the compressed references
    // Top_k is actually O(n + klogk) with select_nth_unstable
    // Optionally rerank the best candidates with the full vectors
    fn search(&self, query: &Vector, top_k: usize) -> Vec<CandidateScore> {
        if top_k == 0 || self.hashed_references.is_empty() {
            return vec![];
        }
        let k = std::cmp::min(top_k, self.hashed_references.len());

        let lut = self.create_lut(query);
        let num_centroids = self.config.num_centroids;
        // when centering, the lut gives q·(x - mean) - add q·mean back so the scores are still q·x
        let mut offset = 0f32;
        if let Some(mean) = &self.mean {
            offset = self.config.distance.compute(query, mean);
        }

        let mut scores: Vec<(usize, f32)> = self
            .hashed_references
            .par_iter()
            .enumerate()
            .map(|(idx, vec)| {
                let mut current_score = offset;
                for (subspace, centroid_idx) in vec.iter().enumerate() {
                    current_score += lut[subspace * num_centroids + (*centroid_idx as usize)];
                }
                (idx, current_score)
            })
            .collect();

        let is_similarity = match self.config.distance {
            Distance::DotProductRaw | Distance::Cosine => true,
            Distance::L2Norm | Distance::L1Norm => false,
        };

        if self.config.rerank == 0 {
            keep_top_k(&mut scores, k, is_similarity);
        } else {
            // Reranking
            // Use the hashed scores only to pick a shortlist of candidates
            // Score the shortlist again with the full vectors and keep the best top_k of those
            let shortlist_size = std::cmp::min(std::cmp::max(self.config.rerank, k), scores.len());
            keep_top_k(&mut scores, shortlist_size, is_similarity);

            for candidate in scores.iter_mut() {
                candidate.1 = self
                    .config
                    .distance
                    .compute(query, &self.references[candidate.0]);
            }

            keep_top_k(&mut scores, k, is_similarity);
        }

        scores
            .iter()
            .map(|&(idx, score)| CandidateScore { index: idx, score })
            .collect()
    }
}
