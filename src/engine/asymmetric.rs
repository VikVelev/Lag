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
        };
    }

    fn build_codebook(&mut self) {
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

            self.codebooks.push(centroids);
        }

        println!(
            "Successfully built {} codebooks. Proceeding to compression...",
            self.codebooks.len()
        );
    }

    fn find_codebook_index(&mut self, vec: Vector, subspace: usize) -> usize {
        let mut idx: usize = std::usize::MAX;
        let mut distance = std::f32::MAX;

        for (eid, centroid) in self.codebooks[subspace].clone().into_iter().enumerate() {
            let current_distance = self.config.distance.compute(&vec, &centroid);
            if current_distance < distance {
                distance = current_distance;
                idx = eid;
            }
        }

        return idx;
    }

    fn create_lut(&self, query: &Vector) -> Vec<Vec<f32>> {
        let mut lut = Vec::<Vec<f32>>::new();
        let num_subvectors = self.config.vector_size / self.config.subvector_size;

        for i in 0..num_subvectors {
            let start = (i * self.config.subvector_size) as usize;
            let end = ((i + 1) * self.config.subvector_size) as usize;
            let subvec = query[start..end].to_vec();

            let mut distances = vec![0f32; self.config.num_centroids.into()];
            for (cidx, centroid) in self.codebooks[i as usize].iter().enumerate() {
                distances[cidx] = self.config.distance.compute(&subvec, centroid);
            }
            lut.push(distances);
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

        self.build_codebook();

        let num_subvectors = self.config.vector_size / self.config.subvector_size;

        println!("Hashing Vectors...");
        for (idx, vec) in self.references.iter().enumerate() {
            if idx % 100000 == 0 && idx > 0 {
                println!("  Processed {}/{} vectors", idx, self.references.len());
            }

            let mut hashed_vector = vec![0; num_subvectors as usize];

            for i in 0..num_subvectors {
                let start = (i * self.config.subvector_size) as usize;
                let end = ((i + 1) * self.config.subvector_size) as usize;
                let closest_centroid_idx =
                    self.find_codebook_index(vec[start..end].to_vec(), i.try_into().unwrap());
                hashed_vector[i as usize] = closest_centroid_idx;
            }
            self.hashed_references.push(hashed_vector);
        }

        let hashed_bytes = (self.hashed_references.capacity() * std::mem::size_of::<Vec<usize>>())
            + self
                .hashed_references
                .iter()
                .map(|v| v.capacity() * std::mem::size_of::<usize>())
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

    // TODO: Implement
    fn amend(&self, vec: Vector) {
        panic!();
    }

    // Assymetric Search
    // Compute a Look-up table
    // Use it to go over the compressed references
    // Top_k is actually O(n + klogk) with select_nth_unstable
    fn search(&self, query: &Vector, top_k: usize) -> Vec<CandidateScore> {
        if top_k == 0 || self.hashed_references.is_empty() {
            return vec![];
        }
        let k = std::cmp::min(top_k, self.hashed_references.len());

        let lut = self.create_lut(query);

        let mut scores: Vec<(usize, f32)> = self
            .hashed_references
            .iter()
            .enumerate()
            .map(|(idx, vec)| {
                let mut current_distance = 0f32;
                for (subspace, centroid_idx) in vec.iter().enumerate() {
                    current_distance += lut[subspace][*centroid_idx];
                }
                (idx, current_distance)
            })
            .collect();

        let (top_k_slice, _, _) = scores.select_nth_unstable_by(k - 1, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        top_k_slice.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        top_k_slice
            .iter()
            .map(|&(idx, score)| CandidateScore {
                index: idx,
                score,
            })
            .collect()
    }
}
