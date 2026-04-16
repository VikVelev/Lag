use crate::{
    engine::engine::{CandidateScore, Distance, VSEngine},
    vector::Vector,
};

pub struct BruteForceEngine<'a> {
    references: &'a Vec<Vector>,
    distance: Distance,
}

impl<'a> BruteForceEngine<'a> {
    pub fn new(references: &'a Vec<Vector>, distance: Distance) -> BruteForceEngine<'a> {
        BruteForceEngine {
            references,
            distance,
        }
    }
}

impl<'a> VSEngine for BruteForceEngine<'a> {
    fn build(&mut self) {
        // No-op for BruteForceEngine, as references are set at construction.
    }

    fn amend(&self, vec: Vector) {
        // TODO: Implement
        panic!();
    }

    // Almost the dummest algorithm possible:
    // Compute distance to each vector - store them in an array
    // Compute cut-off and get the top_k
    // Top_k is actually O(n + klogk) with select_nth_unstable
    fn search(&self, query: &Vector, top_k: usize) -> Vec<CandidateScore> {
        if top_k == 0 || self.references.is_empty() {
            return vec![];
        }
        let k = std::cmp::min(top_k, self.references.len());

        let mut scores: Vec<(usize, f32)> = self
            .references
            .iter()
            .enumerate()
            .map(|(idx, reference)| (idx, self.distance.compute(query, reference)))
            .collect();

        let is_similarity = match self.distance {
            Distance::DotProductRaw | Distance::Cosine => true,
            Distance::L2Norm | Distance::L1Norm => false,
        };

        let (top_k_slice, _, _) = scores.select_nth_unstable_by(k - 1, |a, b| {
            if is_similarity {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            }
        });

        top_k_slice.sort_by(|a, b| {
            if is_similarity {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            }
        });

        top_k_slice
            .iter()
            .map(|&(idx, score)| CandidateScore { index: idx, score })
            .collect()
    }
}
