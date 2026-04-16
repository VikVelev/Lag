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

    // The dummest algorithm possible:
    // Compute distance to each vector - store them in an array
    // Compute cut-off and get the top_k
    fn search(&self, query: &Vector, top_k: usize) -> Vec<CandidateScore> {
        let mut scores: Vec<CandidateScore> = self
            .references
            .iter()
            .map(|reference| CandidateScore {
                index: 0,
                candidate: reference.to_vec(),
                score: self.distance.compute(query, reference),
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
