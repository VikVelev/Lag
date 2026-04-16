use crate::vector::{Vector, cosine, l1_norm, l2_norm, dot};
use rayon::prelude::*;

pub enum Distance {
    L2Norm,
    L1Norm,
    DotProductRaw,
    Cosine,
}

pub enum CentroidComputerType {
    KMeans,
    // TODO: Add more
}

impl Distance {
    pub fn compute(&self, vec_a: &[f32], vec_b: &[f32]) -> f32 {
        match self {
            Distance::L2Norm => l2_norm(vec_a, vec_b),
            Distance::L1Norm => l1_norm(vec_a, vec_b),
            Distance::Cosine => cosine(vec_a, vec_b),
            Distance::DotProductRaw => dot(vec_a, vec_b),
        }
    }
}

#[derive(Debug)]
pub struct CandidateScore {
    pub index: usize,
    // pub candidate: Vector,
    pub score: f32,
}

pub trait VSEngine: Sync {
    fn build(&mut self);
    fn amend(&self, vec: Vector);
    fn search(&self, query: &Vector, top_k: usize) -> Vec<CandidateScore>;
    
    fn search_batch(&self, queries: &[Vector], top_k: usize) -> Vec<Vec<CandidateScore>> {
        queries.par_iter().map(|q| self.search(q, top_k)).collect()
    }
}
