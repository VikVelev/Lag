use crate::vector::{Vector, cosine, l1_norm, l2_norm};

pub enum Distance {
    L2Norm,
    L1Norm,
    Cosine,
}

pub enum CentroidComputerType {
    KMeans,
    // TODO: Add more
}

impl Distance {
    pub fn compute(&self, vec_a: &Vector, vec_b: &Vector) -> f32 {
        match self {
            Distance::L2Norm => l2_norm(vec_a, vec_b),
            Distance::L1Norm => l1_norm(vec_a, vec_b),
            Distance::Cosine => cosine(vec_a, vec_b),
        }
    }
}

#[derive(Debug)]
pub struct CandidateScore {
    pub index: usize,
    pub candidate: Vector,
    pub score: f32,
}

pub trait VSEngine {
    fn build(&mut self);
    fn amend(&self, vec: Vector);
    fn search(&self, query: &Vector, top_k: usize) -> Vec<CandidateScore>;
}
