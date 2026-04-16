use num_traits::{Num, ToPrimitive};

pub type Vector = Vec<f32>;

pub type HashedVector = Vec<usize>;

pub fn l2_norm<T: ToPrimitive, U: ToPrimitive>(vec_a: &[T], vec_b: &[U]) -> f32 {
    assert_eq!(vec_a.len(), vec_b.len(), "Vectors must be the same length");

    let sum_of_squares: f32 = vec_a
        .iter()
        .zip(vec_b.iter())
        .map(|(a, b)| {
            // Convert both to f32 to handle subtraction and squaring
            let diff = a.to_f32().unwrap_or(0.0) - b.to_f32().unwrap_or(0.0);
            diff * diff
        })
        .sum();

    sum_of_squares.sqrt()
}

pub fn l1_norm<T: ToPrimitive, U: ToPrimitive>(vec_a: &[T], vec_b: &[U]) -> f32 {
    assert_eq!(vec_a.len(), vec_b.len(), "Vectors must be the same length");

    vec_a
        .iter()
        .zip(vec_b.iter())
        .map(|(a, b)| {
            let diff = a.to_f32().unwrap_or(0.0) - b.to_f32().unwrap_or(0.0);
            diff.abs()
        })
        .sum()
}

pub fn cosine<T: Num + Copy + ToPrimitive>(vec_a: &[T], vec_b: &[T]) -> f32 {
    return dot(vec_a, vec_b) / (magnitude(vec_a) * magnitude(vec_b));
}

pub fn magnitude<T: Num + Copy + ToPrimitive>(vec: &[T]) -> f32 {
    let mut sum_of_squares = 0.0f32;

    for &el in vec {
        // Convert T to f32 safely
        let val = el.to_f32().unwrap_or(0.0);
        sum_of_squares += val * val;
    }

    sum_of_squares.sqrt()
}

pub fn dot<T: Num + ToPrimitive, U: Num + ToPrimitive>(a: &[T], b: &[U]) -> f32 {
    // Basic safety check for vector operations
    if a.len() != b.len() {
        return 0.0; // Or handle as an Error/Panic
    }

    a.iter()
        .zip(b.iter())
        .map(|(val_a, val_b)| {
            let a_f = val_a.to_f32().unwrap_or(0.0);
            let b_f = val_b.to_f32().unwrap_or(0.0);
            a_f * b_f
        })
        .sum()
}
