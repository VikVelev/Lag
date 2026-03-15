pub type Vector = Vec<f32>;

pub fn l2_norm(vec_a: &Vector, vec_b: &Vector) -> f32 {
    panic!()
}

pub fn l1_norm(vec_a: &Vector, vec_b: &Vector) -> f32 {
    panic!()
}

pub fn cosine(vec_a: &Vector, vec_b: &Vector) -> f32 {
    return dot(vec_a, vec_b) / (magnitude(vec_a) * magnitude(vec_b));
}

fn magnitude(vec: &Vector) -> f32 {
    let mut sum_of_squares = 0.0f32;
    for el in vec {
        sum_of_squares += el * el;
    }

    return sum_of_squares.sqrt();
}

fn dot(vec_a: &Vector, vec_b: &Vector) -> f32 {
    let mut result = 0f32;
    for i in 0..vec_a.len() {
        result += vec_a[i] * vec_b[i];
    }
    return result;
}
