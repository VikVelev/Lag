use std::{
    fs::File,
    io::{BufReader, Read},
    path::Path,
};

use crate::vector::Vector;

// The files written by prepare_dbpedia.py all look the same:
// u32 number of rows, u32 number of columns, then the rows one after another (little endian)
fn read_rows<T>(path: &Path, limit: Option<usize>, decode: fn([u8; 4]) -> T) -> Vec<Vec<T>> {
    let file = File::open(path).unwrap_or_else(|err| {
        panic!(
            "Can't open {} ({}) - run src/eval/download_eval_data.sh first",
            path.display(),
            err
        )
    });
    let mut reader = BufReader::with_capacity(4 * 1024 * 1024, file);

    let mut header = [0u8; 8];
    reader.read_exact(&mut header).expect("File is too short");
    let mut num_rows = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
    let num_cols = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;

    if let Some(limit) = limit {
        num_rows = std::cmp::min(num_rows, limit);
    }

    let mut rows = Vec::with_capacity(num_rows);
    let mut row_bytes = vec![0u8; num_cols * 4];
    for _ in 0..num_rows {
        reader.read_exact(&mut row_bytes).expect("File is too short");
        rows.push(
            row_bytes
                .chunks_exact(4)
                .map(|bytes| decode(bytes.try_into().unwrap()))
                .collect(),
        );
    }

    return rows;
}

pub fn read_vectors(path: &Path, limit: Option<usize>) -> Vec<Vector> {
    return read_rows(path, limit, f32::from_le_bytes);
}

pub fn read_ids(path: &Path, limit: Option<usize>) -> Vec<Vec<u32>> {
    return read_rows(path, limit, u32::from_le_bytes);
}
