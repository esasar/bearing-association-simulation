use crate::algos::AssignmentResult;

/// Row-by-row greedy assignment.
///
/// Finds an un-optimal *maximum* assignment in `n^2` time, where `n` is the cardinality of the
/// input matrix.
pub fn greedy(c: &Vec<Vec<f64>>) -> Option<AssignmentResult> {
    let start = std::time::Instant::now();
    let n_rows = c.len();
    if n_rows == 0 {
        return None
    }
    let n_cols = c[0].len();

    let mut assignment = vec![usize::MAX; n_cols];
    let mut taken = vec![false; n_cols];

    for row in 0..n_rows {
        let best_col = (0..n_cols)
            .filter(|&j| !taken[j])
            .max_by(|&a, &b| c[row][a].partial_cmp(&c[row][b]).unwrap());

        if let Some(col) = best_col {
            assignment[row] = col;
            taken[col] = true;
        }
    }

    let runtime = start.elapsed();

    Some(AssignmentResult {
        assignment,
        runtime
    })
}

#[cfg(test)]
mod tests {
    use crate::algos::greedy;

    #[test]
    fn test_3x3() {
        let cost = vec![
            vec![8.0, 5.0, 9.0],
            vec![4.0, 2.0, 4.0],
            vec![7.0, 3.0, 8.0],
        ];
        let result = greedy(&cost).unwrap();
        assert_eq!(result.assignment, vec![1, 0, 2]);
    }
}