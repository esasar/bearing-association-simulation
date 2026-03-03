use crate::algos::AssignmentResult;

/// Bertsekas auction
pub fn auction(cost_mat: &Vec<Vec<f64>>, eps: f64) -> Option<AssignmentResult> {
    let start = std::time::Instant::now();
    let n_rows = cost_mat.len();
    if n_rows == 0 {
        return None
    }
    let n_cols = cost_mat[0].len();

    let mut prices = vec![0.0f64; n_cols];
    let mut row_to_col = vec![usize::MAX; n_rows];
    let mut col_to_row = vec![usize::MAX; n_cols];

    let mut unassigned: Vec<usize> = (0..n_rows).collect();

    while !unassigned.is_empty() {
        let mut next_unassigned: Vec<usize> = Vec::new();

        for &bidder in &unassigned {
            let mut best_col = usize::MAX;
            let mut best_val = f64::NEG_INFINITY;
            let mut second_val = f64::NEG_INFINITY;

            for j in 0..n_cols {
                let val = cost_mat[bidder][j] - prices[j];
                if val > best_val {
                    second_val = best_val;
                    best_val = val;
                    best_col = j;
                } else if val > second_val {
                    second_val = val;
                }
            }

            if best_col == usize::MAX {
                continue;
            }

            let bid_increment = if second_val == f64::NEG_INFINITY {
                best_val + eps
            } else {
                (best_val - second_val) + eps
            };

            prices[best_col] += bid_increment;

            let prev_owner = col_to_row[best_col];
            if prev_owner != usize::MAX {
                row_to_col[prev_owner] = usize::MAX;
                next_unassigned.push(prev_owner);
            }

            row_to_col[bidder] = best_col;
            col_to_row[best_col] = bidder;
        }

        unassigned = next_unassigned;
    }

    let runtime = start.elapsed();

    Some(AssignmentResult {
        assignment: row_to_col,
        runtime
    })
}