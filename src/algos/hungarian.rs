use crate::algos::AssignmentResult;
use crate::math::math::{stringify_matrix};

/// Adapted from https://en.wikipedia.org/wiki/Hungarian_algorithm.
///
/// Finds the optimal *minimum* assignment in `n^3` time, where `n` is the cardinality of the
/// cost matrix.
pub fn hungarian(cost_mat: &Vec<Vec<f64>>) -> Option<AssignmentResult> {
    let start = std::time::Instant::now();
    let j_len = cost_mat.len();
    let w_len = cost_mat[0].len();
    assert!(j_len <= w_len);

    let mut job = vec![-1i64; w_len + 1];
    let mut ys = vec![0.0f64; j_len];
    let mut yt = vec![0.0f64; w_len + 1];
    let mut costs = Vec::with_capacity(j_len);
    let inf = f64::INFINITY;

    for j_cur in 0..j_len {
        let mut w_cur = w_len;
        job[w_cur] = j_cur as i64;

        let mut min_to = vec![inf; w_len + 1];
        let mut prev = vec![-1i64; w_len + 1];
        let mut in_z = vec![false; w_len + 1];

        while job[w_cur] != -1 {
            in_z[w_cur] = true;
            let j = job[w_cur] as usize;
            let mut delta = inf;
            let mut w_next = 0usize;

            for w in 0..w_len {
                if !in_z[w] {
                    let reduced = cost_mat[j][w] - ys[j] - yt[w];
                    if reduced < min_to[w] {
                        min_to[w] = reduced;
                        prev[w] = w_cur as i64;
                    }
                    if min_to[w] < delta {
                        delta = min_to[w];
                        w_next = w;
                    }
                }
            }

            for w in 0..=w_len {
                if in_z[w] {
                    if job[w] != -1 {
                        ys[job[w] as usize] += delta;
                    }
                    yt[w] -= delta;
                } else {
                    min_to[w] -= delta;
                }
            }

            w_cur = w_next;
        }

        while w_cur != w_len {
            let w_prev = prev[w_cur] as usize;
            job[w_cur] = job[w_prev];
            w_cur = w_prev;
        }

        costs.push(-yt[w_len]);
    }

    let mut assignment = vec![0usize; j_len];
    for w in 0..w_len {
        if job[w] != -1 {
            assignment[job[w] as usize] = w;
        }
    }

    let runtime = start.elapsed();

    /*
    let mut s = String::new();
    s.push_str(&stringify_matrix(cost_mat));
    s.push_str(&stringify_matrix(&vec![assignment.clone()]));
    println!("{s}");
    */

    Some(AssignmentResult {
        assignment,
        runtime
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_3x3() {
        let cost = vec![
            vec![8.0, 5.0, 9.0],
            vec![4.0, 2.0, 4.0],
            vec![7.0, 3.0, 8.0],
        ];
        let result = hungarian(&cost).unwrap();
        assert_eq!(result.assignment, vec![0, 2, 1]);
    }
}