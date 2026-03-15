#[pyo3::pymodule]
mod bearing_simulation {
    use ordered_float::OrderedFloat;
    use pyo3::prelude::*;
    use pathfinding::prelude::{Matrix, kuhn_munkres};
    use rand::rng;
    use rand_distr::{Normal, Distribution};
    use std::time::{Duration, Instant};
    use rayon::iter::Positions;
    use rayon::prelude::*;
    use itertools::Itertools;

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Position {
        x: f64,
        y: f64,
    }

    fn wrap_angle(angle: f64) -> f64 {
        (angle + std::f64::consts::PI) % (2.0 * std::f64::consts::PI) - std::f64::consts::PI
    }

    fn likelihood_matrix(
        measurements: &[f64],
        predicted: &[f64],
        std: f64
    ) -> Vec<Vec<f64>> {
        let n = measurements.len();
        let mut ll_mat = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                let diff = wrap_angle(measurements[i] - predicted[j]);
                ll_mat[i][j] = (-0.5 * (diff / std).powi(2)).exp();
            }
        }

        ll_mat
    }

    fn predicted_bearing(sensor_pos: &Position, target_pos: Position) -> f64 {
        let dy = target_pos.x - sensor_pos.x;
        let dx = target_pos.y - sensor_pos.y;

        dy.atan2(dx)
    }

    fn linear_space(start: f64, end: f64, n: usize) -> Vec<f64> {
        let step = (end - start) / (n - 1) as f64;

        (0..n).map(|i| start + step * i as f64).collect()
    }

    fn convert_to_ordered_float(matrix: &[Vec<f64>]) -> Matrix<OrderedFloat<f64>> {
        Matrix::from_rows(
            matrix.iter()
                .map(|row| row.iter().map(|&v| OrderedFloat(v)).collect())
                .collect::<Vec<Vec<OrderedFloat<f64>>>>()
        ).unwrap()
    }

    fn competitive_permutations(l: &[Vec<f64>], best_score: f64, threshold: f64) -> usize {
        let n = l.len();

        let cutoff = best_score * (1.0 - threshold);

        let mut count = 0;

        for perm in (0..n).permutations(n) {
            let score: f64 = perm.iter().enumerate().map(|(i, &j)| l[i][j]).sum();
            if score >= cutoff {
                count += 1;
            }
        }

        count
    }

    fn run_trial(
        measurements: &[f64],
        predictions: &[f64],
        std: f64,
        n_targets: usize
    ) -> (bool, usize) {
        let ll_mat = likelihood_matrix(&predictions, &measurements, std);

        let weights: Matrix<OrderedFloat<f64>> = convert_to_ordered_float(&ll_mat);

        let (_, assigned) = kuhn_munkres(&weights);

        let best_score: f64 = assigned.iter().enumerate().map(|(i, &j)| ll_mat[i][j]).sum();

        let competitive_count = competitive_permutations(&ll_mat, best_score, 0.05);

        let expected: Vec<usize> = (0..n_targets).collect();
        (assigned == expected, competitive_count)
    }

    #[pyfunction]
    fn monte_carlo(
        separation: f64,
        std: f64,
        n_targets: usize,
        n_trials: usize,
    ) -> PyResult<(f64, f64, f64)> {
        let predictions = linear_space(0.0, separation * ((n_targets - 1) as f64), n_targets);
        let start = Instant::now();

        let (correct, total_competitive) = (0..n_trials)
            .into_par_iter()
            .fold(
                || (0u32, 0u64),
                |mut acc, _| {
                    let measurements = predictions
                        .iter()
                        .map(|&p| p + Normal::new(0.0, std).unwrap().sample(&mut rng()))
                        .collect::<Vec<f64>>();

                    let (is_correct, competitive) =
                        run_trial(&measurements, &predictions, std, n_targets);

                    if is_correct {
                        acc.0 += 1;
                    }

                    acc.1 += competitive as u64;

                    acc
                },
            )
            .reduce(
                || (0, 0),
                |a, b| (a.0 + b.0, a.1 + b.1),
            );

        let total_duration = start.elapsed();

        let accuracy = correct as f64 / n_trials as f64;
        let avg_runtime = total_duration.as_secs_f64() / n_trials as f64;
        let avg_competitive = total_competitive as f64 / n_trials as f64;

        Ok((accuracy, avg_runtime, avg_competitive))
    }
}