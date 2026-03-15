use itertools::Itertools;
use ordered_float::OrderedFloat;
use rand::distr::Distribution;
use rand_distr::Normal;
use rand::rng;
use crate::math;
use pathfinding::prelude::{Matrix, kuhn_munkres};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Position {
    pub x: f64,
    pub y: f64,
}

/// Sensor has a position and a measurement error.
pub struct Sensor {
    pub pos: Position,
    pub std: f64
}

/// Simulation has sensors and targets.
pub struct Simulation {
    pub sensors: Vec<Sensor>,
    pub targets: Vec<Position>
}

impl Simulation {
    pub fn simulate(&self) -> (bool, usize, u128) {
        let measurements = self.generate_measurements();

        let start = std::time::Instant::now();
        let ll_mat = self.combined_likelihood_matrix(&measurements);
        let weights = Self::convert_to_ordered_float(&ll_mat);
        let (_, assigned) = kuhn_munkres(&weights);
        let runtime = start.elapsed();

        let best_score: f64 = assigned.iter().enumerate().map(|(i, &j)| ll_mat[i][j]).sum();

        let competitive_count = Self::competitive_permutations(&ll_mat, best_score, 0.2);

        let expected: Vec<usize> = (0..self.targets.len()).collect();
        (assigned == expected, competitive_count, runtime.as_nanos())
    }

    pub fn predicted_bearing(sensor: &Sensor, target: &Position) -> f64 {
        let dy = target.x - sensor.pos.x;
        let dx = target.y - sensor.pos.y;

        dy.atan2(dx)
    }

    pub fn generate_measurements(&self) -> Vec<Vec<f64>> {
        self.sensors.iter().map(|sensor| {
            self.targets.iter().map(|target| {
                let pred = Self::predicted_bearing(sensor, target);
                let noise = Normal::new(0.0, sensor.std).unwrap().sample(&mut rng());
                pred + noise
            }).collect::<Vec<f64>>()
        }).collect::<Vec<Vec<f64>>>()
    }

    pub fn combined_likelihood_matrix(&self, measurements: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = self.targets.len();
        let mut ll_mat = vec![vec![1.0; n]; n];

        for (s_idx, sensor) in self.sensors.iter().enumerate() {
            for i in 0..n {
                for j in 0..n {
                    let measurement = measurements[s_idx][i];
                    let prediction = Self::predicted_bearing(sensor, &self.targets[j]);
                    ll_mat[i][j] *= math::normal_pdf(
                        math::wrap_angle(measurement - prediction),
                        0.0,
                        sensor.std
                    );
                }
            }
        }

        ll_mat
    }

    pub fn convert_to_ordered_float(matrix: &[Vec<f64>]) -> Matrix<OrderedFloat<f64>> {
        Matrix::from_rows(
            matrix.iter()
                .map(|row| row.iter().map(|&v| OrderedFloat(v)).collect())
                .collect::<Vec<Vec<OrderedFloat<f64>>>>()
        ).unwrap()
    }

    pub fn competitive_permutations(
        ll_mat: &[Vec<f64>],
        best_score: f64,
        threshold: f64
    ) -> usize {
        let n = ll_mat.len();
        let cutoff = best_score * (1.0 - threshold);
        let mut count = 0;

        for perm in (0..n).permutations(n) {
            let score: f64 = perm.iter().enumerate().map(|(i, &j)| ll_mat[i][j]).sum();
            if score >= cutoff {
                count += 1;
            }
        }

        count
    }
}