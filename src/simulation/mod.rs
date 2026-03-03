//! Simulation framework for observer-target assignment evaluation.

use std::time::{Duration};
use rand::distr::Distribution;
use rand::prelude::SliceRandom;
use rand::rng;
use rand_distr::Normal;
use crate::algos::Solver;
use crate::math::math::{normal_log_pdf, normal_pdf, wrap_angle};

/// 2D position representation.
#[derive(Debug, Clone, Copy)]
pub struct Position {
    pub x: f64,
    pub y: f64,
}

/// Observer.
pub struct Observer {
    /// Position of the observer.
    pub pos: Position,
    /// Observation error in radians.
    pub std: f64
}

/// Simulation.
pub struct Simulation {
    /// Observers.
    pub observers: Vec<Observer>,
    /// Targets.
    pub targets: Vec<Position>,
}

/// Simulation result.
pub struct SimulationResult {
    /// Number of correct assignments.
    pub correct: usize,
    /// Runtime of the simulation.
    pub runtime: Duration
}

impl Simulation {
    /// Run a single simulation trial with the given solver.
    pub fn simulate(&self, solver: &Solver) -> Option<SimulationResult> {
        let mut rng = rng();

        let mut measurements = self.generate_measurements();

        // shuffle measurements, as having them diagonally introduces bias
        let mut perm: Vec<usize> = (0..measurements[0].len()).collect();
        perm.shuffle(&mut rng);
        measurements = measurements.iter()
            .map(|row| perm.iter().map(|&i| row[i].clone()).collect())
            .collect();

        let ll_mat = self.combined_likelihood_matrix(&measurements);

        // let expected: Vec<usize> = (0..self.targets.len()).collect();
        // permutation used in shuffling is the expected order
        let expected = perm;

        let solution = solver.solve(&ll_mat)?;

        let correct = solution.assignment
            .iter()
            .zip(&expected)
            .filter(|(assigned, expected)| assigned == expected)
            .count();

        Some(SimulationResult {
            correct,
            runtime: solution.runtime
        })
    }

    pub fn predicted_bearing(observer: &Observer, target: &Position) -> f64 {
        let dx = target.x - observer.pos.x;
        let dy = target.y - observer.pos.y;

        dy.atan2(dx)
    }

    pub fn generate_measurements(&self) -> Vec<Vec<f64>> {
        self.observers.iter().map(|observer| {
            self.targets.iter().map(|target| {
                let pred = Self::predicted_bearing(observer, target);
                let noise = Normal::new(0.0, observer.std).unwrap().sample(&mut rng());
                pred + noise
            }).collect::<Vec<f64>>()
        }).collect::<Vec<Vec<f64>>>()
    }

    pub fn combined_likelihood_matrix(&self, measurements: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = self.targets.len();
        let mut ll_mat = vec![vec![0.0; n]; n];

        for (s_idx, observer) in self.observers.iter().enumerate() {
            for i in 0..n {
                for j in 0..n {
                    let measurement = measurements[s_idx][i];
                    let prediction = Self::predicted_bearing(observer, &self.targets[j]);
                    ll_mat[i][j] += normal_log_pdf(
                        wrap_angle(measurement - prediction),
                        0.0,
                        observer.std
                    );
                }
            }
        }

        ll_mat
    }
}