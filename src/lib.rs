//! Python bindings via PyO3.

mod algos;
mod math;
mod simulation;

use pyo3::prelude::*;
use rayon::prelude::*;

#[pymodule]
mod bearing_simulation {
    use crate::algos::{Auction, Greedy, Hungarian, Solver};
    use crate::simulation::{Position, Observer, Simulation};
    use super::*;

    #[pyfunction]
    fn monte_carlo(
        observers: Vec<(f64, f64, f64)>,
        targets: Vec<(f64, f64)>,
        n_trials: usize,
        solver: &str,
    ) -> PyResult<(f64, f64)> {
        validate_solver(solver)?;
        let observers = observers.into_iter()
            .map(|(x, y, std)| Observer { pos: Position { x, y }, std })
            .collect::<Vec<_>>();

        let targets = targets.into_iter()
            .map(|(x, y)| Position { x, y })
            .collect::<Vec<_>>();
        let n_targets = targets.len();

        let simulation = Simulation { observers, targets };

        let results = (0..n_trials).into_par_iter().map(|_| {
            let solver = match solver {
                "auction" => Solver::Auction(Auction { eps: 0.01 }),
                "greedy" => Solver::Greedy(Greedy),
                "hungarian" => Solver::Hungarian(Hungarian),
                _ => unreachable!(),
            };
            simulation.simulate(&solver)
        }).collect::<Vec<_>>();

        let success_rate = results.iter()
            .filter_map(|s| s.as_ref())
            .map(|s| s.correct)
            .sum::<usize>() as f64 / (n_trials * n_targets) as f64;
        let avg_runtime = results.iter()
            .filter_map(|s| s.as_ref())
            .map(|s| s.runtime.as_micros())
            .sum::<u128>() as f64 / n_trials as f64;

        Ok((success_rate, avg_runtime))
    }

    fn validate_solver(solver: &str) -> PyResult<()> {
        if !["auction", "greedy", "hungarian"].contains(&solver) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid solver. Must be 'auction', 'greedy', or 'hungarian'."
            ));
        }
        Ok(())
    }
}