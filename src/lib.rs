mod math;
mod simulation;

pub use math::*;
pub use simulation::*;

#[pyo3::pymodule]
pub mod bearing_simulation {
    use super::*;

    use pyo3::prelude::*;
    use rayon::prelude::*;

    #[pyfunction]
    fn monte_carlo(
        sensors: Vec<(f64, f64, f64)>,
        targets: Vec<(f64, f64)>,
        n_trials: usize,
    ) -> PyResult<(f64, f64, f64)> {
        let sensors = sensors.into_iter()
            .map(|(x, y, std)| Sensor { pos: Position { x, y }, std })
            .collect::<Vec<_>>();

        let targets = targets.into_iter()
            .map(|(x, y)| Position { x, y })
            .collect::<Vec<_>>();

        let simulation = Simulation { sensors, targets };

        let results = (0..n_trials).into_par_iter().map(|_| {
            let (success, competitive_count, runtime) = simulation.simulate();
            (success, competitive_count, runtime)
        }).collect::<Vec<_>>();

        let success_rate = results.iter().filter(|(success, _, _)| *success).count() as f64 / n_trials as f64;
        let avg_competitive_count = results.iter().map(|(_, count, _)| *count).sum::<usize>() as f64 / n_trials as f64;
        let avg_runtime = results.iter().map(|(_, _, runtime)| *runtime).sum::<u128>() as f64 / n_trials as f64;

        Ok((success_rate, avg_competitive_count, avg_runtime))
    }
}