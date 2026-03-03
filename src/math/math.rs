use std::fmt::Display;

/// Wraps an angle to the range [-pi, pi].
pub fn wrap_angle(angle: f64) -> f64 {
    (angle + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI
}

/// Generates `n` linearly spaced values in the range `start`, `end` inclusive.
#[allow(dead_code)]
pub fn linear_space(start: f64, end: f64, n: usize) -> Vec<f64> {
    let step = (end - start) / (n - 1) as f64;

    (0..n).map(|i| start + step * i as f64).collect()
}

/// Evaluates PDF of normal distribution with `mean` and `std` at `x`.
///
/// ```
/// f(x) = 1/(σ * sqrt(2π)) * exp(-0.5 * ((x - μ)/σ)^2)
/// ```
#[allow(dead_code)]
pub fn normal_pdf(x: f64, mean: f64, std: f64) -> f64 {
    let coefficient = 1.0 / (std * (2.0 * std::f64::consts::PI).sqrt());
    let exponent = -0.5 * ((x - mean) / std).powi(2);
    coefficient * exponent.exp()
}

/// Evaluates log PDF of normal distribution with `mean` and `std` at `x`.
///
/// ```
/// log f(x) = -ln(σ) - 0.5 * ln(2π) - 0.5 * ((x - μ)/σ)²
/// ```
#[allow(dead_code)]
pub fn normal_log_pdf(x: f64, mean: f64, std: f64) -> f64 {
    let log_coefficient = -(std.ln() + 0.5 * (2.0 * std::f64::consts::PI).ln());
    let exponent = -0.5 * ((x - mean) / std).powi(2);
    log_coefficient + exponent
}

/// Flips sign of given matrix.
#[allow(dead_code)]
pub fn negate_mat(c: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    c.into_iter().map(|row| row.into_iter().map(|v| -v).collect()).collect()
}

/// For debugging
#[allow(dead_code)]
pub fn stringify_matrix<T: Display>(m: &Vec<Vec<T>>) -> String {
    let mut s = String::from("{\n");
    for row in m.iter() {
        let entries = row.iter()
            .map(|x| format!("{}", x))
            .collect::<Vec<_>>()
            .join(", ");
        s.push_str(&format!("    {{ {entries}, }},\n"));
    }
    s.push('}');
    s
}