const FRAC_1_SQRT_2_PI: f64 = std::f64::consts::FRAC_1_SQRT_2 * std::f64::consts::FRAC_2_SQRT_PI;

/// Wraps an angle to the range [-pi, pi].
pub fn wrap_angle(angle: f64) -> f64 {
    (angle + std::f64::consts::PI) % (2.0 * std::f64::consts::PI) - std::f64::consts::PI
}

/// Generates `n` linearly spaced values in the range [start, end].
pub fn linear_space(start: f64, end: f64, n: usize) -> Vec<f64> {
    let step = (end - start) / (n - 1) as f64;

    (0..n).map(|i| start + step * i as f64).collect()
}

/// Evaluates PDF of normal distribution with `mean` and `std` at `x`.
pub fn normal_pdf(x: f64, mean: f64, std: f64) -> f64 {
    let coefficient = FRAC_1_SQRT_2_PI / std.sqrt();
    let exponent = -0.5 * ((x - mean) / std).powi(2);
    coefficient * exponent.exp()
}
