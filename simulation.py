import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import linear_sum_assignment

def wrap_angle(angle):
    """
    Wrap angle to [-pi, pi].
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

def predicted_bearings(sensor_pos, target_positions):
    """
    Calculate predicted bearings from sensor and target positions.
    """
    sx, sy = sensor_pos
    bearings = []

    for tx, ty in target_positions:
        theta = np.arctan2(ty - sy, tx - sx)
        bearings.append(theta)

    return np.array(bearings)

def generate_measurements(true_bearings, noise_std):
    """
    Add Gaussian noise to the bearings.
    """
    noise = np.random.normal(0, noise_std, size=true_bearings.shape)
    return true_bearings + noise

def likelihood_matrix(measurements, predicted, sigma):
    """
    Compute likelihood matrix from the measurement/prediction residual.
    """
    n = len(measurements)
    L = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            diff = wrap_angle(measurements[i] - predicted[j])
            L[i, j] = np.exp(-0.5 * (diff / sigma)**2)

    return L

def monte_carlo(delta, sigma, n_targets=2, radius=1.0, n_trials=10_000):
    """
    Monte Carlo simulation to estimate association accuracy.
    """
    sensor = (0, 0)

    r = float(radius)

    # Compute angles so that adjacent targets are separated by `delta`.
    indices = np.arange(n_targets)
    center = (n_targets - 1) / 2.0
    angles = (indices - center) * delta

    target_positions = [(r * np.cos(a), r * np.sin(a)) for a in angles]

    predicted = predicted_bearings(sensor, target_positions)

    correct = 0

    for _ in range(n_trials):
        measurements = generate_measurements(predicted, sigma)

        L = likelihood_matrix(measurements, predicted, sigma)

        row_ind, col_ind = linear_sum_assignment(-L)

        order = np.argsort(row_ind)
        assigned = col_ind[order]

        expected = np.arange(len(predicted))
        if assigned.shape == expected.shape and np.all(assigned == expected):
            correct += 1

    return correct / n_trials

if __name__ == "__main__":
    deltas = np.logspace(np.log10(0.001), np.log10(0.75), 20)
    sigmas = [0.01, 0.05, 0.1, 0.2]

    plt.figure(figsize=(10, 6))

    for sigma in sigmas:
        accuracies = [monte_carlo(delta, sigma, n_targets=2, radius=1.0, n_trials=10_000) for delta in deltas]
        plt.plot(deltas, accuracies, label=f"σ = {sigma}")

    plt.xlabel('Target separation (radians)')
    plt.ylabel('Association accuracy')
    plt.title('Bearing association accuracy vs. target separation')
    plt.xlim(0, max(deltas))
    plt.grid(True)
    plt.legend()
    plt.show()