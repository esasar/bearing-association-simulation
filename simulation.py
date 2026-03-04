import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass

# dataclasses
@dataclass(frozen=True)
class Position:
    x: float
    y: float

# geometry
def wrap_angle(angle: float) -> float:
    """
    Wrap angle to [-pi, pi].
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

def predicted_bearings(sensor_pos: Position, target_positions: list[Position]) -> NDArray[np.float64]:
    """
    Calculate predicted bearings from sensor and target positions.
    """
    bearings: list[float] = []

    for target_position in target_positions:
        dy = target_position.y - sensor_pos.y
        dx = target_position.x - sensor_pos.x
        theta = np.arctan2(dy, dx)
        bearings.append(theta)

    return np.array(bearings)

def generate_measurements(true_bearings: NDArray[np.float64], noise_std: float) -> NDArray[np.float64]:
    """
    Add Gaussian noise to the bearings.
    """
    noise: NDArray[np.float64] = np.random.normal(0, noise_std, size=true_bearings.shape)
    return true_bearings + noise

def likelihood_matrix(measurements: NDArray[np.float64], predicted: NDArray[np.float64], sigma: float) -> NDArray[np.float64]:
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

def monte_carlo(delta: float, sigma: float, n_targets: int = 2, 
                radius: float = 1.0, n_trials: int = 10_000):
    """
    Monte Carlo simulation to estimate association accuracy.
    """
    sensor= Position(0, 0)

    true_angles = np.linspace(0, delta * (n_targets - 1), n_targets)

    # Target position calculation with radius is probably unnecessary, as
    # we only care about the angles
    target_positions = [Position(radius * np.cos(a), radius * np.sin(a)) for a in true_angles]

    predicted = predicted_bearings(sensor, target_positions)

    correct = 0

    for _ in range(n_trials):
        measurements = generate_measurements(predicted, sigma)

        # L[i, j] = 'likelihood that measurement i came from target j'
        L = likelihood_matrix(measurements, predicted, sigma)

        # linear_sum_assignment uses hungarian to find optimal assignment
        row_ind, col_ind = linear_sum_assignment(-L)

        # sort assignments so that assigned[i] = j
        # which means 'measurement i is assigned to target j'
        order = np.argsort(row_ind)
        assigned = col_ind[order]

        expected = np.arange(len(predicted))
        if assigned.shape == expected.shape and np.all(assigned == expected):
            correct += 1

    return correct / n_trials

if __name__ == "__main__":
    # uses logspace to get more points with small deltas, which is where
    # the assignement problem is degenerate
    deltas = np.logspace(np.log10(0.001), np.log10(0.75), 20, dtype=np.float64)
    sigmas = [0.01, 0.05, 0.1, 0.2]

    fig = plt.figure(figsize=(10, 6)) # type: ignore

    for sigma in sigmas:
        accuracies = [monte_carlo(delta, sigma, n_targets=2, radius=1.0, n_trials=10_000) for delta in deltas]
        plt.plot(deltas, accuracies, label=f"σ = {sigma}") # type: ignore

    plt.xlabel('Target separation (radians)') # type: ignore
    plt.ylabel('Association accuracy') # type: ignore
    plt.title('Bearing association accuracy vs. target separation') # type: ignore
    plt.xlim(0, max(deltas)) # type: ignore
    plt.grid(True) # type: ignore
    plt.legend() # type: ignore
    plt.show() # type: ignore