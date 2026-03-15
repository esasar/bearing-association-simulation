import bearing_simulation
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # uses logspace to get more points with small deltas, which is where
    # the assignment problem is degenerate
    deltas = np.linspace(0.001, np.pi, 50)
    std = 1.0

    fig = plt.figure(figsize=(10, 6))

    results = [bearing_simulation.monte_carlo(delta, std, 2, 10_000) for delta in deltas]

    accuracies = [acc for acc, _, _ in results]
    runtimes = [rt for _, rt, _ in results]
    competitive = [comp for _, _, comp in results]

    plt.subplot(1, 3, 1)
    plt.plot(deltas / std, accuracies, marker='o')
    plt.xlabel('Normalized target separation (radians / std)')
    plt.ylabel('Association accuracy (correctly associated targets / total associations)')
    plt.title('Bearing association accuracy vs. normalized target separation')
    plt.xlim(0, max(deltas / std))
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(deltas / std, runtimes, marker='o')
    plt.xlabel('Normalized target separation (radians / std)')
    plt.ylabel('Runtime (seconds)')
    plt.title('Bearing association runtime vs. normalized target separation')
    plt.xlim(0, max(deltas / std))
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(deltas / std, competitive, marker='o')
    plt.xlabel('Normalized target separation (radians / std)')
    plt.ylabel('Competitive ratio (cost / optimal cost)')
    plt.title('Bearing association competitive ratio vs. normalized target separation')
    plt.xlim(0, max(deltas / std))
    plt.grid(True)

    plt.show()