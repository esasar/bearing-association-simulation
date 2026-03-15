import bearing_simulation
import matplotlib.pyplot as plt
import numpy as np

def generate_sensors(n_sensors: int,
                     sensor_separation: float,
                     stds: list[float]):
    sensors = []
    for i in range(n_sensors):
        offset = (i - (n_sensors - 1) / 2) * sensor_separation
        sensors.append((offset, 0, stds[i]))

    return sensors

def generate_targets(n_targets: int, sensor_separation: float, sensor_target_separation: float):
    targets = []
    for i in range(n_targets):
        offset = (i - (n_targets - 1) / 2) * sensor_separation
        targets.append((offset, sensor_target_separation))
    return targets

def draw_bearings(ax, sensors, targets):
    for sensor_idx, sensor in enumerate(sensors):
        sensor_x, sensor_y, std = sensor

        for target in targets:
            target_x, target_y = target
            ax.plot([sensor_x, target_x], [sensor_y, target_y], 'b-', alpha=0.3, linewidth=1)

            true_bearing = np.arctan2(target_y - sensor_y, target_x - sensor_x)

            # draw a line from sensor_x, sensor_y in the direction of true_bearing - std to true_bearing + std
            bearing_start = true_bearing - std
            bearing_end = true_bearing + std

            # length of the cone should be the same as the length of the true bearing line
            cone_length = np.sqrt((target_x - sensor_x) ** 2 + (target_y - sensor_y) ** 2)
            bearing_x = [sensor_x, sensor_x + cone_length * np.cos(bearing_start)]
            bearing_y = [sensor_y, sensor_y + cone_length * np.sin(bearing_start)]
            ax.plot(bearing_x, bearing_y, 'r-', alpha=0.5, linewidth=2)
            bearing_x = [sensor_x, sensor_x + cone_length * np.cos(bearing_end)]
            bearing_y = [sensor_y, sensor_y + cone_length * np.sin(bearing_end)]
            ax.plot(bearing_x, bearing_y, 'r-', alpha=0.5, linewidth=2)



if __name__ == "__main__":
    # simulation params
    target_seps = np.linspace(0.001, np.pi, 50)
    std = 0.2
    n_targets = 5
    n_trials = 10_000
    n_sensors = 5
    sensor_sep = 1.0
    sensor_target_sep = 5.0

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    pos_ax = axs[0, 0]
    accuracy_ax = axs[0, 1]
    competitive_ratio_ax = axs[1, 0]
    runtime_ax = axs[1, 1]

    sensors = generate_sensors(n_sensors, sensor_sep, [std] * n_sensors)

    max_sensor_x = (n_sensors - 1) / 2 * sensor_sep
    max_target_x = (n_targets - 1) / 2 * max(target_seps)
    axis_limit = max(max(max_sensor_x, max_target_x), sensor_target_sep)

    accuracy_data = []
    runtime_data = []
    competitive_data = []

    for target_sep in target_seps:
        targets = generate_targets(n_targets, target_sep, sensor_target_sep)

        success_rate, avg_competitive_count, runtime \
            = bearing_simulation.monte_carlo(sensors, targets, 10_000)

        accuracy_data.append(success_rate)
        competitive_data.append(avg_competitive_count)
        runtime_data.append(runtime)

        pos_ax.clear()
        pos_ax.set_title(f"Target and Sensor Positions")

        draw_bearings(pos_ax, sensors, targets)

        target_x = [target[0] for target in targets]
        target_y = [target[1] for target in targets]
        pos_ax.scatter(target_x, target_y, color='red', label='Targets')

        sensor_x = [sensor[0] for sensor in sensors]
        sensor_y = [sensor[1] for sensor in sensors]

        pos_ax.scatter(sensor_x, sensor_y, color='blue', label='Sensors')
        pos_ax.set_xlim(-axis_limit, axis_limit)
        pos_ax.set_ylim(-axis_limit, axis_limit)

        accuracy_ax.clear()
        accuracy_ax.set_title(f"Localization Accuracy vs Delta")
        accuracy_ax.plot(target_seps[:len(accuracy_data)], accuracy_data, marker='o')
        accuracy_ax.set_xlabel("Delta (radians)")
        accuracy_ax.set_ylabel("Localization Accuracy")

        competitive_ratio_ax.clear()
        competitive_ratio_ax.set_title(f"Average Competitive Count vs Delta")
        competitive_ratio_ax.plot(target_seps[:len(competitive_data)], competitive_data, marker='o')
        competitive_ratio_ax.set_xlabel("Delta (radians)")
        competitive_ratio_ax.set_ylabel("Average Competitive Count")

        runtime_ax.clear()
        runtime_ax.set_title(f"Runtime vs Delta")
        runtime_ax.plot(target_seps[:len(runtime_data)], runtime_data, marker='o')
        runtime_ax.set_xlabel("Delta (radians)")
        runtime_ax.set_ylabel("Runtime (nanoseconds)")

        plt.tight_layout()
        plt.pause(1)

    plt.show()
