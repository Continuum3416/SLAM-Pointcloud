import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def load_trajectory(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("#"):  # Skip comments
                continue
            values = list(map(float, line.strip().split()))
            timestamp, tx, ty, tz = values[:4]  # Extract position data
            data.append([timestamp, tx, ty, tz])
    return np.array(data)

def detect_and_correct_loop_closure(trajectory, loop_threshold=0.1):
    """
    Detect loop closure by checking if the current position is close to a previous position.
    Apply corrections to reduce drift after detecting a loop closure.
    """
    corrected_trajectory = trajectory.copy()
    for i in range(1, len(trajectory)):
        current_pos = trajectory[i, 1:4]
        # Calculate distances to all previous positions
        distances = cdist([current_pos], trajectory[:i, 1:4], metric='euclidean')
        min_distance = np.min(distances)
        min_index = np.argmin(distances)
        
        # If distance is smaller than the threshold, a loop closure is detected
        if min_distance < loop_threshold:
            # Apply a correction by aligning the current trajectory with the detected loop closure point
            correction = trajectory[min_index, 1:4] - current_pos
            corrected_trajectory[i, 1:4] += correction  # Apply the correction to the current position
    return corrected_trajectory

def plot_trajectory_with_correction(original_trajectory, corrected_trajectory, output_image_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the original trajectory
    ax.plot(original_trajectory[:, 1], original_trajectory[:, 2], original_trajectory[:, 3],
            label="Original Trajectory", color='blue')

    # Plot the corrected trajectory
    ax.plot(corrected_trajectory[:, 1], corrected_trajectory[:, 2], corrected_trajectory[:, 3],
            label="Corrected Trajectory", color='green')

    # Mark keyframes
    ax.scatter(original_trajectory[:, 1], original_trajectory[:, 2], original_trajectory[:, 3],
               c='red', marker='o', label="Keyframes")

    # Labels and title
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Trajectory with Loop Closure Correction")
    ax.legend()

    # Save the plot as an image
    plt.savefig(output_image_path)
    print(f"Trajectory plot saved as: {output_image_path}")

    # Show the interactive graph
    plt.show()

def main():
    # Path to your keyframe trajectory file
    trajectory_file = "MyVideoKeyFrameTrajectoryTUMFormat.txt"
    output_image_path = "Corrected_Trajectory.png"

    # Load the trajectory data
    trajectory_data = load_trajectory(trajectory_file)

    # Detect loop closures and apply corrections
    corrected_trajectory = detect_and_correct_loop_closure(trajectory_data)

    # Plot and save the trajectory with corrections
    plot_trajectory_with_correction(trajectory_data, corrected_trajectory, output_image_path)

if __name__ == "__main__":
    main()
