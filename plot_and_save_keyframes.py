import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Load the keyframe trajectory


def load_trajectory(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("#"):  # Skip comments
                continue
            values = list(map(float, line.strip().split()))
            timestamp, tx, ty, tz = values[:4]  # Extract the position data
            # Include timestamp for matching
            data.append([timestamp, tx, ty, tz])
    return np.array(data)

def load_yolo_detections(file_path):
    detections = []
    with open(file_path, 'r') as f:
        for line in f:
            values = line.strip().split(", ")
            # Skip lines that don't have exactly 3 values (timestamp, object_class, confidence)
            if len(values) < 3:
                print(f"Skipping invalid line (incorrect format): {line}")
                continue

            try:
                timestamp, obj, conf = values[:3]  # Only keep timestamp, object_class, and confidence
                timestamp = float(timestamp)  # Ensure timestamp is a float
                conf = float(conf)  # Ensure confidence is a float

                # Only keep "person" detections
                if obj == "person":
                    # Extract only the timestamp for matching with keyframe trajectory later
                    detections.append((timestamp, obj, conf))

            except ValueError as e:
                print(f"Skipping line due to error (invalid value): {line} | Error: {e}")
                continue

    return detections

# Match YOLO detections to keyframes
def match_detections_to_keyframes(detections, keyframes, threshold=0.1):
    marked_keyframes = []
    for det_time, obj, conf in detections:
        closest_keyframe = min(keyframes, key=lambda kf: abs(kf[0] - det_time))
        if abs(closest_keyframe[0] - det_time) <= threshold:
            # Append x, y, z and detection details
            marked_keyframes.append((*closest_keyframe[1:], obj, conf))
    return marked_keyframes

# Plot the trajectory with YOLO detections

def set_axes_equal(ax):
    """
    Set 3D plot axes to equal scale for accurate visualization.
    This is necessary because Matplotlib's 3D plots do not set axes to equal scale by default.
    """
    extents = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
    centers = np.mean(extents, axis=1)
    max_range = np.max(extents[:, 1] - extents[:, 0]) / 2

    ax.set_xlim(centers[0] - max_range, centers[0] + max_range)
    ax.set_ylim(centers[1] - max_range, centers[1] + max_range)
    ax.set_zlim(centers[2] - max_range, centers[2] + max_range)


def plot_and_save_trajectory(data, detections, output_image_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the trajectory (without timestamps)
    ax.plot(data[:, 1], data[:, 2], data[:, 3],
            label="Keyframe Trajectory", color='blue')

    # Mark keyframes
    ax.scatter(data[:, 1], data[:, 2], data[:, 3],
               c='red', marker='o', label="Keyframes")

    # Label the starting point
    ax.scatter(data[0, 1], data[0, 2], data[0, 3], c='green',
               marker='o', s=100, label="Start Point")
    ax.text(data[0, 1], data[0, 2], data[0, 3], "Start", color='green')

    # Label the ending point
    ax.scatter(data[-1, 1], data[-1, 2], data[-1, 3],
               c='purple', marker='o', s=100, label="End Point")
    ax.text(data[-1, 1], data[-1, 2], data[-1, 3], "End", color='purple')

    # Mark YOLO detections for people (just once in the legend)
    people_detected = False
    for detection in detections:
        x, y, z, obj, _ = detection
        # Mark all detected people as orange triangles
        ax.scatter(x, y, z, c='orange', marker='^', s=100)
        if not people_detected:
            ax.scatter(x, y, z, c='orange', marker='^',
                       s=100, label="Person Detected")
            people_detected = True  # Only label once for people detections

    # Labels and title
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    # ax.set_title("Keyframe Trajectory with YOLO Detections")
    # ax.legend()

    # Save the plot as an image
    plt.savefig(output_image_path)
    print(f"Trajectory plot saved as: {output_image_path}")

    # Show the interactive graph
    plt.show()




# Main function


def main():
    # Replace with your keyframe file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    trajectory_file = os.path.join(script_dir, "MyVideoKeyFrameTrajectoryTUMFormat.txt")
    yolo_detection_file = os.path.join(script_dir, "yolo_detection.txt")
    output_image_path = os.path.join(script_dir, "KeyframeTrajectoryWithPersonDetections.png")

    # Load trajectory data
    if not os.path.exists(trajectory_file):
        print(f"Error: File {trajectory_file} not found.")
        return

    if not os.path.exists(yolo_detection_file):
        print(f"Error: File {yolo_detection_file} not found.")
        return

    trajectory_data = load_trajectory(trajectory_file)
    yolo_detections = load_yolo_detections(yolo_detection_file)

    # Match YOLO detections to keyframes
    marked_keyframes = match_detections_to_keyframes(
        yolo_detections, trajectory_data)

    # Plot and save the trajectory
    plot_and_save_trajectory(
        trajectory_data, marked_keyframes, output_image_path)


if __name__ == "__main__":
    main()