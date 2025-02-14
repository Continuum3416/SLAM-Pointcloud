import os
from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

def save_pointcloud_from_ORB_SLAM(input_file: Union[Path, str], output_file: Union[Path,str] = "out.ply"):
    """Converts a comma separated list of map point coordinates into
    PLY format for viewing the generated map.

    Args:
        input_file (str or Path): Path to the input file which is expected to
        be a .csv file with the columns pos_x, pos_y, pos_z designating the
        coordinates of the points in the world reference frame.

        output_file (str or Path): Path to the output .ply file, format is
        described here: https://paulbourke.net/dataformats/ply/
    """

    coords = np.genfromtxt(input_file, delimiter=", ", skip_header=1)

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    ply_header = 'ply\n' \
                'format ascii 1.0\n' \
                'element vertex %d\n' \
                'property float x\n' \
                'property float y\n' \
                'property float z\n' \
                'end_header' % x.shape[0]

    np.savetxt(output_file, np.column_stack((x, y, z)), fmt='%f %f %f', header=ply_header, comments='')

def save_trajectory_from_ORB_SLAM(input_file: Union[Path, str], output_file: Union[Path,str] = "out_trajectory.ply"):
    """Converts the saved trajectory file from ORB-SLAM3 to a point cloud to then
    show alongside the mapped cloud.

    The input file is expected to be in the format (KITTI format with image path prepended, can ignore it):
    /path/to/image0.png R_00 R_01 R_02 t_0 R_10 R_11 R_12 t_1 R_20 R_21 R_22 t_2
    /path/to/image1.png R_00 R_01 R_02 t_0 R_10 R_11 R_12 t_1 R_20 R_21 R_22 t_2

    Where the R terms are the rotation and t terms are the translation terms
    of the homogeneous transformation matrix T_w_cam0.
    """
    x = []
    y = []
    z = []

    with open(input_file, "r") as file:
        lines = file.readlines()

    for line in lines:
        cols = line.strip().split(" ")

        x.append(float(cols[4]))
        y.append(float(cols[8]))
        z.append(float(cols[12]))

    # Each trajectory point is shown as a "point" in the "point cloud" which is the trajectory
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    # RGB values for each point on the trajectory, set to be light green
    r = np.ones_like(x) * 144
    g = np.ones_like(x) * 238
    b = np.ones_like(x) * 144

    ply_header = 'ply\n' \
                'format ascii 1.0\n' \
                'element vertex %d\n' \
                'property float x\n' \
                'property float y\n' \
                'property float z\n' \
                'property uchar red\n' \
                'property uchar green\n' \
                'property uchar blue\n' \
                'end_header' % x.shape

    np.savetxt(output_file, np.column_stack((x, y, z, r, g, b)), fmt='%f %f %f %d %d %d', header=ply_header, comments='')


def plot_3d_point_cloud(ply_file):
    """
    Reads a PLY file and plots a 3D point cloud.

    Parameters:
    ply_file (str): Path to the PLY file.
    """
    # Read the PLY file
    with open(ply_file, "r") as file:
        lines = file.readlines()

    # Find the header's end
    header_end = 0
    for i, line in enumerate(lines):
        if line.strip() == "end_header":
            header_end = i + 1
            break

    # Extract point coordinates (ignoring the header)
    points = np.loadtxt(lines[header_end:], dtype=np.float32)

    # Extract x, y, z coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Create a 3D plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=z, cmap='jet', marker='o', s=2)

    # Labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Point Cloud")

    plt.show()


if __name__ == "__main__":
    input_file = "/home/autonomy/Dev/YOLO_Py/point_cloud/PointCloud.txt" # ReferenceMapPoints seem to work better, use that file
    output_file = "./point_cloud/output.ply"

    input_trajectory = "/home/autonomy/Dev/YOLO_Py/MyVideoKeyFrameTrajectoryTUMFormat.txt"
    output_trajectory = "./output_trajectory.ply"

    save_pointcloud_from_ORB_SLAM(input_file, output_file)
    #save_trajectory_from_ORB_SLAM(input_trajectory, output_trajectory)
    plot_3d_point_cloud(output_file)