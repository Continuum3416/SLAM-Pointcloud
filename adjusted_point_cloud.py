import numpy as np
import open3d as o3d
from sklearn.linear_model import RANSACRegressor

# Load the PLY file
ply_file = "./point_cloud/output.ply"
pcd = o3d.io.read_point_cloud(ply_file)
xyz = np.asarray(pcd.points)

# Extract X, Y, Z coordinates
x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

# Fit a plane to (X, Y) -> Z using RANSAC
ransac = RANSACRegressor()
ransac.fit(np.column_stack((x, y)), z)  # Predict Z from (X, Y)

# Compute the ground plane offset (predicted Z values)
z_plane = ransac.predict(np.column_stack((x, y)))

# Correct Z values by shifting them to align with the estimated plane
z_corrected = z - np.mean(z_plane)

# Save the corrected point cloud
xyz_corrected = np.column_stack((x, y, z_corrected))
corrected_pcd = o3d.geometry.PointCloud()
corrected_pcd.points = o3d.utility.Vector3dVector(xyz_corrected)

# Save to ASCII PLY format
output_file = "./point_cloud/corrected_pointcloud.ply"
o3d.io.write_point_cloud(output_file, corrected_pcd, write_ascii=True)

print(f"Saved corrected point cloud to {output_file} in ASCII format")
