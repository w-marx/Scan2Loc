import numpy as np
import open3d as o3d
from typing import Any

def lines_3d_for_o3d(lines3d:np.ndarray, colors:np.ndarray, radius:float = 0.005, resolution = 6)->list[Any]:
    """
    :param lines3d: Nx6 line array
    :param colors: Nx3 colors
    """
    directions = lines3d[:, 3:]-lines3d[:, :3]
    lengths = np.linalg.norm(directions, axis=1)

    directions_norm = directions/np.linalg.norm(directions, axis=1, keepdims=True)
    mid_points = lines3d[:, :3] + directions/2

    to_vis = []

    for i, _ in enumerate(lines3d):
        if lengths[i] < 1e-6:
            continue

        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius,
            height=lengths[i],
            resolution=resolution
        )
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, directions_norm[i])
        rotation_angle = np.arccos(np.clip(np.dot(z_axis, directions_norm[i]), -1, 1))
    
        if np.linalg.norm(rotation_axis) > 1e-6:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            R = o3d.geometry.TriangleMesh.get_rotation_matrix_from_axis_angle(
                rotation_axis * rotation_angle
            )
            cylinder.rotate(R, center=[0, 0, 0])
    
        cylinder.translate(mid_points[i])
        cylinder.paint_uniform_color(colors[i, :3])
        to_vis.append(cylinder)

    return to_vis