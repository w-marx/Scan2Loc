import numpy as np
from typing import Literal
import open3d as o3d
import matplotlib.pyplot as plt

from shared.assertion_helpers import assert_intrinsic_mat, assert_homogeneous_mat


def pixels_to_rays(intrinsics:np.ndarray, base_t_cam:np.ndarray, uv_s:np.ndarray)->tuple[np.ndarray, np.ndarray]:
    """
    :param intrinsics: 3x3 intrinsic camera matrix
    :param base_t_cam: 4x4 homogeneout base-T_cam transformation matrix
    :param uv_s: Nx2 Pixel coordinates
    :return: length 3 array of the origin and a Nx3 array of normalized ray directions
    """
    assert assert_intrinsic_mat(intrinsics)
    assert assert_homogeneous_mat(base_t_cam, size = 4)
    assert uv_s.ndim == 2 and uv_s.shape[-1] == 2, f"wrong uv shape: {uv_s.shape}"

    n_pixel = uv_s.shape[0]

    if n_pixel < 1:
        return base_t_cam[:3, 3], np.empty((0,3), dtype= np.float32)

    hom_pixels = np.hstack([uv_s, np.ones((n_pixel, 1))])
    cam_rays = hom_pixels @ np.linalg.inv(intrinsics).T
    base_rays =  cam_rays @ base_t_cam[:3, :3].T
    base_rays_norm = base_rays / np.linalg.norm(base_rays, axis=-1, keepdims=True)

    return base_t_cam[:3, 3], base_rays_norm


def ray_pointcloud_intersection(points:np.ndarray, origin:np.ndarray, directions:np.ndarray, max_dist:float = 0.001)->np.ndarray:
    """
    :param points: Nx3 point-cloud
    :param origin: [x y z] origin array
    :param directions: Mx3 normalized direction array (with origin at parameter origin)
    :param max_dist:
    :return: Nx3 array of the intersecting points (np.nan for failed ones)
    """

    assert points.ndim == 2 and points.shape[-1] == 3, f"Wrong Point shape: {points.shape} != (Nx3)"
    assert directions.ndim == 2 and directions.shape[-1] == 3, f"Wrong directions shape: {directions.shape} != (Mx3)"

    if directions.shape[0] < 1:
        return np.empty((0,3), dtype=np.float32)
    from ..utilities.time_tracker import TimeTracker
    tt = TimeTracker()

    projection_points = []

    v = points - origin

    tt.add_time_stamp("Creating v")

    for direction in directions:
        t = np.dot(v, direction)
        points_on_line = origin + t[:, None] * direction
        distances = np.sum((points_on_line-points)**2, axis=-1)

        tt.add_time_stamp("Distances")

        valid_mask = (t > 1e-6) & (distances < max_dist**2)

        if not np.any(valid_mask):
            projection_points.append(np.full((3), np.nan))
            continue
        
        tt.add_time_stamp("Valid mask")

        best_proj_point_idx = np.argmin(distances[valid_mask])
        projection_points.append(points_on_line[valid_mask][best_proj_point_idx])

        tt.add_time_stamp("Finalisation")     

    tt.print_report()
    return np.asarray(projection_points)



def calculate_gripping_difference_4_pixels(
        base_t_cam_green:np.ndarray, 
        base_t_cam_red:np.ndarray, 
        points:np.ndarray, 
        pixels:np.ndarray, 
        intrinsics:np.ndarray,
        distance_type:Literal['avg', 'median'] = 'median',
        valid_distance:float = 0.001,
        visualize:bool = False,
        voxel_downsample : float | None = None
    )->float | None:
    """
    :param base_t_cam_green: A 4x4 homogeneout base-T_cam transformation matrix
    :param base_t_cam_red: Another 4x4 homogeneout base-T_cam transformation matrix
    :param points: Nx3 point-cloud
    :param pixels: Nx2 Pixel coordinates
    :param intrinsics: A 3x3 intrinsic camera matrix
    :param distance_type: Whether to return the median of the distances for the pixels or the average
    :param valid_distance: How close a point has to be to a line to be considered an intersection
    :param visualize: If true will visualize the error in 3d
    :param voxel_downsample: The voxel size to downsample to for speedup
    :return: The computed error or None if not possible
    """
    assert assert_homogeneous_mat(base_t_cam_green, size=4)
    assert assert_homogeneous_mat(base_t_cam_red, size=4)
    assert points.ndim == 2 and points.shape[-1] == 3, f"Wrong Point shape: {points.shape} != [Bx3]"
    assert pixels.ndim == 2 and pixels.shape[-1] == 2, f"wrong pixels shape: {pixels.shape} != [Bx2]"
    assert assert_intrinsic_mat(intrinsics)
    assert distance_type in ['avg', 'median'], f"Invalid distance type: {distance_type}"
    assert valid_distance > 0, f"distance must be positive: {valid_distance}"

    points = points[np.all(np.isfinite(points), axis=-1)]

    if voxel_downsample is not None:
        pc_o3d = o3d.geometry.PointCloud()
        pc_o3d.points = o3d.utility.Vector3dVector(points)
        pc_o3d = pc_o3d.voxel_down_sample(voxel_size=voxel_downsample)
        points = np.asarray(pc_o3d.points)

    origin1, directions1 = pixels_to_rays(intrinsics=intrinsics, base_t_cam=base_t_cam_green, uv_s=pixels)
    origin2, directions2 = pixels_to_rays(intrinsics=intrinsics, base_t_cam=base_t_cam_red, uv_s=pixels)

    gripping_points_1 = ray_pointcloud_intersection(points=points, origin=origin1, directions=directions1, max_dist=valid_distance)
    gripping_points_2 = ray_pointcloud_intersection(points=points, origin=origin2, directions=directions2, max_dist=valid_distance)

    finite_mask = np.all(np.isfinite(gripping_points_1), axis=-1) & np.all(np.isfinite(gripping_points_2), axis=-1)
    distances = np.linalg.norm(gripping_points_1[finite_mask]-gripping_points_2[finite_mask], axis=-1)


    return_dist = None
    if distance_type == 'avg' and distances.shape[0] > 0:
        return_dist = np.mean(distances)
    elif distance_type == 'median' and distances.shape[0] > 0:
        return_dist = np.median(distances)


    if visualize:
        base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([0.2,0.2,0.2])

        to_vis = [base_frame, pcd]


        for col, points, b_t_c in zip([[0, 1, 0], [1, 0, 0]], [gripping_points_1, gripping_points_2], [base_t_cam_green, base_t_cam_red]):
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(points)
            pc.paint_uniform_color(col)
            to_vis.append(pc)


        o3d.visualization.draw_geometries(to_vis, f"Gripping distance vis: {return_dist}")

    return return_dist


def sample_pixel_neighborhood(center:tuple[int, int], size:int = 5)->np.ndarray:
    """
    :param center: cx, cy pixel coordinates
    :param size: Odd size of the patch to be sampled
    :return: (size^2)x2 pixel array
    """
    assert size % 2 == 1, f"Size must be odd: {size}"

    half = size//2
    cx, cy = center
    
    x_range = np.arange(cx - half, cx + half + 1)
    y_range = np.arange(cy - half, cy + half + 1)

    xx, yy = np.meshgrid(x_range, y_range)
    pixels = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    
    return pixels


class FastGrippingError:
    def __init__(
            self, 
            points:np.ndarray, 
            intrinsics:np.ndarray, 
            visualize:bool = False,
            voxel_downsample:float | None = 0.005,
            alpha_mesh_generation:float = 0.01,
            sparse_regions_removal_nb_neighbors:int = 20,
            sparse_regions_removal_std_ratio:float = 2.0
        ) -> None:
        """
        :param points: (N_0x...xN_n)x3 point-cloud
        """

        points = points.reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[np.isfinite(points).all(axis=-1)])

        if voxel_downsample is not None:
            pcd = pcd.voxel_down_sample(voxel_size=voxel_downsample)

        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=sparse_regions_removal_nb_neighbors,
            std_ratio=sparse_regions_removal_std_ratio
        )

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd,
            alpha=alpha_mesh_generation
        )

        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(tmesh)

        self.intrinsics = intrinsics
        
        self.visualize = visualize
        self.points = None
        self.mesh = None
        if self.visualize:
            self.points = points
            self.mesh = mesh

    
    def _pixels_to_hitpoints(
        self,
        base_t_cam: np.ndarray,
        pixels: np.ndarray,
    ) -> np.ndarray:

        origin, directions = pixels_to_rays(
            intrinsics=self.intrinsics,
            base_t_cam=base_t_cam,
            uv_s=pixels
        )

        n = directions.shape[0]

        rays = np.concatenate([
            np.repeat(origin[None], n, axis=0),
            directions
        ], axis=1).astype(np.float32)

        t_hits = self.scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()

        hit_points = np.full((n, 3), np.nan, dtype=np.float32)

        valid = np.isfinite(t_hits)

        hit_points[valid] = (origin[None]+ t_hits[valid, None] * directions[valid])

        return hit_points


    def calculate_gripping_differences_4_pixels(
        self,
        base_t_cam_s:np.ndarray,
        pixels_batch:np.ndarray, 
        distance_type:Literal['avg', 'median'] = 'median',
    )-> np.ndarray:
        """
        :param base_t_cam_s: A Bx4x4 homogeneout base-T_cam transformation matrix batch
        :param base_t_cam_red: Another 4x4 homogeneout base-T_cam transformation matrix
        :param pixels_batch: BxNx2 Pixel coordinates
        :param intrinsics: A 3x3 intrinsic camera matrix
        :param distance_type: Whether to return the median of the distances for the pixels or the average
        :param visualize: If true will visualize the error in 3d
        :return: The computed BxB error matrix (may contain nan)
        """
        b = base_t_cam_s.shape[0]
        hit_points = np.asarray([self._pixels_to_hitpoints(b_t_c, pixels) for b_t_c, pixels in zip(base_t_cam_s, pixels_batch)]) #[B, N, 3]
        
        
        errors = np.full((b,b), np.nan)

        for i in range(b):
            for j in range(i + 1, b):

                p1 = hit_points[i]
                p2 = hit_points[j]

                valid = np.all(np.isfinite(p1), axis=1) & np.all(np.isfinite(p2), axis=1)

                if not np.any(valid):
                    continue

                d = np.linalg.norm(p1[valid] - p2[valid],axis=-1)

                if distance_type == "avg":
                    err = np.mean(d)
                else:
                    err = np.median(d)

                errors[i, j] = err
                errors[j, i] = err


        
        if self.visualize:
            base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.points)
            pcd.paint_uniform_color([0.5,0.0,0.5])

            to_vis = [base_frame, pcd, self.mesh]
            cmap = plt.get_cmap("jet")

            for i in range(b):
                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(hit_points[i])
                pc.paint_uniform_color(color = cmap(i / max(b - 1, 1))[:3])
                to_vis.append(pc)

                cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                cam_frame.transform(base_t_cam_s[i])
                to_vis.append(cam_frame)

            o3d.visualization.draw_geometries(
                to_vis, 
                window_name=f"Gripping distance vis: {np.round(errors*1000,1)} mm",
                mesh_show_wireframe=True
            )

        return errors