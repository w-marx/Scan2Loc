import numpy as np
from dataclasses import dataclass

from .packed_bool_mask_storage import ImageMaskStorage

from ..utilities.union_find import UnionFind


@dataclass(kw_only=True, frozen=True)
class PointCloudMatchingConfig:
    max_centroid_dist:float | None = 0.05
    max_color_dist:float | None = 500
    min_cluster_size:int = 1
    min_number_points_per_detected_object:int = 1000



def match_point_clouds(
        masks:ImageMaskStorage, 
        xyz_images:np.ndarray, 
        mask_image_indices:list[int],
        point_cloud_matching_config:PointCloudMatchingConfig,
        bgr_images:np.ndarray | None = None
    )->list[list[int]]:

    n = 0
    object_avg_colors = []
    object_avg_centers = []
    index_map = []

    for pc_idx, imgidx in enumerate(mask_image_indices):
        mask = masks.see_mask(pc_idx)

        pc_3d = xyz_images[imgidx][mask > 0]
        pc_3d = pc_3d[np.isfinite(pc_3d).all(axis=-1)]

        if pc_3d.shape[0] < point_cloud_matching_config.min_number_points_per_detected_object:
            continue

        n += 1
        index_map.append(pc_idx)

        if point_cloud_matching_config.max_centroid_dist is not None:
            object_avg_centers.append(np.mean(pc_3d, axis = 0))

        if point_cloud_matching_config.max_color_dist is not None and bgr_images is not None:
            pc_color = bgr_images[imgidx][mask > 0]
            object_avg_colors.append(np.mean(pc_color, axis = 0))

    if len(object_avg_colors) < 1 or len(object_avg_centers) < 1:
        return [[]]

    valid_mask = np.ones((n, n), dtype=bool)

    if point_cloud_matching_config.max_centroid_dist is not None:
        object_avg_centers = np.stack(object_avg_centers, axis = 0)

        centroid_distance_matrix = np.linalg.norm(
            object_avg_centers[:, None] - object_avg_centers[None, :], axis=-1
        )
        valid_mask &= (centroid_distance_matrix < point_cloud_matching_config.max_centroid_dist)

    if point_cloud_matching_config.max_color_dist is not None:
        object_avg_colors = np.stack(object_avg_colors, axis = 0)

        color_distance_matrix = np.mean(
            np.abs(
                object_avg_colors[:, None] -
                object_avg_colors[None, :]
            ),
            axis=-1
        )
        valid_mask &= (color_distance_matrix < point_cloud_matching_config.max_color_dist)

    union_find = UnionFind(n)
    for row_idx in range(n):
        for col_idx in range(row_idx+1, n):
            if valid_mask[row_idx, col_idx]:
                union_find.union(row_idx, col_idx)

    clusters = union_find.return_clusters()
    clusters = [c for c in clusters if len(c) >= point_cloud_matching_config.min_cluster_size]

    clusters_old_indices = [[index_map[i] for i in c] for c in clusters]
    return clusters_old_indices