import numpy as np
import torch
from skimage.draw import line
from dataclasses import dataclass
from numbers import Number

from .union_find import UnionFind


torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def remove_short_2d_line_segments(line_segs_2d:np.ndarray, min_line_length_px:float = 5) -> np.ndarray:
    """
    Removes all line_segments with euclidian lengths under the min_line_length threshhold
    :param line_segs_2d: Nx4 numpy array of the structure: [[x1, y1, x2, y2], ...]
    :param min_line_length_px, the threshhold (in pixels)
    :return Mx4 numpy array of the same structure with M <= N
    """
    assert line_segs_2d.ndim == 2 and line_segs_2d.shape[-1] == 4
    assert 0 <= min_line_length_px, f"invalid negative length threshold: {min_line_length_px}"

    lengths = (line_segs_2d[:,2]-line_segs_2d[:, 0])**2+(line_segs_2d[:,3]-line_segs_2d[:,1])**2
    return line_segs_2d[lengths > (min_line_length_px**2)]


def line_segment_to_points_distances_2d(line_seg_2d:np.ndarray, points_2d:np.ndarray) -> np.ndarray:
    """
    Computes the distance between each point and the line segment
    :param line_seg_2d: length 4 numpy array of the structure: [x1, y1, x2, y2]
    :param points_2d: Nx2 numpy array of the points [[xi, yi], ...]
    :return numpy array of length N with the distances
    """
    assert points_2d.ndim == 2 and points_2d.shape[-1] == 2, f"invalid shape: {line_seg_2d.shape} != (N,2)"
    assert line_seg_2d.shape == (4,), f"invalid shape: {points_2d.shape} != (4,)"

    x1, y1, x2, y2 = line_seg_2d
    dx, dy = x2-x1, y2-y1

    seg_length_sq = dx*dx + dy*dy
    if seg_length_sq < 1e-6:
        return np.linalg.norm(points_2d-np.array([x1, y1]), axis = 1) 

    t = np.clip(((points_2d[:, 0]-x1) * dx + (points_2d[:, 1]-y1) * dy)/ seg_length_sq, 0.0, 1.0)
        
    projections = np.column_stack([x1 + t * dx, y1 + t * dy])        
    return np.linalg.norm(projections-points_2d, axis = 1)


def line_to_points_distances_2d(line_seg_2d:np.ndarray, points:np.ndarray, signed:bool = False) -> np.ndarray:
    """
    Computes the absolute distance euclidian between each point and the infinite line spanned by the 2 line points
    :param line_seg_2d: Numpy array of the structure: [x1, y1, x2, y2]
    :param points: Nx2 numpy array of the points [[xi, yi], ...]
    :param signed: if true will return the signed distances else the absolute distances
    :return array of length N of the distances
    """
    assert line_seg_2d.shape == (4,)
    assert points.ndim == 2 and points.shape[-1] == 2

    x1, y1, x2, y2 = line_seg_2d
    dx, dy = x2-x1, y2-y1
    line_points_dist = np.hypot(dx,dy)
    c = x2*y1-y2*x1
    if signed:
        return (points[:, 0] * dy - points[:, 1]*dx + c)/line_points_dist
    return np.abs((points[:, 0] * dy - points[:, 1]*dx + c)/line_points_dist)


def lines_to_points_distances_2d(line_segs_2d:np.ndarray, points:np.ndarray) -> np.ndarray:
    """
    Computes the distance between each point and the infinite line created by the line segment points
    :param line_seg_2d: Nx4 line segment array of the structure: [[x1, y1, x2, y2], ...]
    :param points: Mx2 array of the points [[x1, y1], ...]
    :return Array of size NxM with the absolute euclidian distances
    """
    assert line_segs_2d.ndim == 2 and line_segs_2d.shape[-1] == 4, f"invalid shape: {line_segs_2d.shape} != (N,4)"
    assert points.ndim == 2 and points.shape[-1] == 2, f"invalid shape: {points.shape} != (M,2)"

    x1 = line_segs_2d[:, 0]
    y1 = line_segs_2d[:, 1]
    x2 = line_segs_2d[:, 2]
    y2 = line_segs_2d[:, 3]
    dx, dy = x2-x1, y2-y1
    line_points_dist = np.hypot(dx, dy)
    c = x2*y1-y2*x1

    px = points[:, 0]
    py = points[: , 1]

    numerator = np.abs(
        px[None, :] * dy[:,None]- py[None, :]*dx[:, None] + c[:, None]
    )
    return numerator/line_points_dist


def line_seg_2d_to_3d_points(line_seg_2d:np.ndarray, xyz_image:np.ndarray)-> np.ndarray:
    """
    Creates a 3d point cloud of the points the line segment passes through on the xyz_image
    :param line_seg_2d: [x1, y1, x2, y2] numpy array
    :param xyz_image: HxWx3 numpy array that has a 3d point at each pixel
    :return Nx3 point array
    """
    assert line_seg_2d.shape == (4,), f"2D line seg shape: {line_seg_2d.shape}"
    assert xyz_image.ndim == 3 and xyz_image.shape[-1] == 3

    x1, y1, x2, y2 = np.round(line_seg_2d).astype(int)

    row_cords, col_cords = line(y1, x1, y2, x2)
    valid_points_mask = (0 <= row_cords) & (row_cords < xyz_image.shape[0]) & (0 <= col_cords) & (col_cords < xyz_image.shape[1])

    points_3d = xyz_image[row_cords[valid_points_mask], col_cords[valid_points_mask]]
    points_3d = points_3d[~np.isnan(points_3d).any(axis=-1)]
    return np.empty((0,3)) if len(row_cords) < 1 else points_3d


def merge_line_seg_cluster_into_one(line_segs_2d:np.ndarray)->np.ndarray:
    """
    Joins multiple 2d line segments into one by doint pca regression through all
    :param line_segs_2d: Nx4 array of the structure: [[x1, y1, x2, y2], ...] (with N > 0)
    :return a numpy array: [x1, y1, x2, y2] of the new line
    """
    assert line_segs_2d.ndim == 2 and line_segs_2d.shape[-1] == 4, f"invalid shape: {line_segs_2d.shape} != (N,4)"
    assert line_segs_2d.shape[0] > 0, f"Can join {line_segs_2d.shape[0]}<1 segments"

    if line_segs_2d.shape[0] == 1:
        return line_segs_2d[0]
    
    return pca_2d_3d_points_lineseg_regression(line_segs_2d.reshape(-1,2))


def merge_line_seg_cluster_into_one_weighted(line_segs_2d:np.ndarray)->np.ndarray:
    """
    Joins multiple 2d line segments into one
    :param line_segs_2d: Nx4 array of the structure: [[x1, y1, x2, y2], ...] (with N > 0)
    :return a numpy array: [x1, y1, x2, y2] of the new line
    """
    assert line_segs_2d.ndim == 2 and line_segs_2d.shape[-1] == 4, f"invalid shape: {line_segs_2d.shape} != (N,4)"
    assert line_segs_2d.shape[0] > 0, f"Can join {line_segs_2d.shape[0]}<1 segments"

    if line_segs_2d.shape[0] == 1:
        return line_segs_2d[0]
    
    dxdy = line_segs_2d[:, 2:] - line_segs_2d[:, :2]

    line_segment_lengths = np.linalg.norm(dxdy, axis = 1)

    line_segment_relevances = line_segment_lengths/line_segment_lengths.sum()

    weighted_direction = np.mean(dxdy[:, :]/line_segment_lengths[:, None] * line_segment_relevances[:, None], axis = 0)
    weighted_direction = weighted_direction/np.linalg.norm(weighted_direction)

    weighted_center = np.sum((line_segs_2d[:, :2]+dxdy/2)*line_segment_relevances[:, None], axis = 0)

    centered_points = line_segs_2d.reshape(-1,2)-weighted_center
    dists_to_center = centered_points @ weighted_direction

    start_p = weighted_center + dists_to_center.min()*weighted_direction
    end_p = weighted_center + dists_to_center.max()*weighted_direction

    return np.concatenate([start_p, end_p])


def pca_2d_3d_points_lineseg_regression(points:np.ndarray):
    """
    Calculates a 2d/3d line segment fittet through the points using PCA
    :param points: Nx2/3 array of the point to regress through
    :return [x1, x2, y1, y2] or [x1, x2, x3, y1, y2, y4]
    """
    assert points.ndim == 2, f"point shape: {points.shape} != (N,2/3)"
    assert points.shape[-1] == 2 or points.shape[-1] == 3, f"points neither 2/3d: {points.shape[-1]}"

    xy_center = np.mean(points, axis = 0)
        
    centered = points-xy_center
    cov = centered.T @ centered

    _, eigvecs = np.linalg.eigh(cov)

    direction = eigvecs[:, -1]

    dists_to_center = centered @ direction

    start_p = xy_center + dists_to_center.min()*direction
    end_p = xy_center + dists_to_center.max()*direction

    return np.concatenate([start_p, end_p])

def robust_pca_2d_3d_points_lineseg_regression(points:np.ndarray):
    """
    Calculates a 2d/3d line segment fittet through the points using PCA
    Will use median for the center and 0.05 & 0.95 quantiles for endpoints
    Will remove 0.15 quantile of points that are the farthest from the median
    :param points: Nx2/3 array of the point to regress through
    :return [x1, x2, y1, y2] or [x1, x2, x3, y1, y2, y4]
    """
    assert points.ndim == 2, f"point shape: {points.shape} != (N,2/3)"
    assert points.shape[-1] == 2 or points.shape[-1] == 3, f"points neither 2/3d: {points.shape[-1]}"

    xy_center = np.median(points, axis = 0)
        
    centered = points-xy_center

    d = np.linalg.norm(centered, axis=1)
    centered = centered[d < np.percentile(d, 85)]
    cov = centered.T @ centered

    _, eigvecs = np.linalg.eigh(cov)

    direction = eigvecs[:, -1]

    dists_to_center = centered @ direction

    start_p = xy_center + np.percentile(dists_to_center, 5)*direction
    end_p = xy_center + np.percentile(dists_to_center, 95)*direction

    return np.concatenate([start_p, end_p])


def line_segment_regression_3d_ransaac(xyz_points:np.ndarray|torch.Tensor, inlier_distance:float, itterations:int)->np.ndarray|None:
    """
    :param xyz_points: An Nx3 array of 3d points (will be cast to torch.Tensor if not already)
    :param inlier_distance: distance to line to be considered an inlier
    :param itterations: The number of itteration for ransaac
    :return an numpy array: [x1, y1, z1, x2, y2, z2] that represents the fitted line segment or None if it couldnt be fitted
    """
    if not hasattr(line_segment_regression_3d_ransaac, "_line_fit"):
        from torch_ransac3d.line import line_fit
        line_segment_regression_3d_ransaac._line_fit_3d = line_fit

    assert xyz_points.ndim == 2 and xyz_points.shape[-1] == 3, f"xyz_points shape: {xyz_points.shape}"
    assert 0 <= inlier_distance, f"Inlier distance must be positive: {inlier_distance} < 0 "
    assert 0 < itterations, f"Number of iterations must be greater then 0 {itterations}" 

    if xyz_points.shape[0] < 2:
        return None
    
    if not torch.is_tensor(xyz_points):
        xyz_points = torch.tensor(xyz_points, dtype = torch.float32)

    infinite_line = line_segment_regression_3d_ransaac._line_fit_3d(
        pts = xyz_points,
        thresh = inlier_distance,
        max_iterations = itterations
    )

    inliers = xyz_points[infinite_line.inliers]
    direction = (infinite_line.direction / torch.linalg.norm(infinite_line.direction))
    p0 = infinite_line.point

    dists_to_p0 = torch.matmul(inliers-p0, direction)

    start_p = p0 + torch.min(dists_to_p0) * direction
    end_p = p0 + torch.max(dists_to_p0) * direction

    return torch.cat([start_p, end_p]).cpu().numpy()

def project_point_onto_line_slow(px, py, x1, y1, x2, y2):
    """
    Project point (px, py) onto line defined by (x1,y1)-(x2,y2).
    :returns the closest point on the infinite line.
    """
    dx = x2 - x1
    dy = y2 - y1
    t = ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)
    return x1 + t * dx, y1 + t * dy


def merge_close_line_segments(
        line_segs_2d:np.ndarray,
        max_angle_diff:float = 5,
        max_midpoint_dist:float = 5,
        max_endpoint_dist:float = 15,
        quick_join:bool = False
    ) -> np.ndarray:
    """
    Clusters lines and merges each cluster.
    To join one cluster a line has to have all of the following attributes:
    1. The angle between the line segments has to be smaller then max_angle_diff
    2. The distance between the midpoint of the line and the infinite line of another segment has to be less then the max_midpoint_dist
    3. The line has to overlap or has to have a smaller enpoint-endpoint distance between 2 lines then max_endpoint_dist

    :param line_segs_2d: Nx4 array of the structure: [[x1, y1, x2, y2], ...]
    :param max_angle_diff: maximum angle between 2 lines to be joined (in degrees)
    :param max_midpoint_dist: maximum angle between 2 midpoints, higher causes less colinear lines to be joined
    :param max_endpoint_dist: maximum distance between 2 line-endpoints to be joined, higher causes lines with more distance to be joined
    :param quick_join: if false the line clusters will be segmented further, can help when chaining is a problem (takes ~1/4 longer)
    :return Mx4 array of the same structure with M <= N
    """
    assert line_segs_2d.ndim == 2 and line_segs_2d.shape[-1] == 4, f"invalid shape: {line_segs_2d.shape} != (N,4)"
    assert 0 <= max_angle_diff <= 360, f"invalid angle: {max_angle_diff}°"
    assert 0 <= max_midpoint_dist, f"max midpoint dist must be non-negative: {max_midpoint_dist}"
    assert 0 <= max_endpoint_dist, f"max endpoint dist must be non-negative: {max_endpoint_dist}"

    n_lines = line_segs_2d.shape[0]

    angle_thresh_rad = np.deg2rad(max_angle_diff)
    line_segs_2d = line_segs_2d.copy()

    dxy_s = line_segs_2d[:,2:]-line_segs_2d[:,0:2]
    angle_s = np.atan2(dxy_s[:, 1], dxy_s[:,0])
    mid_points = line_segs_2d[:, 0:2] + dxy_s[:]/2

    # Sort endpoints
    mostly_horizontal = np.abs(dxy_s[:, 0]) > np.abs(dxy_s[:, 1])
    flip_endpoints = np.logical_or(
        mostly_horizontal & (line_segs_2d[:, 0] > line_segs_2d[:, 2]),
        ~mostly_horizontal & (line_segs_2d[:, 1] > line_segs_2d[:, 3])
    )
    line_segs_2d[flip_endpoints] = line_segs_2d[flip_endpoints][:, [2,3,0,1]]

    # Line segments with small angle delta
    angle_diff_matrix = angle_s[:, None] - angle_s[None, :]
    np.abs(angle_diff_matrix, out = angle_diff_matrix)
    angle_diff_matrix = np.minimum(angle_diff_matrix, np.pi - angle_diff_matrix)
    np.abs(angle_diff_matrix, out = angle_diff_matrix)
    angle_diff_mask = angle_diff_matrix < angle_thresh_rad


    midpoint_dist_matrix_mask = lines_to_points_distances_2d(
        line_segs_2d=line_segs_2d,
        points=mid_points
    ) < max_midpoint_dist

    # End points
    ep1 = line_segs_2d[:, :2]
    ep2 = line_segs_2d[:, 2:]

    # Mask for distances between endpoints
    gap_sq_matrix_mask = np.sum(
        (ep1[None, :, :] - ep2[:, None, :])**2,
        axis = -1
    ) < max_endpoint_dist ** 2

    merge_adj_list = []
    merge_adj_matrix = np.zeros((n_lines, n_lines), dtype = bool)
    for i1,l1 in enumerate(line_segs_2d):
        poss_indices = np.arange(i1+1, line_segs_2d.shape[0])
        mask = angle_diff_mask[i1, i1+1:]

        midpoint_mask = midpoint_dist_matrix_mask[i1, poss_indices] | midpoint_dist_matrix_mask[poss_indices, i1]

        ## Extract endpoints
        l1_ep1, l1_ep2 = l1[:2], l1[2:]
        l_other_ep1 = ep1[poss_indices]
        l_other_ep2 = ep2[poss_indices]

        # Check for overlap
        if mostly_horizontal[i1]:
            overlaps = (l1_ep1[0] <= l_other_ep2[:,0]) & (l_other_ep1[:,0] <= l1_ep2[0])
        else:
            overlaps = (l1_ep1[1] <= l_other_ep2[:,1]) & (l_other_ep1[:,1] <= l1_ep2[1])

        combined_mask = mask & midpoint_mask & (overlaps | gap_sq_matrix_mask[i1, poss_indices])
        merge_adj_list.append(poss_indices[combined_mask])
        merge_adj_matrix[i1, i1+1:] = combined_mask[:]

    # Combine all similar Lines
    union_find = UnionFind(line_segs_2d.shape[0])
    for i1, adjecent_idx in enumerate(merge_adj_list):
        for i2 in adjecent_idx:
            union_find.union(i1, i2)
    
    or_clusters =  union_find.return_clusters()

    if quick_join:
        return np.array([merge_line_seg_cluster_into_one_weighted(line_segs_2d[np.array(line_cluster)])
            for line_cluster in or_clusters
        ])
    

    line_lengths = np.linalg.norm(dxy_s, axis = 1)
    adj_matrix = merge_adj_matrix | merge_adj_matrix.T

    line_unused = np.ones(line_segs_2d.shape[0], dtype = bool)
    cluster_representative = np.arange(0, n_lines)


    for cluster in or_clusters:
        cluster = np.array(cluster)
        line_unused = np.ones(cluster.shape[0], dtype = bool)

        while np.any(line_unused):
            unused_lines = cluster[line_unused]
            line_seed = unused_lines[np.argmax(line_lengths[unused_lines])]
            to_add_mask = adj_matrix[line_seed, cluster] & line_unused

            seed_pos = np.where(cluster == line_seed)[0][0]
            to_add_mask[seed_pos] = True


            new_lines = cluster[to_add_mask]
            cluster_representative[new_lines] = cluster_representative[line_seed]
            line_unused &=  ~ to_add_mask
    
    clusters = {}
    for idx in range(n_lines):
        representative = cluster_representative[idx]
        if representative in clusters:
            clusters[representative].append(idx)
        else:
            clusters[representative] = [idx]

    return np.array([merge_line_seg_cluster_into_one_weighted(line_segs_2d[np.array(line_cluster)])
        for line_cluster in list(clusters.values())
    ])



@dataclass(frozen=True, kw_only=True)
class LineMatchingConfig:
    """
    Lines are matched based on common nearby matched points, this allows to tune that
    :param max_point_line_dist_px: Maximum distance between a point and a line to be considered near
    :param min_number_supporting_points: The number of same points that have to be near line1 and line2, to match them
    """
    max_point_line_dist_px:float = 5
    min_number_supporting_points:int = 2

    def __post_init__(self):
        assert isinstance(self.max_point_line_dist_px, Number) and 0 <= self.max_point_line_dist_px
        assert isinstance(self.min_number_supporting_points, Number) and 0 < self.min_number_supporting_points


def match_2d_line_segments(
        lines_img1:np.ndarray, 
        lines_img2:np.ndarray,
        points_img1:np.ndarray, 
        points_img2:np.ndarray,
        line_matching_config:LineMatchingConfig = LineMatchingConfig()
    )->np.ndarray:
    """
    Returns the best matching line segment pairs (only matches one to one)
    :param lines_img1: Lx4 numpy array of the structure: (x1, y1, x2, y2), of lines in image 1
    :param lines_img2: Mx4 numpy array of the structure: (x1, y1, x2, y2), of lines in image 2
    :param points_img1: Nx2 numpy array of points in image 1, p_i in points_img1 has to be matched to p_i in points_img2
    :param points_img2: Ox2 numpy array of points in image 2
    :return Px2x4 numpy array of P linepairs
    """
    assert lines_img1.ndim == 2 and lines_img1.shape[-1] == 4
    assert lines_img2.ndim == 2 and lines_img2.shape[-1] == 4
    assert points_img1.ndim == 2 and points_img1.shape[-1] == 2
    assert points_img2.ndim == 2 and points_img2.shape[-1] == 2

    if lines_img1.shape[0] == 0 or lines_img2.shape[0] == 0 or points_img1.shape[0] == 0 or points_img2.shape[0] == 0:
        return np.empty((0,2,4))
    
    lines_1_point_distances_mask = np.array([line_segment_to_points_distances_2d(line_seg_2d=l1, points_2d=points_img1) < line_matching_config.max_point_line_dist_px for l1 in lines_img1])
    lines_2_point_distances_mask = np.array([line_segment_to_points_distances_2d(line_seg_2d=l2, points_2d=points_img2) < line_matching_config.max_point_line_dist_px for l2 in lines_img2])
    agreement_matrix = np.sum(lines_1_point_distances_mask[:, None, :] & lines_2_point_distances_mask[None, :, :], axis=2)

    best_l2_matching_values = np.full(lines_img2.shape[0], -1)
    best_l2_matchings = np.full(lines_img2.shape[0], -1)

    for i, l1 in enumerate(lines_img1):
        best_l2_index = np.argmax(agreement_matrix[i])
        if agreement_matrix[i,best_l2_index] < line_matching_config.min_number_supporting_points:
            continue
        if best_l2_matching_values[best_l2_index] <= agreement_matrix[i, best_l2_index]:
                best_l2_matching_values[best_l2_index] = agreement_matrix[i, best_l2_index]
                best_l2_matchings[best_l2_index] = i
        
    line_pairs = []
    for l2_idx, l1_idx in enumerate(best_l2_matchings): 
        if l1_idx >= 0:
            line_pairs.append([lines_img1[l1_idx], lines_img2[l2_idx]])
    return np.array(line_pairs) if len(line_pairs) > 0 else np.empty((0,2,4))