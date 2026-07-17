import numpy as np
import torch
from skimage.draw import line
from dataclasses import dataclass
from numbers import Number


torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def assert_nd_line_batch(lines:np.ndarray, dim:int = 2):
    """
    Asserts that something is a line batch:
    [[x0, y0, ...], ... ]
    """
    assert lines.ndim == 2 and lines.shape[-1] == dim, f"Shape: {lines.shape} != (B, {dim})"



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
    assert line_seg_2d.shape == (4,), f"Not 2d line segment: {line_seg_2d}"
    assert points.ndim == 2 and points.shape[-1] == 2, f"Not 2d points: {points.shape}"

    x1, y1, x2, y2 = line_seg_2d
    dx, dy = x2-x1, y2-y1
    line_points_dist = np.hypot(dx,dy)
    c = x2*y1-y2*x1
    if signed:
        return (points[:, 0] * dy - points[:, 1]*dx + c)/line_points_dist
    return np.abs((points[:, 0] * dy - points[:, 1]*dx + c)/line_points_dist)


def line_to_points_distances_3d(line_point:np.ndarray, line_direction:np.ndarray, points:np.ndarray) -> np.ndarray:
    """
    Computes the absolute distance euclidian between each point and the infinite line spanned by the 2 line points
    :param line_point: A 3d point on the line
    :param line_direction: The 3d line direction
    :param points: Nx3 numpy array of the points [[xi, yi, zi], ...]
    :return array of length N of the distances
    """
    assert line_point.shape == (3,) and line_direction.shape == (3,), f"Not 3d line: {line_point} + t * {line_direction}"
    assert points.ndim == 2 and points.shape[-1] == 3, f"Not 3d points: {points.shape}"
    assert np.linalg.norm(line_direction) > 1e-8, f"Direction not long enough: {line_direction}"

    line_dir = line_direction/np.linalg.norm(line_direction)
    t_s = np.dot(points-line_point, line_dir)

    projection_points = line_point + t_s[:, np.newaxis] * line_dir
    return np.linalg.norm(points - projection_points, axis=-1)


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

def robust_pca_2d_3d_points_lineseg_regression(
        points:np.ndarray, 
        line_distance_quantile:float = 0.95,
        itterations:int = 3
    ):
    """
    Calculates a 2d/3d line segment fittet through the points using PCA
    Will use median for the center and remove points far from an infinite line fit
    Will remove 0.15 quantile of points that are the farthest from the median
    :param points: Nx2/3 array of the point to regress through
    :param line_distance_quantile: The quantile of points to use for fitting based on their distance to the infinite line produced when fitting all
    :param itterations: how often to remove outliers before the final fit
    :return [x1, x2, y1, y2] or [x1, x2, x3, y1, y2, y4]
    """
    assert points.ndim == 2, f"point shape: {points.shape} != (N,2/3)"
    assert points.shape[-1] == 2 or points.shape[-1] == 3, f"points neither 2/3d: {points.shape[-1]}"

    # Initial guess

    fpoints = points.copy()
    line_seg = None


    for _ in range(itterations):
        line_seg = pca_2d_3d_points_lineseg_regression(fpoints)
        distances = line_to_points_distances_3d(
            line_point=line_seg[:3],
            line_direction=line_seg[3:],
            points=fpoints,
        )
        distance_outlier_cutoff = np.quantile(distances, line_distance_quantile)
        fpoints = fpoints[distances < distance_outlier_cutoff]
    return line_seg


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



@dataclass(frozen=True, kw_only=True)
class LineMatchingConfig:
    """
    Lines are matched based on common nearby matched points, this allows to tune that
    :param max_point_line_dist_px: Maximum distance between a point and a line to be considered near
    :param min_number_supporting_points: The number of same points that have to be near line1 and line2, to match them
    :param better_factor: A line pair has to be at least better_factor better then the second best line pair
    """
    max_point_line_dist_px:float = 5
    min_number_supporting_points:int = 2
    better_factor:float = 1.1

    def __post_init__(self):
        assert isinstance(self.max_point_line_dist_px, Number) and 0 <= self.max_point_line_dist_px
        assert isinstance(self.min_number_supporting_points, Number) and 0 < self.min_number_supporting_points


def match_2d_line_segments(
        lines_img1:np.ndarray, 
        lines_img2:np.ndarray,
        points_img1:np.ndarray, 
        points_img2:np.ndarray,
        line_matching_config:LineMatchingConfig = LineMatchingConfig(),
        return_indices:bool = False
    )->np.ndarray | list[tuple[int, int]]:
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
        if best_l2_matching_values[best_l2_index] * line_matching_config.better_factor <= agreement_matrix[i, best_l2_index]:
                best_l2_matching_values[best_l2_index] = agreement_matrix[i, best_l2_index]
                best_l2_matchings[best_l2_index] = i
    
    if return_indices:
        line_pairs = []
        for l2_idx, l1_idx in enumerate(best_l2_matchings): 
            if l1_idx >= 0:
                line_pairs.append((l1_idx, l2_idx))
        return line_pairs


    line_pairs = []
    for l2_idx, l1_idx in enumerate(best_l2_matchings): 
        if l1_idx >= 0:
            line_pairs.append([lines_img1[l1_idx], lines_img2[l2_idx]])
    return np.array(line_pairs) if len(line_pairs) > 0 else np.empty((0,2,4))