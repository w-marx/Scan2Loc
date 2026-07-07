import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.axes import Axes
from dataclasses import dataclass, field
from numbers import Real
import logging

from shared.assertion_helpers import assert_mxnx3_np_uint8_image

from .line_utilities import lines_to_points_distances_2d
from ..utilities.union_find import UnionFind


def merge_line_seg_cluster_into_one_weighted(line_segs_2d:np.ndarray, use_pca:bool = False)->np.ndarray:
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
    if use_pca:
        logging.debug("pca for weighted merging used")
        weighted_dxdy = dxdy[:, :]/line_segment_lengths[:, None] * line_segment_relevances[:, None]
        cov = weighted_dxdy.T @ weighted_dxdy
        vals, vecs = np.linalg.eigh(cov)
        weighted_direction = vecs[:, np.argmax(vals)]
        weighted_direction = weighted_direction / np.linalg.norm(weighted_direction)
    else:
        logging.debug("weighted direction for weighted merging used")
        weighted_direction = np.mean(dxdy[:, :]/line_segment_lengths[:, None] * line_segment_relevances[:, None], axis = 0)
        weighted_direction = weighted_direction/np.linalg.norm(weighted_direction)

    weighted_center = np.sum((line_segs_2d[:, :2]+dxdy/2)*line_segment_relevances[:, None], axis = 0)

    centered_points = line_segs_2d.reshape(-1,2)-weighted_center
    dists_to_center = centered_points @ weighted_direction

    start_p = weighted_center + dists_to_center.min()*weighted_direction
    end_p = weighted_center + dists_to_center.max()*weighted_direction

    return np.concatenate([start_p, end_p])


def merge_close_line_segments(
        line_segs_2d:np.ndarray,
        max_angle_diff:float = 5,
        max_midpoint_dist:float = 5,
        max_endpoint_dist:float = 15,
        quick_join:bool = False,
        use_pca_for_merge:bool = False
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
    :param use_pca_for_merge: If true will use pca to find the weighted direction instead of the mean direction
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
        return np.array([merge_line_seg_cluster_into_one_weighted(line_segs_2d[np.array(line_cluster)], use_pca=use_pca_for_merge)
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

    return np.array([merge_line_seg_cluster_into_one_weighted(line_segs_2d[np.array(line_cluster)], use_pca=use_pca_for_merge)
        for line_cluster in list(clusters.values())
    ])


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


def visualize_line_cleanup(
        lines_before:np.ndarray,
        lines_after:np.ndarray,
        background_image:np.ndarray,
        ax_before: Axes | None = None,
        ax_after: Axes | None = None,
)->None:
    """
    Visualizes how the lines change through cleanup
    :param lines_before: Nx4 array of line segments of the style [[x0, y0, x1, y1], ...]
    :param lines_after: Mx4 array of line segments of the style [[x0, y0, x1, y1], ...]
    :param background_image: a HxW greyscale image
    :param ax_before: An matplotlib axis on which before will be plotted (if None it will be created and the plot shown)
    :param ax_after: Same as ax_before for after
    """
    has_to_plot = ax_before is None or ax_after is None
    if has_to_plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 10))
        ax_before = axes[0]
        ax_after = axes[1]

    ax_before.imshow(background_image, cmap='gray')
    ax_after.imshow(background_image, cmap='gray')
    
    ax_before.grid(False)
    ax_after.grid(False)

    lines_xy_raw = [((line[0], line[1]), (line[2], line[3])) for line in lines_before]
    lc1_raw = LineCollection(lines_xy_raw, linewidths=2, alpha=0.8, color = plt.cm.jet(np.linspace(0, 1, lines_before.shape[0])))
    ax_before.add_collection(lc1_raw)
    ax_before.set_title("Lines before cleanup")
    line_colors_processed = plt.cm.jet(np.linspace(0, 1, lines_after.shape[0]))
    lines_xy_processed = [((line[0], line[1]), (line[2], line[3])) for line in lines_after]
    lc1_processed = LineCollection(lines_xy_processed, linewidths=2, alpha=0.8, color = line_colors_processed)
    ax_after.add_collection(lc1_processed)
    ax_after.set_title("Lines after cleanup")

    if has_to_plot:
        plt.show()


@dataclass(frozen=True, kw_only=True)
class LineMerging2dConfig:
    """
    Discribes a line merging pass, that merges the lines and then removes
    lines with line_length < min_line_length.
    Distances are in a percentage of the diagonal length
    :param max_angle_diff: The maximum angle between 2 lines in degree.
    :param max_midpoint_dist: The maximum distance between 2 midpoints (0 to 1).
    :param max_midpoint_dist: The maximal angle between the line midpoint and the infinite other line.
    :param max_endpoint_dist: The maximal shortest distance between an endpoint pair.
    :param min_line_length: The minimum length of the line.
    :param pca: Wheather to use pca for the weighted join or the direction average
    """
    max_angle_diff:float = 2
    max_midpoint_dist:float = 0.005
    max_endpoint_dist:float = 0.01
    min_line_length:float = 0.01
    use_quick_merge:bool = False
    use_pca:bool = True

    def __post_init__(self):
        assert isinstance(self.max_angle_diff, Real) and 0 <= self.max_angle_diff <= 180
        assert isinstance(self.max_angle_diff, Real) and 0 <= self.max_midpoint_dist < 1
        assert isinstance(self.max_angle_diff, Real) and 0 <= self.max_endpoint_dist < 1
        assert isinstance(self.max_angle_diff, Real) and 0 <= self.min_line_length < 1
        assert isinstance(self.use_quick_merge, bool)

line_merging_2d_config_for_short_lines = LineMerging2dConfig(
    max_angle_diff = 2,
    max_midpoint_dist = 3/850,
    max_endpoint_dist = 0.01,
    min_line_length = 10/850,
    use_quick_merge = False,
    use_pca=True
)

line_merging_2d_config_for_longer_lines = LineMerging2dConfig(
    max_angle_diff = 3,
    max_midpoint_dist = 4/850,
    max_endpoint_dist = 0.02,
    min_line_length = 40/850,
    use_quick_merge = False,
    use_pca=True
)

@dataclass(frozen=True, kw_only=True)
class MultiPassLineMergingConfig:
    """
    Config consisting of N line cleanup passes
    """
    passes: list[LineMerging2dConfig] = field(
        default_factory=lambda: (
            [
                line_merging_2d_config_for_short_lines,
                line_merging_2d_config_for_longer_lines
            ]
        )
    )

    def __post_init__(self):
        assert all([isinstance(obj, LineMerging2dConfig) for obj in self.passes])


class LineGenerator:
    def __init__(
            self,
            line_cleanup_config:MultiPassLineMergingConfig = MultiPassLineMergingConfig(),
            lsd_diagonal_size: None | float = None,
            visualize_cleanup:bool | tuple[Axes, Axes] = False,
    ):
        """
        :param line_cleanup_config: The cleanup passes that will be enacted after LSD on any image.
        :param cam2_lsd_diagonal_size: If not None the cam2 images will be scaled to that diagonal size before LSD (smaller -> better runtime).
        :param visualize_cleanup: If True will plot the cleanup
        """
        self.line_cleanup_config = line_cleanup_config
        self.lsd_diagonal_size = lsd_diagonal_size
        self.line_seg_detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE)


        if isinstance(visualize_cleanup, bool):
            self.visualize_cleanup = visualize_cleanup
            self.visualize_axis = (None, None)
        else:
            self.visualize_cleanup = True
            self.visualize_axis = visualize_cleanup


    def _cleanup_lines(self, lines: np.ndarray, diagonal_length: float = 1.0) -> np.ndarray:
        """
        Takes the lines and cleans them up according to the cleanup-config of the instance
        :param lines: Bx4 array of lines of the style: [[x0, y0, x1, y2], ... ]
        :param diagonal_length: The length of the diagonal of the image (lines will be normalized by this)
        :return: B'x4 array of lines of the style: [[x0, y0, x1, y2], ... ], with B' <= B
        """
        assert diagonal_length > 0, f"image cant have size 0 or smaller: diagonal length = {diagonal_length}"

        lines = lines / diagonal_length

        for ref_conf in self.line_cleanup_config.passes:
            lines = remove_short_2d_line_segments(
                merge_close_line_segments(
                    lines, ref_conf.max_angle_diff, ref_conf.max_midpoint_dist, ref_conf.max_endpoint_dist,
                    ref_conf.use_quick_merge, use_pca_for_merge=ref_conf.use_pca
                ),
                min_line_length_px=ref_conf.min_line_length)
        return lines * diagonal_length

    @staticmethod
    def _image_diagonal(image: np.ndarray) -> float:
        h, w = image.shape[:2]
        return np.sqrt(h ** 2 + w ** 2)

    def get_lines(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        Runs LSD on an image that can be scaled down beforehand, then cleans those lines up and scales them back
        :param bgr_image: The HxWx3-uint8 BGR image to be done lsd upon
        :param lsd_at_diag_size: None or the diagonal size with which to run LSD (for runtime improvements)
        :return Nx4 line array of the format: [[x1, y1, x2, y2], ...]
        """
        assert assert_mxnx3_np_uint8_image(bgr_image)

        grey_img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        if self.lsd_diagonal_size is not None:
            h_orig, w_orig = bgr_image.shape[:2]
            lsd_scale = self.lsd_diagonal_size / self._image_diagonal(bgr_image)
            grey_img = cv2.resize(grey_img, (int(w_orig * lsd_scale), int(h_orig * lsd_scale)),
                                  interpolation=cv2.INTER_AREA)

        raw_lines = self.line_seg_detector.detect(grey_img)[0].squeeze(1)
        clean_lines = self._cleanup_lines(raw_lines, diagonal_length=self._image_diagonal(grey_img))

        if self.visualize_cleanup:
            visualize_line_cleanup(
                lines_before=raw_lines, 
                lines_after=clean_lines, 
                background_image=grey_img, 
                ax_before=self.visualize_axis[0], 
                ax_after=self.visualize_axis[1]
            )

        if self.lsd_diagonal_size is not None:
            clean_lines *= self._image_diagonal(bgr_image) / self._image_diagonal(grey_img)

        return clean_lines