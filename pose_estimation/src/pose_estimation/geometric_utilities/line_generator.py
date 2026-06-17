import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.axes import Axes
from dataclasses import dataclass, field
from numbers import Number, Real
import torch

from shared.assertion_helpers import assert_intrinsic_mat, assert_mxnx3_np_uint8_image_batch, assert_mxnx3_np_uint8_image, assert_bgr_xyz_image_pair_batch

from .line_utilities import *
from .time_tracker import *


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
    :param background_image: HxWx3 BGR image as numpy array or HxW Greyscale image
    :param ax_before: An matplotlib axis on which before will be plotted (if None it will be created and the plot shown)
    :param ax_after: Same as ax_before for after
    """
    has_to_plot = ax_before is None or ax_after is None
    if has_to_plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 10))
        ax_before = axes[0]
        ax_after = axes[1]
    if background_image.ndim == 3:
        background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
    ax_before.imshow(background_image)
    ax_after.imshow(background_image)
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
    """
    max_angle_diff:float = 2
    max_midpoint_dist:float = 0.005
    max_endpoint_dist:float = 0.01
    min_line_length:float = 0.01
    use_quick_merge:bool = False

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
    use_quick_merge = False
)

line_merging_2d_config_for_longer_lines = LineMerging2dConfig(
    max_angle_diff = 3,
    max_midpoint_dist = 4/850,
    max_endpoint_dist = 0.02,
    min_line_length = 40/850,
    use_quick_merge = False
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
            cam2_lsd_diagonal_size: None | float = None,
            visualize_cleanup:bool = False
    ):
        """
        :param line_cleanup_config: The cleanup passes that will be enacted after LSD on any image.
        :param cam2_lsd_diagonal_size: If not None the cam2 images will be scaled to that diagonal size before LSD (smaller -> better runtime).
        :param visualize_cleanup: If True will plot the cleanup
        """
        self.line_cleanup_config = line_cleanup_config
        self.lsd_diagonal_size = lsd_diagonal_size
        self.line_seg_detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE)
        self.visualize_cleanup = visualize_cleanup

    def get_lines(self, bgr_image:np.ndarray)->np.ndarray:
        """
        :param bgr_image: The HxWx3-uint8 BGR image to extract the lines from
        :return: Bx4 array of lines of the style: [[x0, y0, x1, y2], ... ]
        """
        assert assert_mxnx3_np_uint8_image(bgr_image)
        return self._lsd_and_cleanup_on_image(bgr_image)


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
                    ref_conf.use_quick_merge
                ),
                min_line_length_px=ref_conf.min_line_length)
        return lines * diagonal_length

    @staticmethod
    def _image_diagonal(image: np.ndarray) -> float:
        h, w = image.shape[:2]
        return np.sqrt(h ** 2 + w ** 2)

    def _lsd_and_cleanup_on_image(self, bgr_image: np.ndarray) -> np.ndarray:
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
            visualize_line_cleanup(lines_before=raw_lines, lines_after=clean_lines, background_image=grey_img)

        if self.lsd_diagonal_size is not None:
            clean_lines *= self._image_diagonal(bgr_image) / self._image_diagonal(grey_img)

        return clean_lines