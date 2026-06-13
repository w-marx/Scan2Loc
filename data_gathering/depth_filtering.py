import pyrealsense2 as rs
from dataclasses import dataclass

@dataclass(frozen=True, kw_only=True)
class DepthOptimisationConfig:
    decimation_magnitude: int = 2

    # Spatial filter
    spatial_alpha: float = 0.35
    spatial_delta: int = 12
    spatial_holes_fill: int = 0

    # Temporal filter
    temporal_alpha: float = 0.25
    temporal_delta: int = 20
    use_temporal: bool = True

    use_hole_filling:bool = False

DEPTH_OPTIMIZATION_CONFIGS = {
    "standard":DepthOptimisationConfig(),
    "standard_no_temporal":DepthOptimisationConfig(use_temporal=False)
}


class DepthFilterPipeline:
    def __init__(self, config:DepthOptimisationConfig):
        
        self.config = config

        self.depth_to_disparity = rs.disparity_transform(True)
        self.disparity_to_depth = rs.disparity_transform(False)

        self.decimation_filter = rs.decimation_filter()
        self.spatial_filter = rs.spatial_filter()
        self.temporal_filter = rs.temporal_filter()
        self.hole_filling_filter = rs.hole_filling_filter()


        self.decimation_filter.set_option(
            rs.option.filter_magnitude,
            config.decimation_magnitude
        )

        self.spatial_filter.set_option(rs.option.filter_smooth_alpha, config.spatial_alpha)
        self.spatial_filter.set_option(rs.option.filter_smooth_delta, config.spatial_delta)
        self.spatial_filter.set_option(rs.option.holes_fill, config.spatial_holes_fill)

        self.temporal_filter.set_option(rs.option.filter_smooth_alpha, config.temporal_alpha)
        self.temporal_filter.set_option(rs.option.filter_smooth_delta, config.temporal_delta)


    def optimize_depth_image(self,depth_frame):
        """            
        :return: Optimized depth image
        """
        
        filtered = self.decimation_filter.process(depth_frame)
        filtered = self.depth_to_disparity.process(filtered)
        filtered = self.spatial_filter.process(filtered)
        
        if self.config.use_temporal:
            filtered = self.temporal_filter.process(filtered)
        
        filtered = self.disparity_to_depth.process(filtered)
        
        if self.config.use_hole_filling:
            filtered = self.hole_filling_filter.process(filtered)
        
        return filtered