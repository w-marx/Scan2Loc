import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
import cv2

from shared.se3_utilities import rotational_difference, translational_difference


@dataclass(frozen=True, kw_only=True)
class FeatureStyleConfig:
    proj_line_style:str = "--"
    obs_line_style:str = "-"
    proj_point_style:str = "s"
    obs_point_style:str = "o"
    point_size:int = 5
    point_alpha:float = 0.8
    arrow_alpha:float = 0.6

    unmatched_alpha:float = 0.1
    unmatched_line_with = 1
    unmatched_point_size = 5
    unmatched_color = 'grey'

    overlay_font_size:int = 20


class InfoCard():
    def __init__(
            self,
            predicted_base_t_cam:np.ndarray | None = None,
            actual_base_t_cam:np.ndarray | None = None,
            additional_infos:list[str] = []
        ) -> None:
        self.predicted_base_t_cam = predicted_base_t_cam
        self.actual_base_t_cam = actual_base_t_cam
        self.additional_info = additional_infos

    def format_text(self)->str:
        lines = []

        t_error = np.nan
        r_error = np.nan
        if self.predicted_base_t_cam is not None and self.actual_base_t_cam is not None:
            t_error = translational_difference(self.predicted_base_t_cam, self.actual_base_t_cam)
            r_error = rotational_difference(self.predicted_base_t_cam, self.actual_base_t_cam)        
            lines.append(f"Translation Error: {t_error*1000:.2f} mm")
            lines.append(f"Rotation Error: {np.rad2deg(r_error):.3f} deg")
        else:
            lines.append(f"Translation Error: unknown")
            lines.append(f"Rotation Error: unknown")
        lines += self.additional_info

        return "\n".join(lines) if lines else ""



class FeatureDrawing:
    def __init__(
            self,
            style_config: FeatureStyleConfig = FeatureStyleConfig(),
            figsize = (12, 8)
        ) -> None:

        fig, ax = plt.subplots(1,1, figsize = figsize, frameon = False)
        self.fig = fig
        self.ax = ax
        self.ax.axis('off')
        self.sc = style_config

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.ax.margins(0, 0)
        self.ax.set_position([0, 0, 1, 1]) 

    def set_bg_image(self,rgb_image:np.ndarray):
        h,w = rgb_image.shape[:2]
        self.ax.imshow(rgb_image)
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(h, 0)
        


    def draw_point_pairs(self, points_observed:np.ndarray, points_projected:np.ndarray):
        n_points = points_observed.shape[0]

        colors = plt.cm.jet(np.linspace(0,1, n_points))

        self.ax.scatter(
            points_projected[:, 0], 
            points_projected[:, 1], 
            c=colors, 
            s=self.sc.point_size, 
            alpha=self.sc.point_alpha, 
            marker = self.sc.proj_point_style, 
            label = 'Projected'
        )
        self.ax.scatter(
            points_observed[:, 0], 
            points_observed[:, 1], 
            c= colors, 
            s= self.sc.point_size, 
            alpha= self.sc.point_alpha, 
            marker = self.sc.obs_point_style, 
            label = 'Observed'
        )

        self.ax.quiver(
            points_projected[:, 0], points_projected[:, 1],
            points_observed[:, 0]-points_projected[:, 0], points_observed[:, 1]-points_projected[:, 1],
            angles='xy', scale_units='xy', scale=1,
            color=colors,
            alpha=self.sc.arrow_alpha,
            width=0.005
        )

    def add_info_overlay(self, info_card: InfoCard) -> None:
        text = info_card.format_text()
        if text:
            self.ax.text(
                10, 30, text,
                fontsize=self.sc.overlay_font_size,
                color='white',
                fontweight='bold',
                family='monospace',
                bbox=dict(
                    boxstyle='round,pad=0.5',
                    facecolor='black',
                    alpha=0.7,
                    edgecolor='white'
                ),
                verticalalignment='top'
            )
    
    def render_to_image(self) -> np.ndarray:
        self.fig.canvas.draw()

        try:
            buffer = self.fig.canvas.buffer_rgba()
            image_array = np.asarray(buffer)
        except AttributeError:
            image_array = np.array(self.fig.canvas.renderer.buffer_rgba())
            
        image_array = np.asarray(buffer)
        return image_array[:, :, :3]

    
    def close(self) -> None:
        plt.close(self.fig)


class VideoGenerator:
    def __init__(
            self, 
            style_config:FeatureStyleConfig = FeatureStyleConfig(),
            fps:int = 5,
            dpi:int = 100
        ) -> None:
        self.style_config = style_config
        self.fps = fps
        self.frames = []
        self.current_feature_drawer = None
        self.dpi = dpi
 

    def start_new_frame(self, bg_bgr_image:np.ndarray)->FeatureDrawing:
        if self.current_feature_drawer is not None:
            self.end_current_frame()

        rgb_image = cv2.cvtColor(bg_bgr_image, cv2.COLOR_BGR2RGB)

        self.current_feature_drawer = FeatureDrawing(
            style_config=self.style_config,
            figsize=(rgb_image.shape[1]/self.dpi, rgb_image.shape[0]/self.dpi)
        )

        self.current_feature_drawer.set_bg_image(rgb_image=rgb_image)
        return self.current_feature_drawer


    def annotate_frame(self, info_card:InfoCard)->None:
        if self.current_feature_drawer is not None:
            self.current_feature_drawer.add_info_overlay(info_card)
 

    def end_current_frame(self)->None:
        if self.current_feature_drawer:
            img_array = self.current_feature_drawer.render_to_image()
            self.frames.append(img_array)
            self.current_feature_drawer.close()
            self.current_feature_drawer = None


    def save_video(self, location:str)->None:
        if not self.frames:
            raise ValueError(f"Cant save empty video to {location}")
        

        height, width = self.frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(location, fourcc, self.fps, (width, height))
        
        for frame in self.frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"Video saved to {location}")


