import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
import cv2
import logging
from io import BytesIO
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D        
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

    connection_line_thickness:int = 1
    connection_line_alpha:float = 0.5

    unmatched_alpha:float = 0.1
    unmatched_line_with = 1
    unmatched_point_size = 5
    unmatched_color = 'grey'
    ellipsoid_3d_plot_extent_around_intersection:float = 0.3

    overlay_font_size:int = 10


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
            figsize = (12, 6),
            provide_second_3d_axis:bool = False,
            figsize_3d = (6,6)
        ) -> None:

        fig, ax = plt.subplots(figsize = figsize, frameon = False)
        self.fig = fig
        self.ax = ax

        if provide_second_3d_axis:
            self.fig3D = plt.figure(figsize=figsize_3d, frameon=False)
            self.ax3D = self.fig3D.add_subplot(111, projection='3d')
        else:
            self.fig3D, self.ax3D = None, None

        self.ax.axis('off')
        self.sc = style_config

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.ax.margins(0, 0)
        self.ax.set_position([0, 0, 1, 1])

        self.resize_factor = None
        self.map_robot_img_coordinates = None
        self.map_headset_img_coordinates = None


    def set_images(self, robot_img_rgb:np.ndarray, headset_img_rgb:np.ndarray):
        h1,w1 = robot_img_rgb.shape[:2]
        h2,w2 = headset_img_rgb.shape[:2]

        resize_factor = h2/h1

        robot_img_rgb_ = cv2.resize(robot_img_rgb, (int(round(resize_factor*w1)), h2), interpolation=cv2.INTER_LINEAR)
        _ , w1_ = robot_img_rgb_.shape[:2]


        x_offset = 10
        cnvs_h, cnvs_w = h2, w1_+w2+x_offset

        canvas = np.zeros((cnvs_h, cnvs_w, 3), dtype = np.uint8)
        canvas[:, :w1_] = robot_img_rgb_
        canvas[:, w1_+x_offset:cnvs_w] = headset_img_rgb

        self.ax.imshow(canvas)

        self.ax.set_xlim(0, cnvs_w)
        self.ax.set_ylim(cnvs_h, 0)

        self.resize_factor = resize_factor
        self.map_robot_img_coordinates = lambda x,y: (x*resize_factor, y*resize_factor)
        self.map_headset_img_coordinates = lambda x,y: (x+w1_+x_offset, y)
        
        self.canvas_width = cnvs_w
        self.canvas_height = cnvs_h


    def plot_matched_points(
            self, 
            robot_img_rgb:np.ndarray, 
            headset_img_rgb:np.ndarray, 
            points1:np.ndarray, 
            points2:np.ndarray,
        ):
        """
        :param img1_rgb: An RGB image as HxWx3-uint8 numpy array
        :param img2_rgb: An RGB image as HxWx3-uint8 numpy array
        :param points1: Nx2 array of 2d points of the form [[x1,y1], ...] in img1_rgb points1[i] is matched to points2[i]
        :param points1: Nx2 array of 2d points
        """
        assert points1.ndim == 2 and points1.shape[-1] == 2, f"invalid 2d points shape: {points1.shape}"
        assert points2.ndim == 2 and points2.shape[-1] == 2, f"invalid 2d points shape: {points2.shape}"
        assert points1.shape == points2.shape, f"Incompatible shapes for matched: {points1.shape} != {points2.shape}"

        if self.map_robot_img_coordinates is None or self.map_headset_img_coordinates is None:
            self.set_images(robot_img_rgb=robot_img_rgb, headset_img_rgb=headset_img_rgb)

        colors = plt.cm.jet(np.linspace(0,1, points1.shape[0]))

        for i, ((x1,y1), (x2, y2)) in enumerate(zip(points1, points2)):
            x1_, y1_ = self.map_robot_img_coordinates(x1, y1)   
            x2_, y2_ = self.map_headset_img_coordinates(x2, y2)         

            self.ax.scatter(x1_, y1_, color=colors[i], s=5, alpha=0.8)
            self.ax.scatter(x2_, y2_, color=colors[i], s=5, alpha=0.8)

            self.ax.plot([x1_, x2_], [y1_, y2_], color=colors[i],
                linewidth=self.sc.connection_line_thickness,
                alpha = self.sc.connection_line_alpha
            )
        self.ax.axis('off')


    def visualize_line_features(
        self,
        robot_lines_2d:np.ndarray,
        headset_lines_2d:np.ndarray, 
    ):
        assert headset_lines_2d.shape == robot_lines_2d.shape and headset_lines_2d.ndim == 2 and headset_lines_2d.shape[-1] == 4

        n_matched_lines = headset_lines_2d.shape[0]
        colors_lines = plt.cm.jet(np.linspace(0,1, n_matched_lines))

        if self.map_headset_img_coordinates is None or self.map_robot_img_coordinates is None:
            logging.info("Could not draw line since mapping was missing")
            return

        for i,(x1, y1, x2, y2) in enumerate(robot_lines_2d):
            (x1_, y1_), (x2_, y2_) = self.map_robot_img_coordinates(x1, y1), self.map_robot_img_coordinates(x2, y2)            
            self.ax.plot([x1_, x2_], [y1_, y2_], color=colors_lines[i], linewidth=1)

        for i,(x1, y1, x2, y2) in enumerate(headset_lines_2d):
            (x1_, y1_), (x2_, y2_) = self.map_headset_img_coordinates(x1, y1), self.map_headset_img_coordinates(x2, y2)            
            self.ax.plot([x1_, x2_], [y1_, y2_], color=colors_lines[i], linewidth=1)


    def transform_gaussians(self, gaussians:np.ndarray)->np.ndarray:
        if self.map_headset_img_coordinates is None:
            logging.info("Could not transform gaussians since mapping was missing")
            return gaussians
        
        mod_gauss = np.copy(gaussians)
        for i in range(gaussians.shape[0]):
            x_, y_ = self.map_headset_img_coordinates(x = mod_gauss[i,0, 2], y = mod_gauss[i, 1, 2])
            mod_gauss[i, 0, 2] = x_
            mod_gauss[i, 1, 2] = y_
        return mod_gauss
    

    def visualize_ellipsoid_features(
        self,
        observed_gaussians:np.ndarray,
        projected_gaussians:np.ndarray,
        proj_match_indices:list[int],
        obs_match_indices:list[int],
        base_t_ellipsoid_s:np.ndarray | None = None,
        primal_quadratic_s:np.ndarray | None = None,
        base_t_cam:np.ndarray | None = None,
        headset_intrinsic_mat:np.ndarray | None = None
    ):
        from ..ellipse_localizer.ellipsoid_utilities_numpy import gaussian_ellipse_s_to_matplotlib_ellipse_s

        mod_observed_gaussians = self.transform_gaussians(observed_gaussians)
        mod_projected_gaussians = self.transform_gaussians(projected_gaussians)

        # Plot matched ones
        n = len(obs_match_indices)
        colors = plt.cm.jet(np.linspace(0,1, n))
        proj_ellipses_matched = gaussian_ellipse_s_to_matplotlib_ellipse_s(
            gaussian_ellipse_s=mod_projected_gaussians[proj_match_indices],
            colors=colors,
            line_style=self.sc.proj_line_style
        )
        obs_ellipses_matched = gaussian_ellipse_s_to_matplotlib_ellipse_s(
            gaussian_ellipse_s=mod_observed_gaussians[obs_match_indices],
            colors=colors,
            line_style=self.sc.obs_line_style
        )
        for proj_e, obs_e in zip(proj_ellipses_matched, obs_ellipses_matched):
            self.ax.add_patch(proj_e)
            self.ax.add_patch(obs_e)

        self.ax.scatter(
            mod_observed_gaussians[obs_match_indices,0,2],
            mod_observed_gaussians[obs_match_indices,1,2],
            color=colors, s=self.sc.point_size, alpha=self.sc.point_alpha, marker = self.sc.obs_point_style)
        
        self.ax.scatter(
            projected_gaussians[proj_match_indices,0,2],
            projected_gaussians[proj_match_indices,1,2],
            color=colors, s=self.sc.point_size, alpha=self.sc.point_alpha, marker = self.sc.proj_point_style)
        
        self.ax.quiver(
            mod_projected_gaussians[proj_match_indices,0,2],
            mod_projected_gaussians[proj_match_indices,1,2],
            mod_observed_gaussians[obs_match_indices,0,2]-mod_projected_gaussians[proj_match_indices,0,2], 
            mod_observed_gaussians[obs_match_indices,1,2]-mod_projected_gaussians[proj_match_indices,1,2],
            angles='xy', scale_units='xy', scale=1,
            color=colors,
            alpha=self.sc.arrow_alpha,
            width=0.005
        )

        if self.fig3D is not None and self.ax3D is not None and primal_quadratic_s is not None and base_t_ellipsoid_s is not None and base_t_cam is not None:
            for i, idx in enumerate(proj_match_indices):

                axis_length = 0.2
                self.ax3D.quiver(*np.array([0,0,0]), *np.array([axis_length, 0, 0]), color='r', arrow_length_ratio=0.1, label='X')
                self.ax3D.quiver(*np.array([0,0,0]), *np.array([0, axis_length, 0]), color='g', arrow_length_ratio=0.1, label='Y')
                self.ax3D.quiver(*np.array([0,0,0]), *np.array([0, 0, axis_length]), color='b', arrow_length_ratio=0.1, label='Z')

                self.plot_ellipsoid_wireframe_matplotlib(
                    self.ax3D, base_t_ellipsoid=base_t_ellipsoid_s[idx], 
                    primal_quadratic=primal_quadratic_s[idx], 
                    color = colors[i]
                )

                # Set camera position:
                extend = self.sc.ellipsoid_3d_plot_extent_around_intersection
                middle = base_t_cam[:3,3]
                if np.abs(base_t_cam[2,3]) < 1e-6 or np.abs(base_t_cam[2,2]) < 1e-6:
                    origin, direction = base_t_cam[:3, 3], base_t_cam[:3, 2]
                    t = -origin[:3, 3] / direction[:3, 2]
                    middle = origin + t * direction

                self.ax3D.set_xlim(middle[0]-extend, middle[0]+extend)
                self.ax3D.set_ylim(middle[1]-extend, middle[1]+extend)
                self.ax3D.set_zlim(middle[2]-extend, middle[2]+extend)
                
                look_dir = -base_t_cam[:3, 2] 
                elev = np.degrees(np.arcsin(look_dir[2] / np.linalg.norm(look_dir)))
                azim = np.degrees(np.arctan2(look_dir[1], look_dir[0]))
                self.ax3D.view_init(elev=elev, azim=azim)
                self.ax3D.axis('off')


    @staticmethod
    def plot_ellipsoid_wireframe_matplotlib(
        ax3d: Axes3D,
        base_t_ellipsoid: np.ndarray, 
        primal_quadratic: np.ndarray,
        color: tuple[float, float, float] | str = "blue",
        resolution: int = 10,
        alpha: float = 0.7,
        linewidth: float = 1.0,
    ):
        from ..ellipse_localizer.ellipsoid_utilities_numpy import sample_points_in_primal_quadratic

        world_points = sample_points_in_primal_quadratic(
            base_t_ellipsoid=base_t_ellipsoid, 
            primal_quadratic=primal_quadratic, 
            resolution=resolution
        )
        
        lines = []

        for i in range(resolution):
            for j in range(resolution):
                start_idx = i * resolution + j
                end_idx = i * resolution + (j + 1) % resolution
                lines.append([start_idx, end_idx])

        for j in range(resolution):
            for i in range(resolution - 1):
                start_idx = i * resolution + j
                end_idx = (i + 1) * resolution + j
                lines.append([start_idx, end_idx])

        for s_idx, end_idx in lines:
            ax3d.plot(
                world_points[[s_idx, end_idx], 0],
                world_points[[s_idx, end_idx], 1],
                world_points[[s_idx, end_idx], 2],
                color=color,
                alpha=alpha,
                linewidth=linewidth,
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
        # Render 2D image
        buf = BytesIO()
        self.fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        
        image = Image.open(buf)
        image_array = np.array(image)
        
        # Render 3D image
        if self.ax3D is not None and self.fig3D is not None:
            buf_3d = BytesIO()
            self.fig3D.savefig(buf_3d, format='png', bbox_inches='tight', pad_inches=0)
            buf_3d.seek(0)
            image_3d = Image.open(buf_3d)
            image_array_3d = np.array(image_3d)
        else:
            image_array_3d = None

        # Combine both images
        if image_array_3d is not None:
            h1, _ = image_array.shape[:2]
            h2, _ = image_array_3d.shape[:2]
            resize_ratio = h1/h2
            image_array_3d = cv2.resize(image_array_3d, (h2, int(h2*resize_ratio)), interpolation=cv2.INTER_LINEAR)
            combined = np.hstack([image_array, image_array_3d])
        else:
            combined = image_array

        return combined[:, :, :3]

    
    def close(self) -> None:
        plt.close(self.fig)
        if self.fig3D is not None:
            plt.close(self.fig3D)


class VideoGenerator:
    def __init__(
            self, 
            style_config:FeatureStyleConfig = FeatureStyleConfig(),
            fps:int = 5,
            figsize:tuple[int, int] = (20, 10),
            use_second_3d_axis:bool = False,
            figsize_3d:tuple[int, int] = (10, 10),
        ) -> None:
        self.style_config = style_config
        self.fps = fps
        self.frames = []
        self.current_feature_drawer = None
        self.figsize = figsize
        self.use_second_3d_axis = use_second_3d_axis
        self.figsize_3d = figsize_3d

        matplotlib.use('Agg')
        import importlib
        import matplotlib.pyplot as plt
        importlib.reload(plt)

 

    def start_new_frame(self)->FeatureDrawing:
        if self.current_feature_drawer is not None:
            self.end_current_frame()

        self.current_feature_drawer = FeatureDrawing(
            style_config=self.style_config,
            figsize=self.figsize,
            provide_second_3d_axis=self.use_second_3d_axis,
            figsize_3d=self.figsize_3d
        )

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