import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
import cv2
import logging
from io import BytesIO
from PIL import Image
import open3d as o3d
from typing import Any
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
    line_widht:int = 1

    connection_line_thickness:int = 1
    connection_line_alpha:float = 0.5

    unmatched_alpha:float = 0.1
    unmatched_line_with = 1
    unmatched_point_size = 5
    unmatched_color = 'grey'
    ellipsoid_3d_plot_extent_around_intersection:float = 0.3

    overlay_font_size:int = 10
    show_bg_point_cloud:bool = True
    show_bg_point_cloud_colored:bool = True
    use_headset_viewpoint:bool = True
    line_widht_3d:float = 20

    show_3d_points:bool = True
    points_3d_size:float = 0.008

    line_pnpl_radius_3d:float = 0.01
    line_pne_use_cylinders:bool = True
    line_pnpe_radius_3d:float = 0.002


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


def create_ellipsoid_cylinder_lineset(
    base_t_ellipsoid:np.ndarray, 
    primal_quadratic:np.ndarray,
    ellipsoid_resolution:int = 20,
    cylinder_resolution:int = 6,
    cylinder_radius:float = 0.001,
    color:np.ndarray = np.array([0, 0, 0, 0])
):
    from ..ellipse_localizer.ellipsoid_utilities_numpy import sample_points_in_primal_quadratic

    world_points = sample_points_in_primal_quadratic(
        base_t_ellipsoid=base_t_ellipsoid, primal_quadratic=primal_quadratic, resolution=ellipsoid_resolution
    )

    lines = []

    for i in range(ellipsoid_resolution):
        for j in range(ellipsoid_resolution):
            start_idx = i * ellipsoid_resolution + j
            end_idx = i * ellipsoid_resolution + (j + 1) % ellipsoid_resolution
            lines.append(np.concatenate([world_points[start_idx], world_points[end_idx]]))


    for j in range(ellipsoid_resolution):
        for i in range(ellipsoid_resolution - 1):
            start_idx = i * ellipsoid_resolution + j
            end_idx = (i + 1) * ellipsoid_resolution + j
            lines.append(np.concatenate([world_points[start_idx], world_points[end_idx]]))
    
    return lines_3d_for_o3d(
        lines3d=np.array(lines),
        colors=np.tile(color, (len(lines),1)),
        radius=cylinder_radius,
        resolution=cylinder_resolution
    )


class FeatureDrawing:
    def __init__(
            self,
            style_config: FeatureStyleConfig = FeatureStyleConfig(),
            figsize = (12, 6),
            provide_second_3d_axis:bool = False,
            figsize_3d:tuple[int, int] = (600,600),
            bgr_images:np.ndarray | None = None,
            xyz_images:np.ndarray | None = None
        ) -> None:

        fig, ax = plt.subplots(figsize = figsize, frameon = False)
        self.fig = fig
        self.ax = ax
        self.sc = style_config


        if provide_second_3d_axis:
            self.o3d_vis = o3d.visualization.Visualizer()
            w, h = figsize_3d

            self.o3d_vis.create_window(window_name='Open3D', width= w, height= h, visible = False)
            # Add scene geometry
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
            self.o3d_vis.add_geometry(coord_frame)

            if self.sc.show_bg_point_cloud and xyz_images is not None:
                world_points = xyz_images.reshape(-1,3)
                no_nan_mask = np.isfinite(world_points).all(axis=-1)

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(world_points[no_nan_mask])

                if self.sc.show_bg_point_cloud_colored and bgr_images is not None:
                    world_colors = bgr_images.reshape(-1,3).astype(np.float32)[:, ::-1]/255
                    pcd.colors = o3d.utility.Vector3dVector(world_colors[no_nan_mask])

                self.o3d_vis.add_geometry(pcd)


            self.curr_geoms = []
        else:
            self.o3d_vis = None

        self.ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.ax.margins(0, 0)
        self.ax.set_position([0, 0, 1, 1])

        self.resize_factor = None
        self.map_robot_img_coordinates = None
        self.map_headset_img_coordinates = None


    def visualize_localizer(
        self,
        headset_img_rgb:np.ndarray, 
        robot_img_rgb:np.ndarray | None = None, 

        # PnP
        points1:np.ndarray | None = None, 
        points2:np.ndarray | None = None,
        points1_3d:np.ndarray | None = None,

        # PnP+L
        robot_lines_2d:np.ndarray | None = None,
        headset_lines_2d:np.ndarray | None = None,
        robot_lines_3d:np.ndarray | None = None,

        # Ellipsoid
        observed_gaussians:np.ndarray | None = None,
        projected_gaussians:np.ndarray | None = None,
        proj_match_indices:list[int] | None = None,
        obs_match_indices:list[int] | None = None,
        base_t_ellipsoid_s:np.ndarray | None = None,
        primal_quadratic_s:np.ndarray | None = None,

        # 3D
        base_t_cam:np.ndarray | None = None,
        headset_intrinsic_mat:np.ndarray | None = None,
    ):
        if self.map_robot_img_coordinates is None or self.map_headset_img_coordinates is None and robot_img_rgb is not None:
            self._set_images(robot_img_rgb=robot_img_rgb, headset_img_rgb=headset_img_rgb)

        if points1 is not None and points2 is not None:
            self._plot_matched_points(points1=points1, points2=points2)

        if robot_lines_2d is not None and headset_lines_2d is not None:
            self._visualize_line_features(robot_lines_2d=robot_lines_2d, headset_lines_2d=headset_lines_2d)

        if all([x is not None for x in [obs_match_indices, observed_gaussians, projected_gaussians, proj_match_indices, obs_match_indices]]):
            n = len(obs_match_indices)
            colors = plt.cm.jet(np.linspace(0,1, n))
            self._visualize_ellipsoid_features(
                observed_gaussians=observed_gaussians,
                projected_gaussians=projected_gaussians,
                proj_match_indices=proj_match_indices,
                obs_match_indices= obs_match_indices,
                colors= colors
            )


        if self.o3d_vis is None:
            return
        
        from ..ellipse_localizer.ellipsoid_utilities_numpy import create_ellipsoid_lineset
        
        to_vis_3d = []
    
        if points1_3d is not None and self.sc.show_3d_points:
            colors = plt.cm.jet(np.linspace(0,1, points1_3d.shape[0]))
            for i, p3d in enumerate(points1_3d):
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.sc.points_3d_size)
                sphere.paint_uniform_color(colors[i, :3])
                to_vis_3d.append(sphere)
                sphere.translate(p3d)
                to_vis_3d.append(sphere)

        if robot_lines_3d is not None:
            colors_lines = plt.cm.jet(np.linspace(0,1, robot_lines_3d.shape[0]))
            to_vis_3d += lines_3d_for_o3d(robot_lines_3d, colors= colors_lines)


        if all([x is not None for x in [primal_quadratic_s, base_t_ellipsoid_s, proj_match_indices]]):
            if self.sc.line_pne_use_cylinders:
                for i, idx in enumerate(proj_match_indices):
                    meshes = create_ellipsoid_cylinder_lineset(
                        base_t_ellipsoid = base_t_ellipsoid_s[idx],
                        primal_quadratic = primal_quadratic_s[idx],
                        ellipsoid_resolution=10,
                        cylinder_radius=self.sc.line_pnpe_radius_3d,
                        color=colors[i][:3]
                    )
                    to_vis_3d += meshes
            else:
                for i, idx in enumerate(proj_match_indices):
                    lineset = create_ellipsoid_lineset(
                        base_t_ellipsoid = base_t_ellipsoid_s[idx],
                        primal_quadratic = primal_quadratic_s[idx],
                        resolution=10
                    )
                    lineset.paint_uniform_color(colors[i][:3])
                    to_vis_3d.append(lineset)


        self._replace_geometry(
            geometries=to_vis_3d, 
            base_t_cam=base_t_cam, 
            headset_intrinsic_mat=headset_intrinsic_mat, 
            headset_image_size=(headset_img_rgb.shape[1], headset_img_rgb.shape[0])
        )





    def _set_images(self, robot_img_rgb:np.ndarray, headset_img_rgb:np.ndarray):
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


    def _plot_matched_points(
            self, 
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

        colors = plt.cm.jet(np.linspace(0,1, points1.shape[0]))

        for i, ((x1,y1), (x2, y2)) in enumerate(zip(points1, points2)):
            x1_, y1_ = self.map_robot_img_coordinates(x1, y1)   
            x2_, y2_ = self.map_headset_img_coordinates(x2, y2)         

            self.ax.scatter(x1_, y1_, color=colors[i], s=self.sc.point_size, alpha=self.sc.point_alpha)
            self.ax.scatter(x2_, y2_, color=colors[i], s=self.sc.point_size, alpha=self.sc.point_alpha)

            self.ax.plot([x1_, x2_], [y1_, y2_], color=colors[i],
                linewidth=self.sc.connection_line_thickness,
                alpha = self.sc.connection_line_alpha
            )
        self.ax.axis('off')


    def _visualize_line_features(
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
            self.ax.plot([x1_, x2_], [y1_, y2_], color=colors_lines[i], linewidth=self.sc.line_widht)

        for i,(x1, y1, x2, y2) in enumerate(headset_lines_2d):
            (x1_, y1_), (x2_, y2_) = self.map_headset_img_coordinates(x1, y1), self.map_headset_img_coordinates(x2, y2)            
            self.ax.plot([x1_, x2_], [y1_, y2_], color=colors_lines[i], linewidth=self.sc.line_widht)


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
    

    def set_camera3d_viewpoint(
        self,
        base_t_cam:np.ndarray | None = None,
        headset_intrinsic_mat:np.ndarray | None = None,
        image_size:tuple[int, int] | None = None
    ):
        if base_t_cam is None or headset_intrinsic_mat is None or self.sc.use_headset_viewpoint is None or image_size is None:
            return
        
        view_control = self.o3d_vis.get_view_control()
        camera_params = view_control.convert_to_pinhole_camera_parameters()

        camera_params.extrinsic = np.linalg.inv(base_t_cam)

        K = headset_intrinsic_mat
        w, h = image_size
        camera_params.intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    
        view_control.convert_from_pinhole_camera_parameters(camera_params)      

    def _replace_geometry(
            self,
            geometries:list[Any],
            base_t_cam:np.ndarray | None = None,
            headset_intrinsic_mat:np.ndarray | None = None,
            headset_image_size:tuple[int, int] | None = None
    ):
        if self.o3d_vis is None:
            return

        for geom in self.curr_geoms:
            self.o3d_vis.remove_geometry(geom)
            self.curr_geoms.clear()

        for geom in geometries:
                self.o3d_vis.add_geometry(geom)
                self.curr_geoms.append(geom)


        if self.sc.use_headset_viewpoint:
            self.set_camera3d_viewpoint(
                base_t_cam=base_t_cam,
                headset_intrinsic_mat=headset_intrinsic_mat,
                image_size=headset_image_size
            )
        self.o3d_vis.poll_events()
        self.o3d_vis.update_renderer()


    def _visualize_ellipsoid_features(
        self,
        observed_gaussians:np.ndarray,
        projected_gaussians:np.ndarray,
        proj_match_indices:list[int],
        obs_match_indices:list[int],
        colors:np.ndarray,
    ):
        from ..ellipse_localizer.ellipsoid_utilities_numpy import gaussian_ellipse_s_to_matplotlib_ellipse_s

        mod_observed_gaussians = self.transform_gaussians(observed_gaussians)
        mod_projected_gaussians = self.transform_gaussians(projected_gaussians)

        # Plot matched ones
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


    def add_info_overlay(self, info_card: InfoCard) -> None:
        text = info_card.format_text()
        if text:
            self.ax.text(10, 30, text,fontsize=self.sc.overlay_font_size,
                color='white', fontweight='bold', family='monospace',
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
        
        image = Image.open(buf).convert('RGB')
        image_array = np.array(image)
        
        # Render 3D image
        if self.o3d_vis is not None:
            image_array_3d = self.o3d_vis.capture_screen_float_buffer(do_render=True)
            image_array_3d = (np.asarray(image_array_3d) * 255).astype(np.uint8)

        else:
            image_array_3d = None

        # Combine both images
        if image_array_3d is not None:
            h1, _ = image_array.shape[:2]
            h2, w2 = image_array_3d.shape[:2]
            resize_ratio = h1/h2
            image_array_3d = cv2.resize(image_array_3d, (int(w2*resize_ratio), int(round(h2*resize_ratio))), interpolation=cv2.INTER_LINEAR)
            combined = np.hstack([image_array, image_array_3d])
        else:
            combined = image_array

        return combined[:, :, :3]

    
    def close(self) -> None:
        plt.close(self.fig)
        if self.o3d_vis is not None:
            for geom in self.curr_geoms:
                    self.o3d_vis.remove_geometry(geom)
            self.curr_geoms.clear()
            self.o3d_vis.destroy_window()
            self.o3d_vis = None

class VideoGenerator:
    def __init__(
            self, 
            style_config:FeatureStyleConfig = FeatureStyleConfig(),
            fps:int = 5,
            figsize:tuple[int, int] = (20, 10),
            use_second_3d_axis:bool = False,
            figsize_3d:tuple[int, int] = (600, 600),
            scan_bgr_images:np.ndarray | None = None,
            scan_xyz_images:np.ndarray | None = None
        ) -> None:
        self.style_config = style_config
        self.fps = fps
        self.frames = []
        self.current_feature_drawer = None
        self.figsize = figsize
        self.use_second_3d_axis = use_second_3d_axis
        self.figsize_3d = figsize_3d

        self.scan_bgr_images = scan_bgr_images
        self.scan_xyz_images = scan_xyz_images

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
            figsize_3d=self.figsize_3d,
            bgr_images=self.scan_bgr_images,
            xyz_images=self.scan_xyz_images
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