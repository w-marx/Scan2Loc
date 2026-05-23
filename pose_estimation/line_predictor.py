import cv2
import numpy as np
from predictor_handling import *
from extractors_and_matchers import *
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from pose_estimation.pnpl_optimizer import optimize_pnpl
from line_utilities import * 

import time

class LinePredictor(PosePredictor):
    def __init__(
            self,
            cam2_intrinsic_mtx:np.ndarray,
            extract_and_match:ExtractAndMatch = ExtractAndMatchLoMa(),

            initial_guess_min_number_inlier_after_ransac:int = 6,
            initial_guess_ransac_itterations:int = 500,
            initial_guess_ransac_reprojection_error:float = 5.0,
            initial_guess_ransac_confidence:float = 0.9,

            pass1_line_merging_2d_max_angle_diff_deg:float = 3,
            pass1_line_merging_2d_max_midpoint_dist_px:float = 3,
            pass1_line_merging_2d_max_endpoint_dist_px:float = 20,
            pass1_min_line_length:float = 20,

            pass2_line_merging_2d_max_angle_diff_deg:float = 10,
            pass2_line_merging_2d_max_midpoint_dist_px:float = 5,
            pass2_line_merging_2d_max_endpoint_dist_px:float = 30,
            pass2_min_line_length:float = 50,

            line_matching_max_dist_line_to_point_px:float = 5,
            line_matching_min_number_supporting_points:int = 2,

            line_fitting_3d_use_ransaac:bool = True,
            line_fitting_3d_iterations:int = 100,
            line_fitting_3d_inlier_distance:float = 0.005,

            pose_optimization_line_vs_point_relevance:float = 0.5,

            debug_dont_refine_ransac:bool = False,
            debug_visualize_2d:bool = False,
            debug_visualize_3d:bool = False
        ):
        super().__init__()
        self.cam2_intrinsic_mtx = cam2_intrinsic_mtx
        self.extract_and_match = extract_and_match

        self.initial_guess_min_number_inlier_after_ransac = initial_guess_min_number_inlier_after_ransac
        self.initial_guess_ransac_itterations = initial_guess_ransac_itterations
        self.initial_guess_ransac_reprojection_error = initial_guess_ransac_reprojection_error
        self.initial_guess_ransac_confidence = initial_guess_ransac_confidence


        line_refinement_pass = lambda line_segs_2d, max_ang_diff, max_ep_diff, max_mp_diff, min_line_length: (
            remove_short_2d_line_segments(
                merge_close_line_segments(
                    line_segs_2d=line_segs_2d, max_angle_diff=max_ang_diff,
                    max_endpoint_dist=max_ep_diff, max_midpoint_dist=max_mp_diff
                ),
            min_line_length_px= min_line_length
        ))

        self.line_refinement = lambda line_segs_2d: (
            line_refinement_pass(
                line_refinement_pass(
                    line_segs_2d=line_segs_2d,
                    max_ep_diff=pass1_line_merging_2d_max_endpoint_dist_px,
                    max_ang_diff=pass1_line_merging_2d_max_angle_diff_deg,
                    max_mp_diff=pass1_line_merging_2d_max_midpoint_dist_px,
                    min_line_length=pass1_min_line_length
                ),
                max_ep_diff=pass2_line_merging_2d_max_endpoint_dist_px,
                max_ang_diff=pass2_line_merging_2d_max_angle_diff_deg,
                max_mp_diff=pass2_line_merging_2d_max_midpoint_dist_px,
                min_line_length=pass2_min_line_length
            )
        )

        self.line_matching_max_dist_line_to_point_px = line_matching_max_dist_line_to_point_px
        self.line_matching_min_number_supporting_points = line_matching_min_number_supporting_points

        self.line_fitting_3d_method = (lambda points3d:(
            line_segment_regression_3d_ransaac(
                xyz_points=points3d,
                inlier_distance=line_fitting_3d_inlier_distance,
                itterations=line_fitting_3d_iterations
            )
        )) if line_fitting_3d_use_ransaac else lambda points3d: robust_pca_2d_3d_points_lineseg_regression(points=points3d)

        self.pose_optimization_line_vs_point_relevance = pose_optimization_line_vs_point_relevance


        self.dont_refine_ransac = debug_dont_refine_ransac
        self.debug_visualize_2d = debug_visualize_2d
        self.debug_visualize_3d = debug_visualize_3d

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.line_seg_detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE)


    def match_2d_line_segments(
            self, 
            lines_img1:np.ndarray, 
            lines_img2:np.ndarray,
            points_img1:np.ndarray, 
            points_img2:np.ndarray
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

        lines_1_point_distances_mask = np.array([line_to_points_distances_2d(line_seg_2d=l1, points=points_img1) < self.line_matching_max_dist_line_to_point_px for l1 in lines_img1])
        lines_2_point_distances_mask = np.array([line_to_points_distances_2d(line_seg_2d=l2, points=points_img2) < self.line_matching_max_dist_line_to_point_px for l2 in lines_img2])

        agreement_matrix = np.sum(lines_1_point_distances_mask[:, None, :] & lines_2_point_distances_mask[None, :, :], axis=2)

        best_l2_matching_values = np.full(lines_img2.shape[0], -1)
        best_l2_matchings = np.full(lines_img2.shape[0], -1)

        for i, l1 in enumerate(lines_img1):
            best_l2_index = np.argmax(agreement_matrix[i])

            if agreement_matrix[i,best_l2_index] < self.line_matching_min_number_supporting_points:
                continue

            if best_l2_matching_values[best_l2_index] <= agreement_matrix[i, best_l2_index]:
                best_l2_matching_values[best_l2_index] = agreement_matrix[i, best_l2_index]
                best_l2_matchings[best_l2_index] = i
        
        line_pairs = []
        for l2_idx, l1_idx in enumerate(best_l2_matchings): 
            if l1_idx >= 0:
                line_pairs.append([lines_img1[l1_idx], lines_img2[l2_idx]])
        return np.array(line_pairs) if len(line_pairs) > 0 else np.empty((0,2,4))


    def visualize_features_2d(
            self, 
            img1:np.ndarray, 
            img2:np.ndarray, 
            lines1_raw:np.ndarray, 
            lines2_raw:np.ndarray, 
            lines1_processed:np.ndarray, 
            lines2_processed:np.ndarray, 
            points1:np.ndarray, 
            points2:np.ndarray,
            line_pairs:np.ndarray,
        ):
        """
        :param img1: NxHxWx3 BGR image as numpy array
        :param img2: NxHxWx3 BGR image as numpy array
        :param lines1: Nx4 array of line segments
        :param lines2: Nx4 array of line segments
        :param line_pairs: Nx2x4 array of matched line pairs
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Display the raw lines
        lines_xy1_raw = [((line[0], line[1]), (line[2], line[3])) for line in lines1_raw]
        lc1_raw = LineCollection(lines_xy1_raw, linewidths=1, alpha=0.4, color = plt.cm.jet(np.linspace(0, 1, lines1_raw.shape[0])))
        axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        axes[0, 0].add_collection(lc1_raw)
        axes[0, 0].set_title("Raw lines cam1")

        lines_xy2_raw = [((line[0], line[1]), (line[2], line[3])) for line in lines2_raw]
        lc2_raw = LineCollection(lines_xy2_raw, linewidths=1, alpha=0.4, color = plt.cm.jet(np.linspace(0, 1, lines2_raw.shape[0])))
        axes[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        axes[0, 1].add_collection(lc2_raw)
        axes[0, 1].set_title("Raw lines cam1")

        # Display the processed lines
        lines_xy1 = [((line[0], line[1]), (line[2], line[3])) for line in lines1_processed]
        lc1 = LineCollection(lines_xy1, linewidths=1, alpha=0.4, color = plt.cm.jet(np.linspace(0, 1, lines1_processed.shape[0])))
        axes[1, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        axes[1, 0].add_collection(lc1)
        axes[1, 0].set_title("Matched & Joined lines cam1")

        matched_lines_1 = [((line[0], line[1]), (line[2], line[3])) for line in line_pairs[:, 0, :]]
        lc1m = LineCollection(matched_lines_1, linewidths=3, alpha=1.0, color = plt.cm.viridis(np.linspace(0, 1, line_pairs.shape[0])))
        axes[1, 0].add_collection(lc1m)


        lines_xy2 = [((line[0], line[1]), (line[2], line[3])) for line in lines2_processed]
        lc2 = LineCollection(lines_xy2, linewidths=1, alpha=0.4, color = plt.cm.jet(np.linspace(0, 1, lines2_processed.shape[0])))
        axes[1, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        axes[1, 1].add_collection(lc2)
        axes[1, 1].set_title("Matched & Joined lines cam2")

        matched_lines_2 = [((line[0], line[1]), (line[2], line[3])) for line in line_pairs[:, 1, :]]
        lc2m = LineCollection(matched_lines_2, linewidths=3, alpha=1.0,  color = plt.cm.viridis(np.linspace(0, 1, line_pairs.shape[0])))
        axes[1, 1].add_collection(lc2m)

        plt.show()
    
    def visualize_features_3d(
            self,
            point_cloud:np.ndarray,
            line_points_3d:np.ndarray,
            line_segments_3d:np.ndarray
        ):
        """
        Visualizes the inputs in 3d in relation to a base frame
        :param point_cloud: Nx3 numpy array of xyz-points
        :param line_points_3d: LxMx3 numpy array of xyz-points
        :param line_segments_3d: Lx6 numpy array of lines with each line: [x1, y1, z1, x2, y2, z2]
        """
        assert point_cloud.ndim == 2 and point_cloud.shape[-1] == 3
        assert line_points_3d.ndim == 3 and line_points_3d.shape[-1] == 3
        assert line_segments_3d.ndim == 2 and line_segments_3d.shape[-1] == 6

        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        pcd.paint_uniform_color([0, 0, 0])

        base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
        
        to_vis = [pcd, base_frame]

        if len(line_points_3d) > 0:
            pcd_lines = o3d.geometry.PointCloud()
            pcd_lines.points = o3d.utility.Vector3dVector(np.concatenate(line_points_3d, axis=0))
            pcd_lines.paint_uniform_color([1, 0, 0])
            to_vis.append(pcd_lines)

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line_segments_3d.reshape(-1,3))
        line_set.lines = o3d.utility.Vector2iVector(np.array([[i, i+1] for i in range(line_segments_3d.shape[0]*2-1) if i % 2 == 0]))
        to_vis.append(line_set)

        o3d.visualization.draw_geometries(to_vis, f"3D features visualization")



    def est_base_t_cam2(self,
            cam1_bgr_image:np.ndarray,
            base_xyz_image:np.ndarray,
            cam2_bgr_image: np.ndarray,
            point_cloud:np.ndarray,
            time_tracker:TimeTracker
        ) -> np.ndarray | None:
        time_tracker.reset_elapsed_time()

        image_points_cam1, image_points_cam2 = self.extract_and_match.get_matched_points(
            cv2.cvtColor(cam1_bgr_image, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(cam2_bgr_image, cv2.COLOR_BGR2RGB)
        )

        time_tracker.add_time_stamp("Extract and Match")


        world_obj_points = np.array([base_xyz_image[int(np.round(y)),int(np.round(x))] for x,y in image_points_cam1])

        if world_obj_points.shape[0] < min(5, self.initial_guess_min_number_inlier_after_ransac):
            return None

        success, r_img_t_obj, t_img_t_obj, inliers = cv2.solvePnPRansac(
            world_obj_points, image_points_cam2, self.cam2_intrinsic_mtx, None,
            iterationsCount = self.initial_guess_ransac_itterations,
            reprojectionError=self.initial_guess_ransac_reprojection_error,
            confidence = self.initial_guess_ransac_confidence,
            flags = cv2.SOLVEPNP_EPNP
        )
        if not success or len(inliers) < self.initial_guess_min_number_inlier_after_ransac:
           return None
        
        time_tracker.add_time_stamp("Pose estimation RAANSAC")

        cam2_t_base_pnp = np.eye(4)
        cam2_t_base_pnp[:3, :3] = cv2.Rodrigues(r_img_t_obj)[0]
        cam2_t_base_pnp[:3, 3] = t_img_t_obj.flatten()

        if self.dont_refine_ransac:
            return np.linalg.inv(cam2_t_base_pnp)


        # Optimize further using lines
        lines_img1_raw = self.line_seg_detector.detect(cv2.cvtColor(cam1_bgr_image, cv2.COLOR_BGR2GRAY))[0].squeeze(1)
        time_tracker.add_time_stamp("LSD image 1")

        lines_img2_raw = self.line_seg_detector.detect(cv2.cvtColor(cam2_bgr_image, cv2.COLOR_BGR2GRAY))[0].squeeze(1)
        time_tracker.add_time_stamp("LSD image 2")

        lines_img1 = self.line_refinement(lines_img1_raw)
        lines_img2 = self.line_refinement(lines_img2_raw)
        time_tracker.add_time_stamp("Line Refinement")


        line_pairs = self.match_2d_line_segments(
            lines_img1=lines_img1, 
            lines_img2=lines_img2,
            points_img1=image_points_cam1,
            points_img2=image_points_cam2
        )
        time_tracker.add_time_stamp("Match 2d line segments")


        matched_lines_2d = []
        matched_lines_3d = []
        
        xyz_points_4_lines = [line_seg_2d_to_3d_points(line_pair[0], base_xyz_image) for line_pair in line_pairs]

        for i, line_pair in enumerate(line_pairs):
            line_segment_3d = self.line_fitting_3d_method(xyz_points_4_lines[i])
            if line_segment_3d is not None:
                matched_lines_3d.append(line_segment_3d)
                matched_lines_2d.append(line_pair[1])
        matched_lines_2d = np.array(matched_lines_2d) if len(matched_lines_2d) > 0 else np.empty((0,4))
        matched_lines_3d = np.array(matched_lines_3d) if len(matched_lines_3d) > 0 else np.empty((0,6))

        time_tracker.add_time_stamp("Line 2d to 3d transformation")

        if self.debug_visualize_2d:
            self.visualize_features_2d(
                img1=cam1_bgr_image,
                img2=cam2_bgr_image,
                lines1_processed=lines_img1,
                lines2_processed=lines_img2,
                lines1_raw=lines_img1_raw,
                lines2_raw=lines_img2_raw,
                line_pairs = line_pairs,
                points1=[],
                points2=[]
            )
        if self.debug_visualize_3d:
            self.visualize_features_3d(
                point_cloud=point_cloud, 
                line_points_3d=[line_seg_2d_to_3d_points(line_pair[0], base_xyz_image) for line_pair in line_pairs],
                line_segments_3d=matched_lines_3d
            )
        
        time_tracker.reset_elapsed_time()

        inlier_indices = inliers.flatten()
        cam2_t_base_bundle_adjustment = optimize_pnpl(
            initial_cam_t_base=cam2_t_base_pnp.copy(),
            points_3d=world_obj_points[inlier_indices].copy(),
            points_2d=image_points_cam2[inlier_indices].copy(),
            intrinsic_cam_mat=self.cam2_intrinsic_mtx.copy(),
            lines_2d = matched_lines_2d,
            lines_3d = matched_lines_3d, 
            line_relevance=self.pose_optimization_line_vs_point_relevance
        )

        time_tracker.add_time_stamp("Bundle adjustment")

        return np.linalg.inv(cam2_t_base_bundle_adjustment)




if __name__ == "__main__":
    data = PredictionData.from_folder("/home/wmarx/AR-Headset-Localization-in-Robot-Scanned-Workspaces-A-Benchmark-Pipeline/data_preprocessing/out_data")
    predictor = LinePredictor(
        data.headset_intrinsics, 
        extract_and_match=ExtractAndLightGlue(),#ExtractAndMatchLoMa(loma_variant="LoMaB"),
        debug_visualize_2d=False, 
        pose_optimization_line_vs_point_relevance=0.5,
        debug_dont_refine_ransac=False,
        line_fitting_3d_use_ransaac=False
    )
    grader = OnePredictorOneDatasetGrader(predictor=predictor, data=data)
    
    #grader.visualize_predictions()
    
    print(f"median rot error: {np.round(np.rad2deg(grader.median_rotational_error()), 2)} degrees")
    print(f"median translational error: {np.round(grader.median_translational_error()*1000, 1)} mm")
    print(f"avg. sub median rot error: {np.round(np.rad2deg(grader.average_sub_median_rotational_error()), 2)} degrees")
    print(f"avg. sub median translational error: {np.round(grader.average_sub_median_translat_error()*1000, 1)} mm")
    print(f"sucess_ratio: {np.round(grader.sucess_ratio(),2)}")

    tt = TimeTracker()
    for i in range(10):
        grader = OnePredictorOneDatasetGrader(predictor=predictor, data=data, time_tracker=tt)
    tt.print_report()
    