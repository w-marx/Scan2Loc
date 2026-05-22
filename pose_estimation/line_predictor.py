import cv2
import numpy as np
from predictor_handling import *
from extractors_and_matchers import *
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from skimage.draw import line

from pose_estimation.pnpl_optimizer import optimize_pnpl


class UnionFind:
    def __init__(self, n:int):
        self.parent = np.arange(n)
        self.rank = np.zeros(n, dtype=int)

    def find(self, e:int):
        while self.parent[e] != e:
            self.parent[e] = self.parent[self.parent[e]]
            e = self.parent[e]
        return e

    def union(self, e1:int, e2:int):
        """
        Union by Rank between a & b, if they arent already in the same Cluster
        """
        representative_e1 = self.find(e1)
        representative_e2 = self.find(e2)

        if representative_e1 == representative_e2:
            return

        if self.rank[representative_e1] < self.rank[representative_e2]:
            self.parent[representative_e1] = representative_e2
        elif self.rank[representative_e1] > self.rank[representative_e2]:
            self.parent[representative_e2] = representative_e1
        else:
            self.parent[representative_e2] = representative_e1
            self.rank[representative_e1] += 1
    
    def return_clusters(self)->list[list[int]]:
        """
        Returns a list of all clusters (each cluster as a list of indices)
        """
        clusters = {}
        for idx, parent in enumerate(self.parent):
            representative = self.find(idx)
            if representative in clusters:
                clusters[representative].append(idx)
            else:
                clusters[representative] = [idx]
        return list(clusters.values())


class LinePredictor(PosePredictor):
    def __init__(
            self,
            cam2_mtx:np.ndarray,
            extract_and_match:ExtractAndMatch = ExtractAndMatchLoMa(),
            min_number_inlier:int = 6,
            ransac_itterations:int = 10000,
            ransac_reprojection_error:float = 5.0,
            ransac_confidence:float = 0.99,
            min_line_length_px:float = 50,
            line_fitting_iterations:int = 1000,
            line_fitting_inlier_distance:float = 0.005,
            line_vs_point_relevance:float = 0.5,
            debug_dont_refine_ransac:bool = False,
            debug_visualize_2d:bool = False,
            debug_visualize_3d:bool = False
        ):
        super().__init__()
        self.cam2_mtx = cam2_mtx
        self.extract_and_match = extract_and_match
        self.min_number_inlier = min_number_inlier
        self.ransac_itterations = ransac_itterations
        self.ransac_reprojection_error = ransac_reprojection_error
        self.ransac_confidence = ransac_confidence
        self.min_line_length_px = min_line_length_px

        # Line matching
        self.distance_for_same = 5
        self.min_number_agreeing_points = 2

        # Line fitting
        from torch_ransac3d.line import line_fit
        self.line_fit_3d = line_fit
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.line_fitting_iterations = line_fitting_iterations
        self.line_fitting_inlier_distance = line_fitting_inlier_distance
        self.line_vs_point_relevance = line_vs_point_relevance

        # Dont use Gradient descend refinement
        self.dont_refine_ransac = debug_dont_refine_ransac
        self.vis_features_2d = debug_visualize_2d
        self.vis_features_3d = debug_visualize_3d


    def join_close_line_segments(
            self, 
            line_segs_2d:np.ndarray,
            max_angle_diff:float = 5,
            max_midpoint_dist:float = 5,
            max_endpoint_dist:float = 15
        ) -> np.ndarray:
        """
        Joins lines with close endpoints & low angles
        :param line_segs_2d: Nx4 numpy array of the structure: (x1, y1, x2, y2)
        :param max_angle_diff: maximum angle between 2 lines to be joined (in pixels)
        :param max_midpoint_dist: maximum angle between 2 midpoints
        :param max_endpoint_dist: maximum distance between 2 line-endpoints to be joined (in degrees)
        :return Mx4 numpy array of the same structure with M <= N
        """
        angle_thresh_rad = np.deg2rad(max_angle_diff)
        line_segs_2d = line_segs_2d.copy()

        dxy_s = line_segs_2d[:,2:]-line_segs_2d[:,0:2]
        angle_s = np.atan2(dxy_s[:, 1], dxy_s[:,0])
        length_s = np.linalg.norm(dxy_s, axis = 1)
        mid_points = line_segs_2d[:, 0:2] + dxy_s[:]/2

        mostly_horizontal = np.abs(dxy_s[:, 0]) > np.abs(dxy_s[:, 1])
        flip_endpoints = np.logical_or(
            mostly_horizontal & (line_segs_2d[:, 0] > line_segs_2d[:, 2]),
            ~mostly_horizontal & (line_segs_2d[:, 1] > line_segs_2d[:, 3])
        )
        line_segs_2d[flip_endpoints] = line_segs_2d[flip_endpoints][:, [2,3,0,1]]


        merge_adj_list = [] # This is missing the backward edges
        for i1,l1 in enumerate(line_segs_2d):
            poss_indices = np.arange(min(i1+1, line_segs_2d.shape[0]), line_segs_2d.shape[0])

            # Check if angle is similar
            angles_to_other_lines_unnorm = np.abs(angle_s[poss_indices]-angle_s[i1])
            angle_diff_s = np.minimum(angles_to_other_lines_unnorm, np.pi - angles_to_other_lines_unnorm)
            poss_indices = poss_indices[angle_diff_s < angle_thresh_rad]

            # Check if midpoint close to the line
            d1 = self.line_to_point_distances(line_seg_2d=l1, points=mid_points[poss_indices])
            d2 = np.array([self.line_to_point_distances(line, np.array([mid_points[i1]]))[0] for line in line_segs_2d[poss_indices]])
            poss_indices = poss_indices[(d1 < max_midpoint_dist) | (d2 < max_midpoint_dist)]

            # Extract endpoints
            l1_ep1, l1_ep2 = l1[:2], l1[2:]
            l_other_ep1, l_other_ep2 = line_segs_2d[poss_indices, :2], line_segs_2d[poss_indices, 2:]


            # Check for overlap
            if mostly_horizontal[i1]:
                overlaps = (
                    (l1_ep1[0] <= l_other_ep2[:,0]) &
                    (l_other_ep1[:,0] <= l1_ep2[0])
                )

            else:
                overlaps = (
                    (l1_ep1[1] <= l_other_ep2[:,1]) &
                    (l_other_ep1[:,1] <= l1_ep2[1])
                )

            gap_sq = np.sum((l_other_ep1-l1_ep2)**2, axis = 1)

            merge_mask = overlaps | (gap_sq < max_endpoint_dist**2)

            merge_adj_list.append(poss_indices[merge_mask])

        union_find = UnionFind(line_segs_2d.shape[0])
        for i1, adjecent_idx in enumerate(merge_adj_list):
            for i2 in adjecent_idx:
                union_find.union(i1, i2)
        merged_lines = np.array([
            self.merge_line_seg_cluster_into_one(line_segs_2d[np.array(line_cluster)])
            for line_cluster in union_find.return_clusters()
        ])

        return merged_lines
        
    
    def remove_short_line_segments(self,line_segs_2d:np.ndarray, min_line_length_px:int = 5) -> np.ndarray:
        """
        Removes all lines under the min_line_length threshhold
        :param line_segs_2d: Nx4 numpy array of the structure: (x1, y1, x2, y2)
        :param min_line_length_px, the threshhold
        :return Mx4 numpy array of the same structure with M <= N
        """
        lengths = (line_segs_2d[:,2]-line_segs_2d[:, 0])**2+(line_segs_2d[:,3]-line_segs_2d[:,1])**2
        return line_segs_2d[lengths > (min_line_length_px**2)]
    
    def line_to_point_distances(self, line_seg_2d:np.ndarray, points:np.ndarray) -> np.ndarray:
        """
        Computes the distance between each point and the line segment
        :param line_seg_2d: Numpy array of the structure: [x1, y1, x2, y2]
        :param points: Nx2 numpy array of the points [[xi, yi], ...]
        :return numpy array of length N with the distances
        """

        x1, y1, x2, y2 = line_seg_2d
        dx, dy = x2-x1, y2-y1

        seg_length_sq = dx*dx + dy*dy
        if seg_length_sq < 1e-6:
            return np.linalg.norm(points-np.array([x1, y1]), axis = 1) 


        t = ((points[:, 0]-x1) * dx + (points[:, 1]-y1) * dy)/ seg_length_sq
        t = np.clip(t, 0.0, 1.0)
        
        projections = np.column_stack([x1 + t * dx, y1 + t * dy])
        
        return np.linalg.norm(projections-points, axis = 1)

    def get_line_pairs(self, lines_img1, lines_img2 ,points_img1, points_img2)->np.ndarray:
        """
        Returns the best matching line pairs
        :param lines_img1: Nx4 numpy array of the structure: (x1, y1, x2, y2), of lines in image 1
        :param lines_img2: Nx4 numpy array of the structure: (x1, y1, x2, y2), of lines in image 2
        :param points_img1: Nx2 numpy array of points in image 1, p_i in points_img1 has to be matched to p_i in points_img2
        :param points_img2: Nx2 numpy array of points in image 2
        :return Mx2x4 numpy array of M linepairs
        """

        if lines_img1.shape[0] == 0 or lines_img2.shape[0] == 0 or points_img1.shape[0] == 0 or points_img2.shape[0] == 0:
            return np.empty((0,2,4))

        lines_1_point_distances_mask = np.array([self.line_to_point_distances(line_seg_2d=l1, points=points_img1) < self.distance_for_same for l1 in lines_img1])
        lines_2_point_distances_mask = np.array([self.line_to_point_distances(line_seg_2d=l2, points=points_img2) < self.distance_for_same for l2 in lines_img2])

        agreement_matrix = np.sum(lines_1_point_distances_mask[:, None, :] & lines_2_point_distances_mask[None, :, :], axis=2)

        best_l2_matching_values = np.full(lines_img2.shape[0], -1)
        best_l2_matchings = np.full(lines_img2.shape[0], -1)

        for i, l1 in enumerate(lines_img1):
            best_l2_index = np.argmax(agreement_matrix[i])

            if agreement_matrix[i,best_l2_index] < self.min_number_agreeing_points:
                continue

            if best_l2_matching_values[best_l2_index] <= agreement_matrix[i, best_l2_index]:
                best_l2_matching_values[best_l2_index] = agreement_matrix[i, best_l2_index]
                best_l2_matchings[best_l2_index] = i
        
        line_pairs = []
        for l2_idx, l1_idx in enumerate(best_l2_matchings): 
            if l1_idx >= 0:
                line_pairs.append([lines_img1[l1_idx], lines_img2[l2_idx]])
        return np.array(line_pairs) if len(line_pairs) > 0 else np.empty((0,2,4))
    
    def line_seg_2d_to_3d_points(self, line_seg_2d, xyz_image):
        """
        Creates a 3d point cloud of the points the line segment passes through on the xyz_image
        :param line_seg_2d: [x1, y1, x2, y2] numpy array
        :param xyz_image: HxWx3 numpy array that has a 3d point at each pixel
        """
        assert line_seg_2d.shape == (4,), f"2D line seg shape: {line_seg_2d.shape}"
        assert xyz_image.ndim == 3 and xyz_image.shape[-1] == 3

        x1, y1, x2, y2 = np.round(line_seg_2d).astype(int)

        row_cords, col_cords = line(y1, x1, y2, x2)
        valid_points_mask = (0 <= row_cords) & (row_cords < xyz_image.shape[0]) & (0 <= col_cords) & (col_cords < xyz_image.shape[1])

        return np.empty((0,3)) if len(row_cords) < 1 else xyz_image[row_cords[valid_points_mask], col_cords[valid_points_mask]]
    
    def merge_line_seg_cluster_into_one(self,line_segs_2d:np.ndarray)->np.ndarray:
        """
        Joins multiple lines into one
        :param line_segs_2d: Nx4 numpy array of the structure: (x1, y1, x2, y2)
        :return a numpy array: (x1, y1, x2, y2) of the new line
        """
        assert line_segs_2d.ndim == 2 and line_segs_2d.shape[-1] == 4
        assert line_segs_2d.shape[0] > 0

        if line_segs_2d.shape[0] == 1:
            return line_segs_2d[0]

        xy_points = line_segs_2d.reshape(-1,2)
        xy_center = np.mean(xy_points, axis = 0)
        

        _, _, Vt = np.linalg.svd(xy_points - xy_center)

        direction = Vt[0]/np.linalg.norm(Vt[0])

        dists_to_center = np.dot(xy_points-xy_center, direction)

        start_p = xy_center + np.min(dists_to_center) * direction
        end_p = xy_center + np.max(dists_to_center)*direction

        return np.array([start_p[0], start_p[1], end_p[0], end_p[1]])


    def line_segment_regression_3d(self,xyz_points:np.ndarray)->np.ndarray|None:
        """
        :param xyz_points: An Nx3 numpy array of points
        :return an numpy array: [x1, y1, z1, x2, y2, z2] that represents the fitted line segment or None if it couldnt be fitted
        """
        assert xyz_points.ndim == 2 and xyz_points.shape[-1] == 3, f"xyz_points shape: {xyz_points.shape}"

        if xyz_points.shape[0] < 2:
            return None
        
        infinite_line = self.line_fit_3d(
            pts = xyz_points,
            thresh = self.line_fitting_inlier_distance,
            max_iterations = self.line_fitting_iterations
        )

        inliers = xyz_points[infinite_line.inliers]
        direction = (infinite_line.direction / np.linalg.norm(infinite_line.direction)).cpu().numpy()
        p0 = infinite_line.point.cpu().numpy()

        dists_to_p0 = np.dot(inliers-p0, direction)

        start_p = p0 + np.min(dists_to_p0) * direction
        end_p = p0 + np.max(dists_to_p0)*direction

        return np.array([start_p[0], start_p[1], start_p[2], end_p[0], end_p[1], end_p[2]])



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



        lines_xy1 = [((line[0], line[1]), (line[2], line[3])) for line in lines1_processed]
        lc1 = LineCollection(lines_xy1, linewidths=1, alpha=0.4, color = plt.cm.jet(np.linspace(0, 1, lines1_processed.shape[0])))
        axes[1, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        axes[1, 0].add_collection(lc1)
        axes[1, 0].set_title("Matched & Joined lines cam1")

        matched_lines_1 = [((line[0], line[1]), (line[2], line[3])) for line in line_pairs[:, 0, :]]
        lc1m = LineCollection(matched_lines_1, linewidths=5, alpha=1.0, color = plt.cm.viridis(np.linspace(0, 1, line_pairs.shape[0])))
        axes[1, 0].add_collection(lc1m)


        lines_xy2 = [((line[0], line[1]), (line[2], line[3])) for line in lines2_processed]
        lc2 = LineCollection(lines_xy2, linewidths=1, alpha=0.4, color = plt.cm.jet(np.linspace(0, 1, lines2_processed.shape[0])))
        axes[1, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        axes[1, 1].add_collection(lc2)
        axes[1, 1].set_title("Matched & Joined lines cam2")

        matched_lines_2 = [((line[0], line[1]), (line[2], line[3])) for line in line_pairs[:, 1, :]]
        lc2m = LineCollection(matched_lines_2, linewidths=5, alpha=1.0,  color = plt.cm.viridis(np.linspace(0, 1, line_pairs.shape[0])))
        axes[1, 1].add_collection(lc2m)

        plt.show()
    
    def visualize_features_3d(
            self,
            point_cloud:np.ndarray,
            line_points_3d:np.ndarray,
            line_segments_3d:np.ndarray
        ):
        """
        :param point_cloud: Nx3 numpy array of xyz-points
        :param line_points_3d: LxMx3 numpy array of xyz-points
        :param line_segments_3d: Lx6 numpy array of lines with each line: [x1, y1, z1, x2, y2, z2]
        """
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
                        ) -> np.ndarray | None:

        image_points_cam1, image_points_cam2 = self.extract_and_match.get_matched_points(
            cv2.cvtColor(cam1_bgr_image, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(cam2_bgr_image, cv2.COLOR_BGR2RGB)
        )

        world_obj_points = np.array([base_xyz_image[int(np.round(y)),int(np.round(x))] for x,y in image_points_cam1])

        if world_obj_points.shape[0] < 5:
            return None

        success, r_img_t_obj, t_img_t_obj, inliers = cv2.solvePnPRansac(
            world_obj_points, image_points_cam2, self.cam2_mtx, None,
            iterationsCount = self.ransac_itterations,
            reprojectionError=self.ransac_reprojection_error,
            confidence = self.ransac_confidence,
            flags = cv2.SOLVEPNP_EPNP
        )
        if not success or len(inliers) < self.min_number_inlier:
           return None

        cam2_t_base_pnp = np.eye(4)
        cam2_t_base_pnp[:3, :3] = cv2.Rodrigues(r_img_t_obj)[0]
        cam2_t_base_pnp[:3, 3] = t_img_t_obj.flatten()


        if self.dont_refine_ransac:
            return np.linalg.inv(cam2_t_base_pnp)

        # Optimize further using lines

        lsd = cv2.createLineSegmentDetector(0)
        lines_img1_all = lsd.detect(cv2.cvtColor(cam1_bgr_image, cv2.COLOR_BGR2GRAY))[0].squeeze(1)
        lines_img2_all = lsd.detect(cv2.cvtColor(cam2_bgr_image, cv2.COLOR_BGR2GRAY))[0].squeeze(1)

        lines_img1 = self.remove_short_line_segments(
            line_segs_2d=lines_img1_all,
            min_line_length_px=5
        )
        lines_img2 = self.remove_short_line_segments(
            line_segs_2d=lines_img2_all,
            min_line_length_px=5
        )

        lines_img1 = self.join_close_line_segments(
            line_segs_2d=lines_img1,
            max_angle_diff=5,
            max_endpoint_dist=5,
            max_midpoint_dist=15
        )

        lines_img2= self.join_close_line_segments(
            line_segs_2d=lines_img2,
            max_angle_diff=5,
            max_endpoint_dist=5,
            max_midpoint_dist=15
        )

        lines_img1 = self.remove_short_line_segments(
            line_segs_2d=lines_img1_all,
            min_line_length_px=50
        )
        lines_img2 = self.remove_short_line_segments(
            line_segs_2d=lines_img2_all,
            min_line_length_px=50
        )


        line_pairs = self.get_line_pairs(
            lines_img1=lines_img1, 
            lines_img2=lines_img2,
            points_img1=image_points_cam1,
            points_img2=image_points_cam2
        )


        matched_lines_2d = []
        matched_lines_3d = []
        for line_pair in line_pairs:
            line_segment_3d = self.line_segment_regression_3d(self.line_seg_2d_to_3d_points(line_pair[0], base_xyz_image))
            if line_segment_3d is not None:
                matched_lines_3d.append(line_segment_3d)
                matched_lines_2d.append(line_pair[1])
        matched_lines_2d = np.array(matched_lines_2d) if len(matched_lines_2d) > 0 else np.empty((0,4))
        matched_lines_3d = np.array(matched_lines_3d) if len(matched_lines_3d) > 0 else np.empty((0,6))



        if self.vis_features_2d:
            self.visualize_features_2d(
                img1=cam1_bgr_image,
                img2=cam2_bgr_image,
                lines1_processed=lines_img1_joined_1,
                lines2_processed=lines_img2_joined_1,
                lines1_raw=lines_img1_all,
                lines2_raw=lines_img2_all,
                line_pairs = line_pairs,
                points1=[],
                points2=[]
            )
        if self.vis_features_3d:
            self.visualize_features_3d(
                point_cloud=point_cloud, 
                line_points_3d=[self.line_seg_2d_to_3d_points(line_pair[0], base_xyz_image) for line_pair in line_pairs],
                line_segments_3d=matched_lines_3d
            )

        inlier_indices = inliers.flatten()
        cam2_t_base_bundle_adjustment = optimize_pnpl(
            initial_cam_t_base=np.round(cam2_t_base_pnp, 1),#cam2_t_base_pnp.copy(),
            points_3d=world_obj_points[inlier_indices].copy(),
            points_2d=image_points_cam2[inlier_indices].copy(),
            intrinsic_cam_mat=self.cam2_mtx.copy(),
            lines_2d = matched_lines_2d,
            lines_3d = matched_lines_3d, 
            line_relevance=self.line_vs_point_relevance
        )

        #If optimization does nothing with np.round(cam2_t_base_pnp, 1): 
        #median rot error: 10.625315285964422 degrees
        #median translational error: 698.5760617300945 mm
        #avg. sub median rot error: 9.417927161060378 degrees
        # avg. sub median translational error: 682.4970834620548 mm
        return np.linalg.inv(cam2_t_base_bundle_adjustment)




if __name__ == "__main__":
    data = PredictionData.from_folder("/home/wmarx/AR-Headset-Localization-in-Robot-Scanned-Workspaces-A-Benchmark-Pipeline/data_preprocessing/out_data")
    predictor = LinePredictor(
        data.headset_intrinsics, 
        extract_and_match=ExtractAndMatchLoMa(loma_variant="LoMaB"),
        debug_visualize_2d=False, 
        line_vs_point_relevance=1.0,
        debug_dont_refine_ransac=False
    )
    grader = OnePredictorOneDatasetGrader(predictor=predictor, data=data)
    
    grader.visualize_predictions()
    
    print(f"median rot error: {np.rad2deg(grader.median_rotational_error())} degrees")
    print(f"median translational error: {grader.median_translational_error()*1000} mm")
    print(f"avg. sub median rot error: {np.rad2deg(grader.average_sub_median_rotational_error())} degrees")
    print(f"avg. sub median translational error: {grader.average_sub_median_translat_error()*1000} mm")
    