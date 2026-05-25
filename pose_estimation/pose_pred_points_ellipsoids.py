import cv2
import numpy as np
import open3d as o3d
from predictor_handling import *
from extractors_and_matchers import *
from image_augmentation import *
import matplotlib.pyplot as plt


def segment_3d_point_cloud_into_objects(
        point_cloud:np.ndarray,
        min_object_num_points:int = 50,
    )->list[np.ndarray]:
    assert point_cloud.ndim == 2 and point_cloud.shape[-1] == 3

    if point_cloud.shape[0] == 0:
        return []
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    
    from sklearn.neighbors import kneighbors_graph
    from sklearn.cluster import AgglomerativeClustering

    connectivity = kneighbors_graph(
        point_cloud,
        n_neighbors=10,
        include_self=False
    )

    clustering = AgglomerativeClustering(
        linkage="complete",
        distance_threshold=0.05,
        n_clusters=None,
        connectivity=connectivity
    )

    labels = clustering.fit_predict(point_cloud)

    unique_labels = np.unique(labels)

    object_point_clouds = []
    for label in unique_labels:
        cluster_points = point_cloud[labels == label]
        if cluster_points.shape[0] > min_object_num_points:
            object_point_clouds.append(cluster_points)
    
    return object_point_clouds




class EllipsoidPredictor(PosePredictor):
    def __init__(
            self,
            cam2_intrinsic_mtx:np.ndarray,
            point_cloud:np.ndarray,
            extract_and_match:ExtractAndMatch = ExtractAndMatchLoMa(),
            ransac_config:RansacPoseEstimationConfig = pose_estimation_ransaac_config_precise,
            debug_visualize_init_result:bool = True
        ):
        super().__init__()
        self.cam2_intrinsic_mtx = cam2_intrinsic_mtx
        self.extract_and_match = extract_and_match
        self.initial_raansac_guess_config = ransac_config

        self.object_segmented_point_cloud = segment_3d_point_cloud_into_objects(
            point_cloud=point_cloud
        )

        if debug_visualize_init_result:
            self.visualize_3d()
    
    def visualize_3d(self):
        pcd_s = self.object_segmented_point_cloud

        base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)

        print(f"number of objects: {len(pcd_s)}")

        colors = plt.cm.jet(np.linspace(0,1, len(pcd_s)))
        to_vis = [base_frame]
        for i, pcd_np in enumerate(pcd_s):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_np)
            pcd.paint_uniform_color(colors[i][:3])
            to_vis.append(pcd)

        o3d.visualization.draw_geometries(to_vis, f"3D features visualization")

    def est_base_t_cam2(self,
                        cam1_bgr_image:np.ndarray,
                        base_xyz_image:np.ndarray,
                        cam2_bgr_image: np.ndarray,
                        time_tracker:TimeTracker
                        ) -> np.ndarray | None:
        
        time_tracker.reset_elapsed_time()
        
        cam1_rgb_image = cv2.cvtColor(cam1_bgr_image, cv2.COLOR_BGR2RGB)
        cam2_rgb_image = cv2.cvtColor(cam2_bgr_image, cv2.COLOR_BGR2RGB)

        image_points_cam1, image_points_cam2 = self.extract_and_match.get_matched_points(
            img1_rgb=cam1_rgb_image, img2_rgb=cam2_rgb_image, plot_results = False
        )
        time_tracker.add_time_stamp("Extract and Match")


        world_obj_points = np.array([base_xyz_image[int(np.round(y)),int(np.round(x))] for x,y in image_points_cam1])

        cam2_t_base_pnp__inliers = estimate_point_pose_ransac(
            img_points=image_points_cam2, 
            world_points=world_obj_points, 
            intrinsic_matrix=self.cam2_intrinsic_mtx, 
            config=self.initial_raansac_guess_config
        )
        if cam2_t_base_pnp__inliers is None:
            return None
        cam2_t_base_pnp, inliers = cam2_t_base_pnp__inliers
    
        return np.linalg.inv(cam2_t_base_pnp)




if __name__ == "__main__":
    data = PredictionData.from_folder("/home/wmarx/AR-Headset-Localization-in-Robot-Scanned-Workspaces-A-Benchmark-Pipeline/data_preprocessing/out_data")
    predictor = EllipsoidPredictor(
        data.headset_intrinsics, 
        point_cloud=data.point_cloud,
        extract_and_match=ExtractAndLightGlue(),
        ransac_config=pose_estimation_ransaac_config_precise
    )
    grader = OnePredictorOneDatasetGrader(predictor=predictor, data=data)
    
    #grader.visualize_predictions()
    
    print(f"median rot error: {np.round(np.rad2deg(grader.median_rotational_error()), 2)} degrees")
    print(f"median translational error: {np.round(grader.median_translational_error()*1000, 1)} mm")
    print(f"avg. sub median rot error: {np.round(np.rad2deg(grader.average_sub_median_rotational_error()), 2)} degrees")
    print(f"avg. sub median translational error: {np.round(grader.average_sub_median_translat_error()*1000, 1)} mm")
    print(f"sucess_ratio: {np.round(grader.sucess_ratio(),2)}")

    #tt = TimeTracker()
    #for i in range(10):
    #    grader = OnePredictorOneDatasetGrader(predictor=predictor, data=data, time_tracker=tt)
    #tt.print_report()