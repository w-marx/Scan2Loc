import cv2
import numpy as np
import torch
import open3d as o3d
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd, numpy_image_to_torch
from lightglue import viz2d
from predictor_handling import *



class NoExtrasPredictor(PosePredictor):
    def __init__(self, cam2_mtx:np.ndarray):
        super().__init__()
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()
        self.matcher = LightGlue(features='superpoint').eval().cuda()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cam2_mtx = cam2_mtx

    def est_cam2_t_cam1(self,
                        cam1_bgr_image:np.ndarray,
                        cam1_xyz_image:np.ndarray,
                        cam2_bgr_image: np.ndarray,
                        point_cloud:np.ndarray,
                        ) -> np.ndarray | None:



        cam1_image = numpy_image_to_torch(cv2.cvtColor(cam1_bgr_image, cv2.COLOR_BGR2RGB))
        cam2_image = numpy_image_to_torch(cv2.cvtColor(cam2_bgr_image, cv2.COLOR_BGR2RGB))

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        feats_cam1 = self.extractor.extract(cam1_image.to(self.device))
        feats_cam2 = self.extractor.extract(cam2_image.to(self.device))

        matches12 = self.matcher({'image0': feats_cam1, 'image1': feats_cam2, })
        feats_cam1, feats_cam2, matches12 = [rbd(x) for x in [feats_cam1, feats_cam2, matches12]]  # remove batch dimension

        matches_cpu = matches12['matches'].cpu()
        feats_cam1_cpu_keypoints = feats_cam1['keypoints'].cpu()
        feats_cam2_cpu_keypoints = feats_cam2['keypoints'].cpu()

        image_points_cam1_cpu = feats_cam1_cpu_keypoints[matches_cpu[..., 0]]  # coordinates in image #1, shape (K,2)
        image_points_cam2_cpu = feats_cam2_cpu_keypoints[matches_cpu[..., 1]]  # coordinates in image #0, shape (K,2)

        object_points_cam1 = np.array([cam1_xyz_image[int(x),int(y)] for x,y in image_points_cam1_cpu.numpy()])
        if object_points_cam1.shape[0] < 6:
            print(f"found only {object_points_cam1.shape[0]} points, cant estimate pose")
            return None


        success, rotation_vector, translation_vector = cv2.solvePnP(
            object_points_cam1, image_points_cam2_cpu.numpy(), self.cam2_mtx, None
        )

        if not success:
           return None
        
        axes = viz2d.plot_images([cam1_image, cam2_image])
        viz2d.plot_matches(image_points_cam1_cpu, image_points_cam2_cpu, color="lime", lw=0.2)
        viz2d.add_text(0, f'Stop after {matches12["stop"]} layers', fs=20)
        viz2d.save_plot("./LG OUT 1")
        kpc0, kpc1 = viz2d.cm_prune(matches12["prune0"]), viz2d.cm_prune(matches12["prune1"])
        viz2d.plot_images([cam1_image, cam2_image])
        viz2d.plot_keypoints([feats_cam1_cpu_keypoints, feats_cam2_cpu_keypoints], colors=[kpc0, kpc1], ps=10)
        viz2d.save_plot("./LG OUT 2")

        transformation = np.eye(4)
        transformation[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
        transformation[:3, 3] = translation_vector.flatten()
        
        return transformation




if __name__ == "__main__":
    data = PredictionData.from_folder("../processed_datasets/r3_small_aruco1")
    predictor = NoExtrasPredictor(data.headset_intrinsics)
    predicted_poses = run_predictions(data, predictor)

    for i, pose in enumerate(predicted_poses):
        print(f"predicted poses {i}: \n\n{pose} \n")
    
    if data.robot_base_t_headset is not None:
        r_err_s, t_err_s = grade_predictions(predictions=predicted_poses, actual=data.robot_base_t_headset)
        for i, (re, te) in enumerate(zip(r_err_s, t_err_s)):
            if re is not None:        
                print(f"rot_error: {np.round(re,1)}° \n\n t_err: {np.round(te,1)}mm")
                if te > 5e6:
                    predicted_poses[i] = None
                    print(f"removed datapoint {i} with outlier error")
    
        median_rot_error = np.median(np.array([x for x in r_err_s if x is not None]), axis = 0)
        median_translat_error = np.median(np.array([x for x in t_err_s if x is not None]), axis = 0)

        print(f"median rot error: {median_rot_error} degrees")
        print(f"median translational error: {median_translat_error}")
    
    #Visualisation:

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data.point_cloud)

    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
    robot_camera_s = []
    headset_location_s = []
    for idx, (b_t_c, b_t_h) in enumerate(zip(data.robot_base_t_robot_camera_s, predicted_poses)):
        if b_t_h is None:
            continue

        robot_camera_s.append(create_3d_camera(
            base_t_camera=b_t_c,
            intrinsics=data.robot_bgr_intrinsics,
            hxw_img=data.robot_bgr_images[0],
            scale=0.1
        ))
    
        headset_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        headset_frame.transform(b_t_h)
        headset_location_s.append(headset_frame)
        #headset_location_s.append(create_3d_camera(
        #    base_t_camera=b_t_h,
        #    intrinsics=data.headset_intrinsics,
        #    hxw_img=data.headset_bgr_image,
        #    scale=0.2
        #))

    to_vis = [pcd,base_frame]+headset_location_s+robot_camera_s

    if data.robot_base_t_headset is not None:
        headset_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        to_vis.append(create_3d_camera(
            base_t_camera=data.robot_base_t_headset,
            intrinsics=data.headset_intrinsics,
            hxw_img=data.headset_bgr_image,
            scale=0.3
        ))
        headset_frame.transform(data.robot_base_t_headset)
        to_vis.append(headset_frame)

    o3d.visualization.draw_geometries(to_vis, f"Predictions visualization")
    