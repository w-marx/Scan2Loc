import cv2
import numpy as np
import torch
import open3d as o3d
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd, numpy_image_to_torch
from lightglue import viz2d
from predictor_handling import *
import matplotlib.pyplot as plt



class NoExtrasPredictor(PosePredictor):
    def __init__(
            self, 
            cam2_mtx:np.ndarray,
            use_rotation_augmentations:bool = True,
            min_number_inlier:int = 6,
            ransac_itterations:int = 10000,
            ransac_reprojection_error:float = 5.0,
            ransac_confidence:float = 0.99
        ):
        super().__init__()
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()
        self.matcher = LightGlue(features='superpoint').eval().cuda()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cam2_mtx = cam2_mtx
        self.min_number_inlier = min_number_inlier
        self.ransac_itterations = ransac_itterations
        self.ransac_reprojection_error = ransac_reprojection_error
        self.ransac_confidence = ransac_confidence
        self.use_rotation_augmentations = use_rotation_augmentations

    def est_base_t_cam2(self,
                        cam1_bgr_image:np.ndarray,
                        base_xyz_image:np.ndarray,
                        cam2_bgr_image: np.ndarray,
                        point_cloud:np.ndarray,
                        plot_matchings:bool = True
                        ) -> np.ndarray | None:

        h, w = cam2_bgr_image.shape[:2]


        rotation_augmentations = [("identity",lambda img: img, lambda img_points:img_points)]
        if self.use_rotation_augmentations:
            rotation_augmentations += [
                ("rot 180°",lambda img: cv2.rotate(img, cv2.ROTATE_180), lambda img_points: np.array([[w-x, h-y] for x,y in img_points])),
            ]

        
        augmentation_options_names = []
        world_obj_points_options = []
        image_points_cam2_options = []

        for (r_aug_name, r_forward_t, r_backward_t) in rotation_augmentations:
            cam1_image = numpy_image_to_torch(cv2.cvtColor(cam1_bgr_image, cv2.COLOR_BGR2RGB))
            cam2_image = numpy_image_to_torch(cv2.cvtColor(r_forward_t(cam2_bgr_image), cv2.COLOR_BGR2RGB))

            feats_cam1 = self.extractor.extract(cam1_image.to(self.device))
            feats_cam2 = self.extractor.extract(cam2_image.to(self.device))

            matches12 = self.matcher({'image0': feats_cam1, 'image1': feats_cam2, })
            feats_cam1, feats_cam2, matches12 = [rbd(x) for x in [feats_cam1, feats_cam2, matches12]]

            feats_cam1_keypoints = feats_cam1['keypoints']
            feats_cam2_keypoints = feats_cam2['keypoints']
            image_points_cam1_cpu = feats_cam1_keypoints[matches12['matches'][..., 0]].cpu()
            image_points_cam2_cpu = feats_cam2_keypoints[matches12['matches'][..., 1]].cpu()

            if plot_matchings:
                axes = viz2d.plot_images([cv2.cvtColor(cam1_bgr_image, cv2.COLOR_BGR2RGB), cv2.cvtColor(r_forward_t(cam2_bgr_image), cv2.COLOR_BGR2RGB)])
                viz2d.plot_matches(
                    feats_cam1_keypoints[matches12['matches'][..., 0]],
                    feats_cam2_keypoints[matches12['matches'][..., 1]],lw=0.1
                )
                plt.show()


            world_obj_points = np.array([base_xyz_image[int(np.round(y)),int(np.round(x))] for x,y in image_points_cam1_cpu.numpy()])
            if world_obj_points.shape[0] > 5:
                augmentation_options_names.append(r_aug_name)
                world_obj_points_options.append(world_obj_points)
                image_points_cam2_options.append(r_backward_t(image_points_cam2_cpu.numpy()))
        
        if len(world_obj_points_options) < 1:
            return None
    
        best_option_idx = np.argmax(np.array([x.shape[0] for x in world_obj_points_options]))
        print(f"chosen rot aug.: {augmentation_options_names[best_option_idx]}")
        world_obj_points = world_obj_points_options[best_option_idx]
        image_points_cam2 = image_points_cam2_options[best_option_idx]


        success, r_img_t_obj, t_img_t_obj, inliers = cv2.solvePnPRansac(
            world_obj_points, image_points_cam2, self.cam2_mtx, None,
            iterationsCount = self.ransac_itterations,
            reprojectionError=self.ransac_reprojection_error,
            confidence = self.ransac_confidence,
            flags = cv2.SOLVEPNP_EPNP
        )
        if not success or len(inliers) < self.min_number_inlier:
           return None
        
        # build results
        cam2_t_base = np.eye(4)
        cam2_t_base[:3, :3] = cv2.Rodrigues(r_img_t_obj)[0]
        cam2_t_base[:3, 3] = t_img_t_obj.flatten()

        return np.linalg.inv(cam2_t_base)




if __name__ == "__main__":
    data = PredictionData.from_folder("/home/wmarx/AR-Headset-Localization-in-Robot-Scanned-Workspaces-A-Benchmark-Pipeline/data_preprocessing/out_data")
    predictor = NoExtrasPredictor(data.headset_intrinsics)
    est_base_t_headset_s = run_predictions(data, predictor)

    for i, pose in enumerate(est_base_t_headset_s):
        print(f"predicted poses {i}: \n\n{pose} \n")
    
    if data.robot_base_t_headset is not None:
        r_err_s, t_err_s = grade_predictions(predictions=est_base_t_headset_s, actual=data.robot_base_t_headset)
        for i, (re, te) in enumerate(zip(r_err_s, t_err_s)):
            if re is not None:        
                print(f"rot_error: {np.round(np.rad2deg(re),1)}deg \n\n t_err: {np.round(te*1000,1)}mm")
                if te > 5e6:
                    est_base_t_headset_s[i] = None
                    print(f"removed datapoint {i} with outlier error")
    
        median_rot_error = np.median(np.array([x for x in r_err_s if x is not None]), axis = 0)
        median_translat_error = np.median(np.array([x for x in t_err_s if x is not None]), axis = 0)

        print(f"number of successful estimates : {len([x for x in r_err_s if x is not None])} / {len(r_err_s)}")
        print(f"median rot error: {np.rad2deg(median_rot_error)} degrees")
        print(f"median translational error: {median_translat_error*1000} mm")
    
    #Visualisation:

    xyz_img_points = data.robot_xyz_images.reshape(-1,3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_img_points)

    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
    robot_camera_s = []
    headset_location_s = []
    for idx, b_t_h in enumerate(est_base_t_headset_s):
        if b_t_h is None:
            continue
        est_headset_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        est_headset_frame.transform(b_t_h)
        headset_location_s.append(est_headset_frame)

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
    