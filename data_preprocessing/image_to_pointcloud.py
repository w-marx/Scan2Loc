from typing import Callable
import torch
import numpy as np
import open3d as o3d
from open3d.cuda.pybind.geometry import PointCloud
from PIL import Image

import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

def create_foreground_masks(
        images:np.ndarray,
        threshold:float = 0.5,
        mask_threshold:float = 0.5,
        visualize_masks:bool = False
    ) -> np.ndarray:
    """
    Uses Sam3 to detect objects/the foreground and returns a mask for each image, which is `True` where an object was detected
    :param images: NxWxHx3 numpy array for the images (RGB)
    :return: NxWxH boolean numpy array of the masks
    """
    from transformers import Sam3Processor, Sam3Model
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    
    if images.dtype == np.float32:
        images = (images*255).astype(np.uint8)

    masks = []

    for idx, image in enumerate(images):
        pil_image = Image.fromarray(image.astype(np.uint8))

        inputs = processor(images=pil_image, text = "distinct objects",return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]

        if 'masks' in results and len(results['masks']) > 0:
            instance_masks = results['masks'].cpu().numpy()
            foreground_mask = np.any(instance_masks, axis=0)
            masks.append(foreground_mask)
        else:
            masks.append(np.full((image.shape[0], image.shape[1]), False))
        
        if visualize_masks:
            plt.figure(figsize=(15, 5))
            plt.imshow(image)
            plt.imshow(masks[-1], alpha=0.5, cmap='jet')
            plt.title(f'Overlay ({len(results.get("masks", []))} objects)')
            plt.show()
    
    return np.array(masks)

def remove_outliers_from_point_cloud(points:np.ndarray, contamination:float = 0.05)->np.ndarray:
    """
    Uses I-Forest to remove points deemed as outliers
    :param contamination: The percentage of points to remove
    :param points: A Nx3-float numpy array of x,y,z points
    :return: A Mx3-float numpy array of x,y,z points with M <= N
    """
    assert 0 <= contamination <= 1.0
    if contamination == 0:
        return points
    if contamination == 1.0:
        return np.empty((0,3))

    forest = IsolationForest(contamination=contamination)
    forest.fit(points)
    prediction = forest.predict(points)
    return points[prediction==1]

def match_poses(poses_to_match:np.ndarray, actual_poses:np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """
    Solves that vggts center is off and that the scaling might be wrong.
    To do this the kabsch umeyama algorithm is used (needs at least 3 poses)

    :param poses_to_match: The R_t_cam poses estimated by vggt
    :param actual_poses: The actual robot_base_t_cam poses
    :return: a function that maps vggt [x,y,z] points into the real coordinate system
    """

    if poses_to_match.shape != actual_poses.shape or poses_to_match.shape[1:] != (4, 4):
        raise Exception(f"Cant match poses on pose shapes: {poses_to_match.shape}, {actual_poses.shape}")

    if poses_to_match.shape[0] < 3:
        raise Exception(f"Not enough poses to correct correctly, only {len(poses_to_match)} provided, need at least 3")

    to_match_points = poses_to_match[:, :3, 3]
    actual_points = actual_poses[:, :3, 3]
    R, c, t = kabsch_umeyama(actual_points, to_match_points)
    return lambda point: t + c * R @ point


def kabsch_umeyama(A:np.ndarray, B:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Taken from: https://zpl.fi/aligning-point-patterns-with-kabsch-umeyama-algorithm/
    :param A: list of 3D points
    :param B: list of 3D points
    :return: R, c, t to translate the points B to the points A
    """
    assert A.shape == B.shape
    n, m = A.shape

    EA = np.mean(A, axis=0)
    EB = np.mean(B, axis=0)
    VarA = np.mean(np.linalg.norm(A - EA, axis=1) ** 2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d])

    R = U @ S @ VT
    c = VarA / np.trace(np.diag(D) @ S)
    t = EA - c * R @ EB

    return R, c, t



def create_point_cloud(
        rgb_images:np.ndarray,
        base_t_cam_s: np.ndarray,
        depth_images:np.ndarray | None = None,
        camera_intrinsics:np.ndarray | None = None,
        confidence_threshold_percent:int = 10,
        image_mask_generator:None = None,
        visualize_pointcloud:bool = False,
)-> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """
    :param rgb_images: A NxHxWx3-uint8/float32 numpy array of RGB images
    :param base_t_cam_s: A Nx4x4-float numpy array of base_t_cam homogeneous transformation matrices
    :param depth_images: A NxHxW-float numpy array of depth images or None
    :param camera_intrinsics: A 3x3-float numpy-matrix of the camera intrinsics
    :param confidence_threshold_percent: Percentage of low confidence points to be removed (between 0 and 100)
    :param image_mask_generator: A Funcion that takes a NxHxW-uint8 image array and returns a NxHxW-bool numpy array of masks
    """

    # Check for valid input:
    assert rgb_images.shape[0] == base_t_cam_s.shape[0] , f"Number of rgb images and poses dont match: {rgb_images.shape}, {base_t_cam_s.shape}"
    assert depth_images is None or depth_images.shape[:3] == rgb_images.shape[:3], f"RGB: {rgb_images.shape}, Depth: {depth_images.shape} image dims dont match"    
    assert camera_intrinsics is None or camera_intrinsics.shape == (3,3), f"Camera intrinsics shape is not 3x3: {camera_intrinsics.shape}"
    assert 0 <= confidence_threshold_percent <= 100, f"confidence_threshhold_percent should be between 0 and 100 is {confidence_threshold_percent}"


    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    from mapanything.models import MapAnything
    from mapanything.utils.image import preprocess_inputs
    from mapanything.utils.image import rgb

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MapAnything.from_pretrained("facebook/map-anything").to(device)

    if rgb_images.dtype == np.uint8:
        rgb_images = rgb_images.astype(np.float32)/255.0

    views = []
    import cv2
    for image, base_t_cam in zip(rgb_images, base_t_cam_s):
        views.append({
            "img":image,
            "camera_poses":base_t_cam,
        })

    if camera_intrinsics is not None:
        for view in views:
            view.update({'intrinsics': camera_intrinsics.astype(np.float32)})

    if depth_images is not None:
        for view, depth_image in zip(views, depth_images):
            view.update({
                'depth_z': depth_image.astype(np.float32),
                'is_metric_scale': torch.tensor([True], device=device),
            })

    processed_views = preprocess_inputs(views)


    rgb_images = [rgb(view['img'], view['data_norm_type'][0])[0] for view in processed_views]

    predictions = model.infer(
        processed_views,
        memory_efficient_inference=True,
        minibatch_size = None,
        use_amp = True,
        amp_dtype = "bf16",
        apply_mask=True,
        mask_edges=True,
        apply_confidence_mask=False,
        confidence_percentile=confidence_threshold_percent,
        use_multiview_confidence=False,
        ignore_calibration_inputs=False,
        ignore_depth_inputs=False,
        ignore_pose_inputs=False,
        ignore_depth_scale_inputs=False,
        ignore_pose_scale_inputs=False,
    )


    world_xyz_images = [view['pts3d'].cpu().numpy() for view in predictions]
    camera_points_for_views = [view['pts3d_cam'].cpu().numpy() for view in predictions]

    all_world_points = np.array(world_xyz_images).reshape(-1, 3)

    if image_mask_generator is not None:
        all_world_points = all_world_points[image_mask_generator(np.array(rgb_images)).reshape(-1)]

    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(all_world_points)

    if visualize_pointcloud:
        o3d.visualization.draw_geometries([pointcloud], window_name = "3D Point cloud visualization")


    return rgb_images, world_xyz_images, np.asanyarray(pointcloud.points)