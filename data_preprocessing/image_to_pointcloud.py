from typing import Callable
import torch
import numpy as np
import open3d as o3d
from PIL import Image
import warnings

import matplotlib.pyplot as plt

def get_image_type_hxw(img:np.ndarray) -> str:
    """
    Takes an numpy image array and returns its image type (mostly for debugging)
    :param img: NxHxWx...
    :return: portrait/square/landscape
    """
    assert img.ndim >= 2
    if img.shape[0] > img.shape[1]:
        return "portrait"
    if img.shape[0] == img.shape[1]:
        return "square"
    return "landscape"

def create_foreground_masks(
        images:np.ndarray,
        threshold:float = 0.5,
        mask_threshold:float = 0.5,
        visualize_masks:bool = False
    ) -> np.ndarray:
    """
    Uses Sam3 to detect objects/the foreground and returns a mask for each image, which is `True` where an object was detected
    :param images: NxHxWx3-uint8/float16/float32/float64 numpy array for the images (BGR)
    :param threshold: certainty needed by sam3 to detect an object
    :param mask_threshold certainty for mask generation by sam3
    :param visualize_masks: Whether to visualize the masks for debugging
    :return: NxHxW boolean numpy array of the foreground masks
    """
    assert images.ndim == 4 and images.shape[0] > 0
    assert 0 <= threshold <= 1.0
    assert 0 <= mask_threshold <= 1.0
    print(f"generating foreground masks for {images.shape[0]} {get_image_type_hxw(images[0])}images")
    if images.dtype in [np.float16,np.float32, np.float64]:
        images = (images*255).astype(np.uint8)

    from transformers import Sam3Processor, Sam3Model
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    masks = []
    for idx, image in enumerate(images):
        # TODO add batching for better performance / better prompts
        inputs = processor(images=Image.fromarray(image.astype(np.uint8)), text = "distinct objects",return_tensors="pt").to(device)
        
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
    assert points.ndim == 2 and points.shape[1] == 3

    if contamination == 0:
        return points
    if contamination == 1.0:
        return np.empty((0,3))

    if points.shape[0] > 1000000:
        warnings.warn(f"Using Iforest on {points.shape[0]} points may take a long time", RuntimeWarning)

    from sklearn.ensemble import IsolationForest
    forest = IsolationForest(contamination=contamination)
    forest.fit(points)
    prediction = forest.predict(points)
    return points[prediction==1]

# TODO remove match_poses and/or kabsch_umeyama if not needed anymore
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
        camera_intrinsics:np.ndarray = None,
        confidence_threshold_percent:int = 10,
        image_mask_generator:Callable[[np.ndarray], np.ndarray]|None = None,
        visualize_point_cloud:bool = False,
)-> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    """
    :param rgb_images: A NxHxWx3-uint8/uint16/uint32/uint64/float32 numpy array of RGB images
    :param base_t_cam_s: A Nx4x4-float numpy array of base_t_cam homogeneous transformation matrices
    :param depth_images: A NxHxW-float32 numpy array of depth images (in meters) or None
    :param camera_intrinsics: A 3x3-float numpy-matrix of the camera intrinsics
    :param confidence_threshold_percent: Percentage of low confidence points to be removed (between 0 and 100)
    :param image_mask_generator: A Function that takes a NxHxW-uint8 image array and returns a NxHxW-bool numpy array of masks
    :param visualize_point_cloud: Whether to visualize the generated point cloud
    :return 
    1. a list of RGB images as numpy array
    2. a list of xyz world point images as numpy array
    3. a point cloud as a Nx3 numpy array
    4. the updated camera matrix (3x3 numpy array)
    """

    assert rgb_images.shape[0] == base_t_cam_s.shape[0] , f"Number of rgb images and poses dont match: {rgb_images.shape}, {base_t_cam_s.shape}"
    assert depth_images is None or depth_images.shape[:3] == rgb_images.shape[:3], f"RGB: {rgb_images.shape}, Depth: {depth_images.shape} image dims dont match"    
    assert camera_intrinsics.shape == (3,3), f"Camera intrinsics shape is not 3x3: {camera_intrinsics.shape}"
    assert 0 <= confidence_threshold_percent <= 100, f"confidence_threshhold_percent should be between 0 and 100 is {confidence_threshold_percent}"


    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    from mapanything.models import MapAnything
    from mapanything.utils.image import preprocess_inputs
    from mapanything.utils.image import rgb

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MapAnything.from_pretrained("facebook/map-anything").to(device)
    if rgb_images.dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
        rgb_images = rgb_images.astype(np.float32)/255.0

    views = []
    for image, base_t_cam in zip(rgb_images, base_t_cam_s):
        views.append({
            "img":image,
            "camera_poses":base_t_cam,
            "intrinsics": camera_intrinsics.astype(np.float32)
        })

    if depth_images is not None:
        for view, depth_image in zip(views, depth_images):
            view.update({
                'depth_z': depth_image.astype(np.float32),
                'is_metric_scale': torch.tensor([True], device=device),
            })

    processed_views = preprocess_inputs(views)


    rgb_images = [rgb(view['img'], view['data_norm_type'][0])[0] for view in processed_views]
    rgb_images = [((img*255).astype(np.uint8) if img.dtype in [np.float16, np.float32, np.float64] else img) for img in rgb_images]

    camera_intrinsics = [view['intrinsics'][0].cpu().numpy() for view in processed_views][0]

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


    world_xyz_images = [view['pts3d'][0].cpu().numpy() for view in predictions]
    all_world_points = np.array(world_xyz_images).reshape(-1, 3)

    if image_mask_generator is not None:
        all_world_points = all_world_points[image_mask_generator(np.array(rgb_images)).reshape(-1)]

    if visualize_point_cloud:
        vis_point_cloud = o3d.geometry.PointCloud()
        vis_point_cloud.points = o3d.utility.Vector3dVector(all_world_points)
        o3d.visualization.draw_geometries([vis_point_cloud], window_name = "3D Point cloud visualization")

    assert np.array(rgb_images).shape == np.array(world_xyz_images).shape, f"rgb: {np.array(rgb_images).shape} xyz {np.array(world_xyz_images).shape}"
    return rgb_images, world_xyz_images,all_world_points, camera_intrinsics


def create_point_cloud_simple(
        depth_images:np.ndarray,
        depth_cam_mtx:np.ndarray,
        base_t_camera_s:np.ndarray,
        image_masks: np.ndarray | None = None,
        distance_cutoff:float = 1.0,
        visualize_point_cloud:bool = True,
):
    """
    Uses the depth images to create xyz-images and a point cloud
    :param depth_images: an array of NxHxW-float numpy arrays
    :param depth_cam_mtx: the intrinsic matrix of the depth camera
    :param base_t_camera_s: the homogeneous transformation matrices from base to camera
    :param image_masks: A NxHxW-bool numpy array of what parts of the images to occlude from point cloud generation
    :param distance_cutoff: rays in the depth_images that are longer will be replaced with np.nan and won't be in the point cloud
    :param visualize_point_cloud: whether to visualize point cloud or not
    """
    assert depth_images.ndim == 3
    assert base_t_camera_s.shape == (depth_images.shape[0],4,4)
    assert image_masks is None or image_masks.shape == depth_images.shape and image_masks.dtype == np.bool
    assert 0 <= distance_cutoff
    assert depth_cam_mtx.shape == (3,3)
    print(f"Creating simple point cloud with {depth_images.shape[0]}, {get_image_type_hxw(depth_images[0])} depth images")


    fx, fy, cx, cy = depth_cam_mtx[0,0], depth_cam_mtx[1,1], depth_cam_mtx[0,2], depth_cam_mtx[1,2]
    h, w = depth_images.shape[1:3]
    v_map, u_map = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    base_xyz_images = []
    for d_img, base_t_cam in zip(depth_images, base_t_camera_s):
        d_img[d_img > distance_cutoff] = np.nan
        z = d_img
        x = (u_map-cx)*z / fx
        y = (v_map-cy)*z / fy
        cam_xyz1_points = np.stack([x,y,z, np.ones((h,w))], axis=-1).reshape(-1,4)
        base_xyz1_points = (base_t_cam @ cam_xyz1_points.T).T
        base_xyz_images.append(base_xyz1_points[:, :3].reshape((h,w,3)))

    point_cloud = np.array(base_xyz_images).reshape(-1,3)
    if image_masks is not None:
        point_cloud = point_cloud[image_masks.reshape(-1)]
    point_cloud = point_cloud[~np.isnan(point_cloud[:, 0])]

    if visualize_point_cloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        o3d.visualization.draw_geometries([pcd], "PointCloud visualization")

    return base_xyz_images, point_cloud