from typing import Callable
import torch
import numpy as np
import open3d as o3d
from open3d.cuda.pybind.geometry import PointCloud
from PIL import Image

from sklearn.ensemble import IsolationForest

def use_vggt_on_images(images:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    :param images: NxWxHx3 RGB numpy array of the images
    :return:
    1. NxWxHx3 xyz points for the images (as numpy array)
    2. the confidence for each point in a NxWxH numpy array
    3. The extrinsic camera matrices NX3x4 numpy array
    4. The intrinsic camera matrices
    5. The depth map
    6. The depth map confidences
    """

    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    import cv2

    resized_images = []
    for img in images:
        new_width = int(img.shape[1] * 0.3)
        new_height = int(img.shape[0] * 0.3)
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        resized_images.append(resized)
    images = np.array(resized_images)


    device = "cuda" if torch.cuda.is_available() else "cpu"

    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    if images.dtype == np.uint8:
        images = images.astype(np.float32)/255.0

    #padd each image with white so that its width and height is a multiple of 14 (needed by vggt)
    orig_h = images.shape[1]
    orig_w = images.shape[2]
    padding_h = (14-orig_h%14)%14
    padding_w = (14-orig_w%14)%14

    padded_images = np.pad(images, ((0,0),(0,padding_h), (0,padding_w), (0,0)), mode="constant", constant_values=0)

    for idx, image in enumerate(padded_images):
        cv2.imshow(f"image {idx}",image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #transform to image tensor
    image_tensor = torch.from_numpy(padded_images).to(device, dtype=dtype)
    image_tensor = image_tensor.permute(0,3,1,2)

    #compute predictions
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(image_tensor)

    #unpad the points
    np_points = predictions['world_points'].cpu().numpy()[0]
    np_confidences = predictions['world_points_conf'].cpu().numpy()[0]
    points_np_unpad = np_points[:,:orig_h, :orig_w]
    confidence_np_unpad = np_confidences[:,:orig_h, :orig_w]
    assert(np.shape(points_np_unpad) ==np.shape(images))

    # get the R_T_H estimates
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions['pose_enc'],(orig_h,orig_w))
    extrinsic = extrinsic[0].cpu().numpy()
    intrinsic = intrinsic[0].cpu().numpy()
    depth_map = predictions['depth'][0].cpu().numpy()[:,:orig_h, :orig_w, :]
    depth_map_conf = predictions['depth_conf'][0].cpu().numpy()[:,:orig_h, :orig_w]

    return points_np_unpad, confidence_np_unpad, extrinsic, intrinsic, depth_map, depth_map_conf

def get_confidence_masks(confidences:np.ndarray, quantile:float = 0.1) -> np.ndarray:
    """
    :param confidences: An array of the form N x H x W - float
    :param quantile: The quantile of pixels with low confidences to disregard
    :return: An array of the form N x H x W - boolean
    """
    confidence_threshhold = np.quantile(confidences, quantile)
    return np.where(confidences > confidence_threshhold, True, False)

def create_point_cloud_from_image_points(
        method = "3D points",
        images_points_3d_and_conf:tuple[np.ndarray, np.ndarray] | None = None,
        images_depth_maps_and_conf:tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]| None = None,
        masks:np.ndarray|None = None,
        confidence_quantile:float = 0.1,
        iforest_quantile:float = 0.05,
        visualize:bool = True
    ) -> PointCloud:
    """
    Creates a point cloud using only the points where the mask is true.
    :param method: What method to use to create the point cloud, can be `3D points` or `Depth`
    :param images_points_3d_and_conf: An array of the form N x H x W x 3 - float and the confidences in a NxHxW array
    :param images_depth_maps_and_conf: An array of the form N x H x W - float and the confidences in a NxHxW array additionaly the extrinsic and intrinsic camera matrices
    :param confidence_quantile: The quantile of pixels with low confidences to disregard
    :param iforest_quantile: The quantile of points that are anomalies according to IForest to disregard
    :param masks: what pixels to mask of the form number_images x height x width - boolean
    :param visualize: whether to visualize the resulting point cloud
    :return: nothing
    """

    if method != "3D points" and method != "Depth":
        raise Exception(f"The method {method} is not supported in this version")

    if method == "3D points" and images_points_3d_and_conf is None:
        raise Exception("Tried to do 3D points for point cloud reconstruction but images_points_3d and their confidences is None")

    if method == "Depth" and images_depth_maps_and_conf is None:
        raise Exception("Tried to do Depth for point cloud reconstruction but images_depth_maps and their confidences is None")

    points = None
    conf_masks = None

    if method == "3D points":
        print("Generating the 3D Point Cloud using the 3D points directly")
        images_points_3d, images_points_3d_conf = images_points_3d_and_conf
        conf_masks = get_confidence_masks(images_points_3d_conf, confidence_quantile)
        points = images_points_3d

    if method == "Depth":
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        print("Generating the 3D Point Cloud by unprojecting the depth map")
        images_depth_maps, images_depth_maps_conf, extrinsic, intrinsic = images_depth_maps_and_conf
        conf_masks = get_confidence_masks(images_depth_maps_conf, confidence_quantile)
        points = unproject_depth_map_to_point_map(images_depth_maps, extrinsics_cam=extrinsic, intrinsics_cam=intrinsic)

    masks = conf_masks & masks if masks is not None else conf_masks
    masks = np.reshape(np.array(masks), (-1))
    points = np.reshape(np.array(points), (-1, 3))
    points = points[masks]
    points = remove_outliers_from_point_cloud(points, contamination=iforest_quantile)

    # Generate Ply file
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if visualize:
        o3d.visualization.draw_geometries([pcd], window_name = "visualization")
    return pcd


def create_foreground_masks(images:np.ndarray) -> np.ndarray:
    """
    Uses Sam3 to detect objects/the foreground and returns a mask for each image, which is `True` where an object was detected
    :param images: NxWxHx3 numpy array for the images (RGB)
    :return: NxWxH boolean numpy array of the masks
    """

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    masks = []
    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    for image in images:
        mask = np.full((image.shape[0], image.shape[1]), False)

        inference_state = processor.set_image(Image.fromarray(image))
        output = processor.set_text_prompt(prompt="distinct objects", state=inference_state)


        cpu_masks = output["masks"].cpu().numpy()
        mask = mask | np.any(cpu_masks, axis=0)[0]

        masks.append(mask)

    print(f"masks: {np.shape(np.array(masks))}")
    return np.array(masks)

def remove_outliers_from_point_cloud(points:np.ndarray, contamination:float = 0.05)->np.ndarray:
    """
    Uses I-Forest to remove points deemed as outliers
    :param contamination: The percentage of points to remove
    :param points: A Nx3-float numpy array of x,y,z points
    :return: A Mx3-float numpy array of x,y,z points with M <= N
    """
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
        confidence_threshhold_percent:int = 10,
        image_masks:np.ndarray | None = None,

):
    """
    :param rgb_images: A NxHxWx3-uint8/float32 numpy array of RGB images
    :param base_t_cam_s: A Nx4x4-float numpy array of base_t_cam homogeneous transformation matrices
    :param depth_images: A NxHxW-float numpy array of depth images or None
    :param camera_intrinsics: A 3x3-float numpy-matrix of the camera intrinsics
    :param confidence_threshhold_percent: Percentage of low confidence points to be removed (between 0 and 100)
    :param image_masks: A NxHxW-bool numpy array of masks
    """

    # Check for valid input:
    assert rgb_images.shape[0] == base_t_cam_s.shape[0] , f"Number of rgb images and poses dont match: {rgb_images.shape}, {base_t_cam_s.shape}"
    assert depth_images is None or depth_images.shape[:3] == rgb_images.shape[:3], f"RGB: {rgb_images.shape}, Depth: {depth_images.shape} image dims dont match"
    assert image_masks is None or image_masks.shape[:3] == rgb_images.shape[:3], f"Mask {image_masks.shape} and Images {rgb_images.shape} dims dont match"
    assert camera_intrinsics is None or camera_intrinsics.shape == (3,3), f"Camera intrinsics shape is not 3x3: {camera_intrinsics.shape}"
    assert 0 <= confidence_threshhold_percent <= 100, f"confidence_threshhold_percent should be between 0 and 100 is {confidence_threshhold_percent}"


    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    from mapanything.models import Mapanything
    from mapanything.utils.image import preprocess_inputs

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MapAnything.from_pretrained("facebook/map-anything").to(device)

    if rgb_images.dtype == np.uint8:
        rgb_images = rgb_images.astype(np.float32)/255.0

    views = []
    for image, base_t_cam in zip(rgb_images, base_t_cam_s):
        views.append({
            "img":image,
        #    "camera_poses":base_t_cam,
        })

    if camera_intrinsics is not None:
        for view in views:
            view.update({'intrinsics': camera_intrinsics})

    if depth_images is not None:
        for view, depth_image in zip(views, depth_images):
            view.update({
                'depth_z': depth_image,
                'is_metric_scale': torch.tensor([True], device=device),
            })

    print(f"created mapanything dictionary: {views}")

    processed_views = preprocess_inputs(views)
    predictions = model.infer(
        processed_views,
        memory_efficient_inference=True,
        minibatch_size = None,
        use_amp = True,
        amp_dtype = "bf16",
        apply_mask=True,                  # Apply masking to dense geometry outputs
        mask_edges=True,                  # Remove edge artifacts by using normals and depth
        apply_confidence_mask=False,      # Filter low-confidence regions
        confidence_percentile=confidence_threshhold_percent,         # Remove bottom 10 percentile confidence pixels
        use_multiview_confidence=False,
        ignore_calibration_inputs=False,
        ignore_depth_inputs=False,
        ignore_pose_inputs=False,
        ignore_depth_scale_inputs=False,
        ignore_pose_scale_inputs=False,
    )

    world_points = predictions['pts3d'].cpu().numpy()
    print(f"world points shape: {world_points.shape}")

    camera_points = predictions['pts3d_cam'].cpu().numpy()
    print(f"camera points shape: {camera_points.shape}")


    world_points_masked = world_points.copy().reshape(-1, 3)

    if image_masks is not None:
        world_points_masked = world_points_masked[image_masks.reshape(-1)]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(world_points_masked)


    if True:
        o3d.visualization.draw_geometries([pcd], window_name = "visualization")
    return pcd











