import cv2
import torch
import os
import numpy as np
import open3d as o3d
from open3d.cuda.pybind.geometry import PointCloud

from vggt.models.vggt import VGGT

from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

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
        masks:np.ndarray = None,
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
        print("Generating the 3D Point Cloud by unprojecting the depth map")
        images_depth_maps, images_depth_maps_conf, extrinsic, intrinsic = images_depth_maps_and_conf
        conf_masks = get_confidence_masks(images_depth_maps_conf, confidence_quantile)
        points = unproject_depth_map_to_point_map(images_depth_maps, extrinsics_cam=extrinsic, intrinsics_cam=intrinsic)

    masks = conf_masks & masks
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

def match_poses(poses1:list[np.ndarray], poses2:list[np.ndarray]) -> np.ndarray:
    """
    Solves that vggts center is off and that the scaling might be wrong, therefore solve for:

    For all i:
    poses2[i] = T @ rescale(poses1[i],s)

    Where T is a 4x4 homogeneous transformation matrix and s scales the translational part of poses1[i]

    :param poses1: The R_t_cam poses estimated by vggt
    :param poses2: The actual robot_base_t_cam poses
    :return: the correction matrices
    """


