from typing import Callable, Literal
from dataclasses import dataclass
import torch
import numpy as np
import open3d as o3d
from PIL import Image
import cv2
import sys, os
import matplotlib.pyplot as plt
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_utilities import get_image_type_hxw, compute_pose_pseudo_median

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

    from sklearn.ensemble import IsolationForest
    forest = IsolationForest(contamination=contamination)
    forest.fit(points)
    prediction = forest.predict(points)
    return points[prediction==1]


def kabsch_umeyama(A:np.ndarray, B:np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """
    Taken from: https://zpl.fi/aligning-point-patterns-with-kabsch-umeyama-algorithm/
    :param A: list of 3D points
    :param B: list of 3D points
    :return: A function to transform the points B to the points A
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

    return lambda points: (t.reshape(3,1) + c * R @ points.T).T


@dataclass(frozen=True, kw_only=True)
class ICPAlignmentConfig:
    """
    :param neighboar_dist_threshhold: The distance between points in meters, for consideration in icp
    :param max_number_itterations: The number of itterations for icp optimization per pointcloud
    :param presample_voxel_size: If bigger then 0, the pointclouds will be downsampled to that voxel size for quicker realignment
    """
    neighboar_dist_threshhold:float = 0.01
    max_number_itterations:int = 1000
    presample_voxel_size:float = 0.0

    def __post__init__(self):
        assert 0 < self.max_number_itterations, f"Number of ICP itterations must be positive: {self.max_number_itterations}"
        assert 0 < self.neighboar_dist_threshhold, f"ICP neighboar distance must be positive: {self.neighboar_dist_threshhold}"

ICPAlignmentConfigs = {
    "standard": ICPAlignmentConfig(),
    "downsample_1mm": ICPAlignmentConfig(presample_voxel_size = 0.001),
    "downsample_5mm": ICPAlignmentConfig(presample_voxel_size = 0.005)
}


def align_point_clouds_icp(
        point_clouds:list[np.ndarray], 
        config:ICPAlignmentConfig = ICPAlignmentConfig(),
        visualize:bool = True
    )->list[np.ndarray]:
    """
    Takes M > 1 Pointclouds and returns them aligned around the geometric median of the pointclouds.
    Aligns all of the point_clouds with point_cloud_0 using ICP 
    and then applies the geometric median of the transformations on all pointclouds
    :param point_clouds a list of N_i x 3-float numpy arrays
    :param icp_threshhold
    :param icp_max_number_itterations
    :param icp_voxel_size, the voxel size to downsample to before doing icp, if 0 no downsampling will be done
    :param visualize wheather to visualize the unaligned and aligned pointclouds
    :returns a list of the aligned point clouds (same size & order of clouds and their points as input)
    """
    assert len(point_clouds) > 1
    assert all([pc.ndim == 2 and pc.shape[-1] == 3 for pc in point_clouds])

    o3d_point_clouds = []
    for i, point_cloud in enumerate(point_clouds):
        o3d_point_clouds.append(o3d.geometry.PointCloud())
        o3d_point_clouds[-1].points = o3d.utility.Vector3dVector(point_cloud)
        if config.presample_voxel_size > 1e-9:
            o3d_point_clouds[-1] = o3d_point_clouds[-1].voxel_down_sample(voxel_size=config.presample_voxel_size)

    ref_pc = o3d_point_clouds[0]

    ref_t_pci_s = [np.eye(4)]

    print("aligning pointclouds using ICP")
    for pci in tqdm(o3d_point_clouds[1:]):
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pci, 
            ref_pc, 
            config.neighboar_dist_threshhold, 
            np.eye(4), 
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=config.max_number_itterations
            )
        )
        ref_t_pci_s.append(np.linalg.inv(reg_p2p.transformation))
    
    ref_t_median_pci = compute_pose_pseudo_median(ref_t_pci_s)

    pci_t_median_pci_s = [np.linalg.inv(ref_t_pci) @ ref_t_median_pci for ref_t_pci in ref_t_pci_s]

    hom_point_clouds = [np.hstack([pci, np.ones((pci.shape[0],1))]) for pci in point_clouds]
    aligned_point_clouds = [((pci_t_median_pci @ pci.T).T)[:,:3] for pci_t_median_pci, pci in zip(pci_t_median_pci_s, hom_point_clouds)]

    if visualize:
        aligned_o3d_point_clouds = []
        for point_cloud in aligned_point_clouds:
            aligned_o3d_point_clouds.append(o3d.geometry.PointCloud())
            aligned_o3d_point_clouds[-1].points = o3d.utility.Vector3dVector(point_cloud)
            aligned_o3d_point_clouds[-1].paint_uniform_color([0, 0, 0])
        o3d.visualization.draw_geometries(o3d_point_clouds+aligned_o3d_point_clouds)

    return aligned_point_clouds



def create_point_cloud(
        bgr_images:np.ndarray,
        base_t_cam_s: np.ndarray,
        depth_images:np.ndarray | None = None,
        camera_intrinsics:np.ndarray = None,
        confidence_threshold_percent:int = 10,
        xyz_image_aligner:Callable[[np.ndarray], np.ndarray] | None = None,
        visualize_point_cloud:bool = False,
        alginment_method:Literal["none", "simple", "kabsch-umeyama"] = "kabsch-umeyama",
        crop_square:bool = True
)-> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    :param bgr_images: A NxHxWx3-uint8/uint16/uint32/uint64/float32 numpy array of BGR images
    :param base_t_cam_s: A Nx4x4-float numpy array of base_t_cam homogeneous transformation matrices
    :param depth_images: A NxHxW-float32 numpy array of depth images (in meters) or None, SHOULD NOT BE USED IF DEPTH AND BGR ARE NOT ALIGNED
    :param camera_intrinsics: A 3x3-float numpy-matrix of the camera intrinsics
    :param confidence_threshold_percent: Percentage of low confidence points to be removed (between 0 and 100)
    :param image_mask_generator: A Function that takes a NxHxWx3-uint8 image array and returns a NxHxW-bool numpy array of masks
    :param xyz_image_aligner: A Function that takes a NxHxW-float xyz image array and aligns the images and returns the aligned images
    :param visualize_point_cloud: Whether to visualize the generated point cloud
    :param crop_square: Crops the bgr images to be sqare to utilize the full ~500x500 image size allowed by mapanything (dont use if distortion is not 0)
    :return 
    1. NxWxHx3-uint8 BGR images as numpy array
    2. NxWxHx3-float larray of xyz world point images
    3. the updated camera matrix (3x3 numpy array)
    """

    assert bgr_images.shape[0] == base_t_cam_s.shape[0] , f"Number of bgr images and poses dont match: {bgr_images.shape}, {base_t_cam_s.shape}"
    assert depth_images is None or depth_images.shape[:3] == bgr_images.shape[:3], f"BGR: {bgr_images.shape}, Depth: {depth_images.shape} image dims dont match"    
    assert camera_intrinsics.shape == (3,3), f"Camera intrinsics shape is not 3x3: {camera_intrinsics.shape}"
    assert 0 <= confidence_threshold_percent <= 100, f"confidence_threshhold_percent should be between 0 and 100 is {confidence_threshold_percent}"

    if crop_square:
        h_orig = bgr_images.shape[1]
        w_orig = bgr_images.shape[2]

        if w_orig > h_orig+2:
            crop_amount = int((w_orig-h_orig)/2)
            bgr_images = bgr_images[:,:,crop_amount:-crop_amount,:]
            depth_images = depth_images[:,:,crop_amount:-crop_amount]
            camera_intrinsics[0,2] -= crop_amount
        elif h_orig > w_orig+2:
            crop_amount = int((h_orig-w_orig)/2)
            bgr_images = bgr_images[:,crop_amount:-crop_amount,:,:]
            depth_images = depth_images[:, crop_amount:-crop_amount,:]
            camera_intrinsics[1,2] -= crop_amount

    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    from mapanything.models import MapAnything
    from mapanything.utils.image import preprocess_inputs
    from mapanything.utils.image import rgb

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MapAnything.from_pretrained("facebook/map-anything").to(device)
    if bgr_images.dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
        bgr_images = bgr_images.astype(np.float32)/255.0 # wrong in the documentation :( needs 0-1

    views = []
    print(f"image shape before processing: {bgr_images.shape}")
    for image, base_t_cam in zip(bgr_images, base_t_cam_s):
        views.append({
            "img":image,
            "camera_poses": base_t_cam,
            "intrinsics": camera_intrinsics.astype(np.float32)
        })

    if depth_images is not None:
        for view, depth_image in zip(views, depth_images):
            view.update({
                'depth_z': depth_image.astype(np.float32),
                'is_metric_scale': torch.tensor([True], device=device),
            })

    processed_views = preprocess_inputs(views)


    bgr_images = [rgb(view['img'], view['data_norm_type'][0])[0] for view in processed_views]
    bgr_images = [((img*255).astype(np.uint8) if img.dtype in [np.float16, np.float32, np.float64] else img) for img in bgr_images]

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


    world_xyz_images = np.array([view['pts3d'][0].cpu().numpy() for view in predictions])

    if alginment_method == "simple":
        world_xyz_images = []
        cam_xyz_images = [view['pts3d_cam'][0].cpu().numpy() for view in predictions]
        for cam_img, base_t_cam in zip(cam_xyz_images, base_t_cam_s):
            cam_points = cam_img.reshape(-1,3)
            cam_points_hom = np.hstack([cam_points, np.ones((cam_points.shape[0],1))])
            world_points = (base_t_cam @ cam_points_hom.T).T
            world_xyz_images.append(world_points[:, :3].reshape(cam_img.shape[0], cam_img.shape[1], cam_img.shape[2]))
        world_xyz_images = np.array(world_xyz_images)
    elif alginment_method == "kabsch-umeyama":
        camera_true_positions = base_t_cam_s[:,:3,3]
        camera_new_positions = np.array([view['cam_trans'][0].cpu().numpy() for view in predictions])
        transform_points_to_old = kabsch_umeyama(camera_true_positions, camera_new_positions)
        world_xyz_images = transform_points_to_old(world_xyz_images.reshape(-1,3)).reshape(world_xyz_images.shape)
    else:
        print("didnt to camera alignment")


    if xyz_image_aligner is not None:
        world_xyz_images = xyz_image_aligner(world_xyz_images)

    # Generate pointcloud
    if visualize_point_cloud:
        vis_point_cloud = o3d.geometry.PointCloud()
        vis_point_cloud.points = o3d.utility.Vector3dVector(world_xyz_images.reshape(-1, 3))
        o3d.visualization.draw_geometries([vis_point_cloud], window_name = "3D Point cloud visualization")

    print(f"xyz images shape: {world_xyz_images.shape}")
    assert np.array(bgr_images).shape == np.array(world_xyz_images).shape, f"bgr: {np.array(bgr_images).shape} xyz {np.array(world_xyz_images).shape}"
    return bgr_images, world_xyz_images, camera_intrinsics


def create_point_cloud_depth_reproject(
        depth_images:np.ndarray,
        depth_cam_mtx:np.ndarray,
        base_t_camera_s:np.ndarray,
        distance_cutoff:float = 1.0,
        visualize_point_cloud:bool = True,
):
    """
    Uses the depth images to create xyz-images and a point cloud
    !!! If color cam and depth cam are not Aligned the xyz-imgs cant really be used !!!
    :param depth_images: an array of NxHxW-float numpy arrays
    :param depth_cam_mtx: the intrinsic matrix of the depth camera
    :param base_t_camera_s: the homogeneous transformation matrices from base to camera
    :param distance_cutoff: rays in the depth_images that are longer will be replaced with np.nan and won't be in the point cloud
    :param visualize_point_cloud: whether to visualize point cloud or not
    """
    assert depth_images.ndim == 3
    assert base_t_camera_s.shape == (depth_images.shape[0],4,4)
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

    if visualize_point_cloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        o3d.visualization.draw_geometries([pcd], "PointCloud visualization")

    return base_xyz_images