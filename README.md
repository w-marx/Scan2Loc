<div align="center">
  
# Scan2Loc
  
### AR Headset Localization in Robot Scanned Workspaces

**A Benchmark Pipeline**
</div>

## Overview
**Scan2Loc** provides a complete benchmark pipeline for evaluating marker-free AR headset localization in robot-scanned workspaces. This repository accompanies the bachelor thesis *"AR Headset Localization in Robot Scanned Workspaces: A Benchmark Pipeline"*.

#### Key Features

| Feature | Description |
|---------|-------------|
| **Datasets** | Desktop datasets with `.vrs` recordings + TUM-RGBD integration |
| **Dataset Creation** | Scripts for generating new robot-scanned environments |
| **Three Localizers** | PnP, PnP+Lines, and Ellipsoid-based refinement |
| **Evaluation Suite** | ATE/RPE metrics, time analysis, and gaze-intersection error |


## Datasets
<p align="center">
  <img width="300" style="margin: 10px;" alt="image" src="https://github.com/user-attachments/assets/12635292-cf01-457a-af56-5c45c6874a94" />
  <img width="300" style="margin: 10px;" alt="tum_first10_percent(1)" src="https://github.com/user-attachments/assets/eb786e5a-7f29-463e-a99e-012c5561c02d" />
  <br>
  <em>Figure: own dataset (left) and TUM dataset (right).</em>
</p>

Multiple desktop datasets are provided in [`example_datasets/`](./example_datasets), including:
- Scenes with ArUco/ChArUco markers for ground truth
- Matching `.vrs` recordings from Meta Aria Gen1 glasses
  
Furthermore this repo supports the [TUM RGB-D Dataset](https://cvg.cit.tum.de/data/datasets/rgbd-dataset).


## Dataset Creation
Create new robot-scanned datasets with:
- Automated scanning using Franka Emika Panda + Intel RealSense D435
- Hand-eye calibration and ground truth trajectory generation via fiducial markers
- Advanced marker removal using [LaMa](https://github.com/advimman/lama.git) 
- Direct dataset creation using Meta Aria glasses

**Guide**: [`DataGathering.ipynb`](./notebooks/DataGathering.ipynb).


## Localization
**Scan2Loc** provides 3D reconstruction from robot scans and 3 localizer families to locate a camera in them.

#### Reconstruction
Direct scene reconstruction via [MapAnything](https://github.com/facebookresearch/map-anything.git) is provided, supporting depth images. Furthermore ICP alignment and scene cleanup are directly provided.

#### Three Localizer Families
<p align="center">
  <img width="800" alt="example localization using the ellipsoid localizer" src="https://github.com/user-attachments/assets/0d9bde49-3df4-4dac-9ff5-abbc087f7ad8" />
  <br>
  <em>Figure: Example localization using the ellipsoid localizer</em>
</p>

| Localizer | Description | Key Features |
|-----------|-------------|--------------|
| **PnP** | Baseline keypoint matching | [LightGlue](https://github.com/cvg/LightGlue.git) · [LoMa](https://github.com/davnords/LoMa.git) · [E-LoFTR](https://github.com/zju3dv/efficientloftr) |
| **PnP+L** | Points + Lines refinement | Line segment detection & matching |
| **Ellipsoid** | Object-level refinement | Ellipse-ellipsoid bounding boxes |

## Evaluation Framework
The [`headset_localization`](./headset_localization) package provides comprehensive evaluation tools for benchmarking localizer performance, including comparison to ground truth and other localizers.

#### Available Metrics

| Metric | Description |
|--------|-------------|
| **Absolute Trajectory Error** | Translational & rotational error for trajectories and frames |
| **Relative Pose Error (RPE)** | Drift between consecutive poses (SLAM-style) |
| **6D Signed Errors** | Per-axis breakdown for failure analysis |
| **Gaze-Intersection Error (GIE)** | Intuitive gaze-based HRI metric (ray-scene intersection) |
| **Timing Profiling** | Online/offline inference speed per component |
| **Success Rates** | Fraction of successful localizations (with definable success thresholds) |


## Getting Started
### For Inference and Evaluation
All utilities for inference and evaluation are provided by the [`headset_localization`](./headset_localization) package. The dependencies to run the package are provided in the [`env_3090.yml`](./env_3090.yml) file.

```bash
# Create environment (RTX 3090, CUDA 12.2)
conda env create -f env_3090.yml -p ./env
conda activate ./env
```
Note: Other CUDA versions / GPUs may require different library versions.


The `headset_localization` package can then be installed using the [`install_pose_pred_dependencies.sh`](./install_pose_pred_dependencies.sh) script into any environment. 
```bash
bash install_pose_pred_dependencies.sh 
```
This script also installs [MapAnything](https://github.com/facebookresearch/map-anything.git), [LightGlue](https://github.com/cvg/LightGlue.git) and [LoMa](https://github.com/davnords/LoMa.git). Both the LightGlue and LoMa installations are optional and only necessary if you wish to use them.


**Example:** Using a localizer to predict the pose of a camera in a scanned environment (inference only):
```python
from headset_localization import HeadsetRecording, Scanned3dEnvironment, PnPLocalizer
from shared import CompleteRobotScan

headset_data = HeadsetRecording.from_vrs_file("../example_datasets/small_aruco1_sitting_20fps.vrs")

workspace_reconstruction = Scanned3dEnvironment.from_gathered_robot_data(
    robot_data = CompleteRobotScan.from_folder("../example_datasets/example_small_aruco1")
)

localizer = PnPLocalizer(
    cam2_intrinsic_mtx=headset_data.intrinsic_cam_mtx,
    cam1_bgr_images=workspace_reconstruction.robot_bgr_images,
    cam1_xyz_images=workspace_reconstruction.robot_xyz_images,
)

INDEX = 30
query_image = headset_data.bgr_image_s[INDEX]

base_t_cam = localizer.est_base_t_cam2(query_image)
print(base_t_cam)
```

For detailed usage examples including evaluation refer to the [notebooks](./notebooks), especially [`Quickstart.ipynb`](./notebooks/Quickstart.ipynb). Some notebooks require the `fr2/desk` dataset in a `./tum_datasets` folder [Download](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download).


### For dataset creation only
To create your own dataset, the following dependencies, which don't require CUDA, need to be installed.

```bash
cd data_gathering
conda env create -f environment.yml -p ./data_gather_env
conda activate ./data_gather_env
pip install -e ../shared
```
This environment needs access to a working [deoxys](https://github.com/UT-Austin-RPL/deoxys_control) installation to be able to control a Franka Panda Emika robot.
For a tutorial on how to scan a workspace using deoxys and adding marker information refer to [DataGathering.ipynb](./notebooks/DataGathering.ipynb).

For this a `.csv` table of joint coordinates (positions = rows, n_joints = columns) to scan at is required of which multiple are provided: [pos11](./data_gathering/pos11.csv), [positions_panda_63](./data_gathering/positions_panda_63).   
Alternatively new ones for the Franka Panda Emika robot can be created using the script: [read_joint.py](./data_gathering/read_joint.py).   
To do this enable hand guidance on the robot and run:   
```bash
conda activate ./data_gather_env
python read_joint.py
# press 'A'/'a' to add the current position of the robot to the .csv (multiple times)
# press 'S'/'s' to save the csv to ./positions.csv
```
Optional usage with hpyerparameters:
```bash
python read_joint.py --interface-cfg "charmander.yml" --folder "./positions.csv"
```


## License
This project is licensed under the **GNU Affero General Public License v3.0**. See the [LICENSE](./LICENSE) file for details.

This project depends on [MapAnything](https://github.com/facebookresearch/map-anything.git), [LightGlue](https://github.com/cvg/LightGlue.git) and [LoMa](https://github.com/davnords/LoMa.git). Their licenses are available under `./external/[repo]/Licence` after installation.

This project also depends on various Python packages installed via Conda and pip. Dependencies have their own licenses; please refer to their repositories.

## File Formats
Different datasets and environment representations can directly be created from folders, this section defines the file formats.
**Note**: Folders are sorted alphabetically. Use padding for proper ordering e.g. `0-8`, `000-115`.
#### Raw Robot Scan

```
folder
├── robot
│   └── multiple folders (0 - N) with the contents:
│       ├──  rgb.png
│       ├──  poses.json
│       └──  depth.npz
└── robot_cam_calibration.json
```
robot_cam_calibration.json has to have the following attributes:
- `camera_intrinsic_matrix`: the intrinsic camera matrix of the camera: `[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]`
- `camera_distortion_coefficients`: the distortion coefficients of the camera e.g.`[0.0, 0.0, 0.0, 0.0, 0.0]`

#### Complete Robot Scan
```
folder
├── robot
│   └── multiple folders (0 - N) with the contents:
│       ├──  rgb.png
│       ├──  depth.npz
│       └──  poses.json
├── gripper_t_cam.npy
├── marker_detector_config.json
└── robot_cam_calibration.json
```

robot_cam_calibration.json has to have the same attributes as for the raw robot scan. 

the depth.npz files contain a metric `HxWxnp.float32` depth image, under the key `depth`

marker_detector_config.json has to have the following attributes:
- `"marker_type"`: the marker type: `Aruco` | `Charuco` | `null`
- `"marker_side_length"`: ArUco marker side length in meters
- `"aruco_marker_dictionary"`: A string describing the dict, e.g. "6X6_250"
- `"board_size"`: `null` | `[length1, length2]`
- `"square_size"`: Size of the chessboard squares in meters
- `"min_fraction_of_markers"`: fraction from 0.0-1.0 of how much of the board must be visible for detection.


The `poses.json` files have to contain the following attributes:
- `base_t_gripper`: 4x4 transformation matrix as a row-colum nested list of floats
- `camera_t_marker`: 4x4 transformation matrix as a row-colum nested list of floats | `null`


#### Scanned 3D Environment
```
folder
├── robot_cam_calibration.json
└── robot
   └── multiple folders (0 - N) with the contents:
       ├── xyz.npy
       ├── robot_base_t_robot_camera.json
       └── rgb.png image
```

The `xyz.npy` files contain the metric 3D images (Each HxWx3).
The `robot_base_t_robot_camera.json` contains only the 4x4 robot->robot_camera transformation matrix as a row-colum nested list of floats.

#### Headset Recording

```
folder
├── headset_cam_calibration.json
└── headset
    └── multiple folders (0 - N) with the contents:
        ├── label.json (optional)
        └── rgb.png
```

The `label.json` contains only the 4x4 robot->headset ground truth transformation matrix as a row-colum nested list of floats. 
