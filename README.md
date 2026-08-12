# AR-Headset-Localization-in-Robot-Scanned-Workspaces-A-Benchmark-Pipeline

-----

For gaze based HRI in scanned desktop environments, using AR-Headsets it is necessary to locate the headset in the environment.
This repo offers a comprehensive data collection and performance evaluation pipeline for marker-free localization of a rgb-camera's within a robot-scanned workspaces.
Furthermore it offers three inference ready localizers: a baseline PnP approach using keypoint matching, a PnP+L variant that refines estimates with line features, and a PnP+Ellipsoids variant leveraging object bounding boxes. 

-----

## Provided Functionalities:

#### Datasets
Multiple desktop datasets are provided in the `example_datasets` folder. Consisting of multiple scenes and matching `.vrs` recordings in them.

#### Dataset creation
The scripts in the `data_gathering` folder along with the `shared` package allows the creation of new datasets. A guide on this is provided in `notebooks/DataGathering.ipynb`. 

#### Three Inference ready Localizers
This repo offers three families of inference ready localizers. The first relying on PnP solutions provided by `LoMa`, `LightGlue` or `E-LoFTR`. The second uses lines to refine the PnP solution and the third ellipsoid bounding boxes.

#### Evaluation Framework
The `headset_localization` package can be used to evaluate those localizers on new datasets and to finetune them.

## Getting Started
#### For dataset creation
To create your own dataset the following dependencies need to be installed:
```
cd data_gathering
conda env create -f environment.yml -p ./data_gather_env
conda activate ./data_gather_env
pip install -e ../shared
```
This environment needs access to a working deoxys installation (https://github.com/UT-Austin-RPL/deoxys_control) to be able to control a Franka Panda Emika robot.
For example usage refer to `notebooks/DataGathering.ipynb`.

#### For Inference and Evaluation
All utilities for inference and evaluation are provided by the `headset_localization` package.
The dependencies to run the `headset_localization` package are provided in the `env_3090.yml` file. Those work on a NVIDIA GeForce RTX 3090 with CUDA 12.2 . Other CUDA versions and graphics cards might require different library versions.
```
conda env create -f env_3090.yml -p ./env
conda activate ./env
```
The `headset_localization` package can then be installed using the `install_pose_pred_dependencies.sh` script into any environment. 
This script also installs MapAnything (https://github.com/facebookresearch/map-anything.git), LightGlue (https://github.com/cvg/LightGlue.git), LoMa (https://github.com/davnords/LoMa.git). Both the LightGlue and LoMa installations are optional and only necessary if you wish to use them.
```
bash install_pose_pred_dependencies.sh 
```
For usage examples refer to `notebooks/Quickstart.ipynb`.

For the licences refer to the conda installation process and the License agreements of the installed repos in `external/*`.

## File Formats
Different datasets and environment representations can directly be created from folders, this section defines the file formats.
The folders from each frame are sorted alphabetically, so best practice is e.g. `0-8`, `000-115`.
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

