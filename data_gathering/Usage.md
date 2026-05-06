# Data gathering

This module provides the function to gather data using a panda2robot in the following format:

```
`output-folder`
├── headset.vrs
├── robot
│   └── [000000 : number of positions]
│       ├──  rgb.png
│       ├──  depth.npy
│       └──  poses.json
├── metadata.json
└── robot_cam_calibration.json
```

## Arguments:

`--no-data-gathering` can be used to optimize an output folder without needing the robot & `deoxys`

`--stabilisation-timeout` Timeout in seconds between robot moved to position and picture is taken, doesnt seem to improve performance in any way

### Folder locations
`--output-folder`: specifies the path where to create the folder

`--cam-t-gripper-path`: specifies the path to a `cam_t_gripper.npy` if one should be used else can be left to `None` (then it will be estimated)

### Markers
The programm supports both `ArUCo` and `Charuco` or `None` for data gathering, this can be selected using the `--marker-detection` argument.

Aruco

1. Use `--aruco-marker-side-length` to specify the marker side length in meter
2. Use `--aruco-marker-dictionary` to specify the dictionary type e.g. "5X5_250"

Charuco

1. Use `--aruco-marker-dictionary` to specify the dictionary type e.g. "5X5_250"
2. Use `--aruco-marker-side-length` to specify the side length of the markers in meter
3. Use `--charuco-square-side-length` to specify the size of a chessboard tile in meters
4. Use `--charuco-board-size` to specify the number of markers on the board

None

### Result analysation
parser.add_argument("--pose-outlier-quants", type=float, default=[0.2,0.2], nargs=2, help="The quantiles of base_t_marker estimates to remove 1st arg: translation, 2nd arg: rotation")

parser.add_argument("--no-result-analysation", action = "store_false", help = "If used there wont by any result analysation (pose deviation analysis)", dest = "analyze_results")
parser.add_argument("--no-pdf-store", action = "store_false", help = "If used the analysis wont be stored into the dataset", dest = "save_analysis_results")
parser.add_argument("--analysis-baseline", type=str, default="median", help = "Decides wheather to assume that the mean or the median is assumed to be the actual base_t_marker cam be `median`/`mean` in analysis")



# Examply usages:
## With charuco boards:
### Small board:
With the small board (with data gathering) and output folder charuco1`
```
python robot_data_gathering.py --output-folder charuco1 --marker-detection Charuco --aruco-marker-side-length 0.0146 --charuco-square-side-length 0.0188 --charuco-board-size 14 9 --aruco-marker-dictionary 5X5_250 --no-data-gathering
```

Without data gathering:
```
/opt/miniconda3/envs/deoxys/bin/python /workspace/franka_pipeline/robot_data_gathering.py --output-folder charuco1 --marker-detection Charuco --aruco-marker-side-length 0.0146 --charuco-square-side-length 0.0188 --charuco-board-size 14 9 --aruco-marker-dictionary 5X5_250 --no-data-gathering
```

### Big board:

## With aruco markers:
### Small marker:
```
/opt/miniconda3/envs/deoxys/bin/python /workspace/franka_pipeline/robot_data_gathering.py --output-folder small_aruco1 --marker-detection Aruco --aruco-marker-side-length 0.072 --aruco-marker-dictionary 6X6_250 
```

### Big marker:
```
/opt/miniconda3/envs/deoxys/bin/python /workspace/franka_pipeline/robot_data_gathering.py --output-folder big_aruco1 --marker-detection Aruco --aruco-marker-side-length 0.146 --aruco-marker-dictionary 6X6_250 
```
