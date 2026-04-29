# How to use:
(ecsamples)

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
