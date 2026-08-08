## AR-Headset-Localization-in-Robot-Scanned-Workspaces-A-Benchmark-Pipeline

-----
For gaze based HRI in scanned desktop environments, using AR-Headsets it is necessary to locate the headset in the environment.
This repo offers a comprehensive pipeline for marker-free localization of an egocentric headset within a robot-scanned environment.
Furthermore it offers three inference ready localizers: a baseline PnP approach using keypoint matching, a PnP+L variant that refines estimates with line features, and a PnP+Ellipsoids variant leveraging object bounding boxes. 
-----

### Provided Functionalities:

#### Datasets
Multiple desktop datasets are provided in the `example_datasets` folder. Consisting of multiple scenes and matching ``.vrs` recordings in them.

##### Dataset creation

##### Three Inference ready Localizers

##### Evaluation Framework

## Getting Started
##### Cloning Repo
```
git clone git@github.com:w-marx/AR-Headset-Localization-in-Robot-Scanned-Workspaces-A-Benchmark-Pipeline.git
cd AR-Headset-Localization-in-Robot-Scanned-Workspaces-A-Benchmark-Pipeline
```

##### For dataset creation
To create your own dataset the following dependencies need to be installed:
```
cd data_gathering
conda env create -f environment.yml -p ./data_gather_env
conda activate ./data_gather_env
pip install -e ../shared
```
For example usage refer to ``notebooks/DataGathering.ipynb`


#### For Inference and Evaluation

```
conda env create -f env_3090.yml -p ./env
conda activate ./env
bash install_pose_pred_dependencies.sh 
```

## File Formats
