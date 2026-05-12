# Setup:
In general:
```
git clone git@github.com:w-marx/AR-Headset-Localization-in-Robot-Scanned-Workspaces-A-Benchmark-Pipeline.git
cd AR-Headset-Localization-in-Robot-Scanned-Workspaces-A-Benchmark-Pipeline
```

## For data gathering:
```
cd data_gathering
conda env create -f environment.yml -p ./data_gather_env
conda activate ./data_gather_env
```


## For Data preprocessing
Only tested with an 3090ti

```
cd data_preprocessing
conda env create -f environment_3090.yml -p ./data_prep_env
conda activate ./data_prep_env
```

```
git clone https://github.com/facebookresearch/map-anything.git
cd map-anything
pip install -e .
cd ..
```


## For Pose estimation

````commandline
cd pose_estimation
conda env create -f environment.yml -p ./pose_est_env
conda activate ./pose_est_env
````

````commandline
git clone https://github.com/cvg/LightGlue.git && cd LightGlue
python -m pip install -e .
cd ..
````
