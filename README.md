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
pip install -e ../shared
```

## For Headset localisation & data preprocessing

```
conda env create -f env_3090.yml -p ./env
conda activate ./env
bash install_pose_pred_dependencies.sh 
```

Then select this environment for the notebooks