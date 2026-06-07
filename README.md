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

## For Pose estimation & data preprocessing
In `./pose_estimation`:

```
conda env create -f env_3090.yml -p ./env
conda activate ./env
pip install -e ../shared
```

```
git clone https://github.com/facebookresearch/map-anything.git
cd map-anything
pip install -e .
cd ..
```

When using LightGlue for matching:
````commandline
git clone https://github.com/cvg/LightGlue.git && cd LightGlue
python -m pip install -e .
cd ..
````

When using LoMa for matching:
````commandline
git clone https://github.com/davnords/LoMa.git && cd LoMa
pip install -e . --no-deps
cd ..
pip install einops tyro
````