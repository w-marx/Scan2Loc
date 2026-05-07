## Setup:

# Setup V2

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
### 1080 + vggt (old)
```
cd data_preprocessing
conda env create -f environment_1080.yml -p ./data_prep_env
conda activate ./data_prep_env
```

```
git clone git@github.com:facebookresearch/vggt.git
cd vggt
pip install .
cd ..
```

```
git clone https://github.com/facebookresearch/sam3.git
cd sam3
git checkout 86ed77094094e5cabb16b0414ec60c5ba9ce0a0f
cd ..
```
Then mark the directory `sam3` as sources root
### 3090 + mapanything (new)

```
cd data_preprocessing
conda env create -f environment_1080.yml -p ./data_prep_env
conda activate ./data_prep_env
```

```
git clone https://github.com/facebookresearch/map-anything.git
cd map-anything
pip install -e .
cd ..
```




## For Pose estimation

TODO

```
git clone https://github.com/cvg/LightGlue.git && cd LightGlue
python -m pip install -e .
cd ..
```
