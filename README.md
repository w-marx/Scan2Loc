## Setup:


### Setup for the data creation:
Clone the Project:
```
git clone git@github.com:w-marx/AR-Headset-Localization-in-Robot-Scanned-Workspaces-A-Benchmark-Pipeline.git
cd AR-Headset-Localization-in-Robot-Scanned-Workspaces-A-Benchmark-Pipeline
python3.12 -m venv venv
```

Setup the Eviroment and install dependencies:

```
source venv/bin/activate
pip install -r requirements.txt
```

Install vggt:

```
git clone git@github.com:facebookresearch/vggt.git
cd vggt
pip install -e .
cd ..
```

Install sam3:

```
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e ".[train,dev,notebooks]"
cd ..
```

### Additional Setup for the Pose prediction:
```
git clone https://github.com/cvg/LightGlue.git && cd LightGlue
python -m pip install -e .
cd ..
```
