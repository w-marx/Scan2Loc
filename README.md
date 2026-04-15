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

Now open the project in Pycharm

Install vggt:

```
git clone git@github.com:facebookresearch/vggt.git
cd vggt
pip install .
cd ..
```

Install sam3:

```
git clone https://github.com/facebookresearch/sam3.git
cd sam3
git checkout 86ed77094094e5cabb16b0414ec60c5ba9ce0a0f
cd ..
```
Then mark the directory `sam3` as sources root

### Additional Setup for the Pose prediction:
```
git clone https://github.com/cvg/LightGlue.git && cd LightGlue
python -m pip install -e .
cd ..
```
