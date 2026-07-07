#!/bin/bash
set -e

echo "Creating external folder"
mkdir -p external


echo "Cloning map-anything"
if [ ! -d "external/map-anything" ]; then
    git clone https://github.com/facebookresearch/map-anything.git external/map-anything
else
    echo "map-anything already exists, skipping clone"
fi


echo "Cloning LightGlue"
if [ ! -d "external/LightGlue" ]; then
    git clone https://github.com/cvg/LightGlue.git external/LightGlue
else
    echo "LightGlue already exists, skipping clone"
fi


echo "Cloning LoMa"
if [ ! -d "external/LoMa" ]; then
    git clone https://github.com/davnords/LoMa.git external/LoMa
else
    echo "LoMa already exists, skipping clone"
fi


echo "Installing shared ..."
pip install -e ./shared


echo "Installing pose estimation"
pip install -e ./headset_localization


echo "Installing LightGlue, LoMa, map-anything"
pip install -e external/map-anything
pip install -e external/LightGlue
pip install -e external/LoMa --no-deps


echo "DONE"