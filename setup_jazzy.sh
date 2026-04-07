#!/bin/bash
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate

export PYTHONPATH=$PYTHONPATH:/opt/ros/jazzy/lib/python3.12/site-packages
export PYTHONPATH=$PYTHONPATH:~/Documents/stage/ropradeau_ws_extrait/install/lib/python3.12/site-packages

cd src/rosbag_extraite