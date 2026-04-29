import subprocess
import os

script_path = "/home/rhomdev/Documents/stage/rosbag_extraction/src/rosbag_extraite/lidar/lidar_mapping_mesh_separation_optimized_1.py"

cmd = f"""
source /opt/ros/jazzy/setup.bash
python3 {script_path}
"""

subprocess.run(cmd, shell=True, executable="/bin/bash")