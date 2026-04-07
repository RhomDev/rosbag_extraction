import os
import argparse

from tqdm import tqdm

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from cv_bridge import CvBridge, CvBridgeError
import cv2

import extracte_setup

bridge = CvBridge()

# === CONFIGURATION TYPES ===
def configuration(type_map, topic_names):
    import datetime
    maintenant = datetime.datetime.now()
    format_specifique = maintenant.strftime("%d_%m_%Y_%H_%M_%S")
    topic_names_image = [t for t in topic_names if t.endswith('/image_raw')]

    msg_types = [get_message(type_map[name]) for name in topic_names_image]
    save_dirs = [f"../../out/data_{format_specifique}/" + name.replace("/ez10_gen1/camera_", "", 1).replace("/image_raw", "", 1)
                 for name in topic_names_image]

    for save in save_dirs:
        os.makedirs(""+save, exist_ok=True)

    return topic_names_image, msg_types, save_dirs

# === EXTRACTION DES IMAGES ===
def extraction(reader, topics: list[str], msg_types: list, save_dirs: list[str], max:bool):

    messages = []
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic in topics:
            messages.append((topic, data))

    # Parcours avec barre de progression
    for topic, data in tqdm(messages, desc="Extraction images"):
        idx = topics.index(topic)
        msg = deserialize_message(data, msg_types[idx])
        try:
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            if not max:
                filename = os.path.join(save_dirs[idx],
                                        f"image_{msg.header.stamp.sec}_{msg.header.stamp.nanosec}.jpg")
            else:
                filename = os.path.join(save_dirs[idx],
                                    f"image_{msg.header.stamp.sec}.jpg")
            cv2.imwrite(filename, cv_img)
        except CvBridgeError as e:
            print(e)


# === CLEAN DES DOSSIERS ===
def clean(save_dirs: list[str]):
    files_sets = [set(os.listdir(d)) for d in save_dirs]
    common_files = set.intersection(*files_sets)

    for folder in save_dirs:
        files_to_remove = [f for f in os.listdir(folder) if f not in common_files]
        for f in tqdm(files_to_remove, desc=f"Suppression dans {folder}"):
            os.remove(os.path.join(folder, f))

# === MAIN ===
def main(reader, topic_names, type_map, max):
    import time
    global topics_list, msg_types, save_dirs
    # Étapes avec barre de progression globale
    steps = ["Configuration", "Extraction", "Clean"]
    for step in tqdm(steps, desc="Progression Traitement Image", ncols=80):
        if step == "Configuration":
            topics_list, msg_types, save_dirs = configuration(type_map, topic_names)
            time.sleep(0.3)
        elif step == "Extraction":
            extraction(reader, topics_list, msg_types, save_dirs, max)
            time.sleep(0.3)
        elif step == "Clean" and max:
            clean(save_dirs)
            time.sleep(0.3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traitement d'images des ROS2 bag")
    # default="../../rosbag/ez10_gen1_sensors_2026-04-02-16-45-39"
    parser.add_argument("--bag", type=str, required=True, help="Chemin du rosbag")
    parser.add_argument("--clean", type=bool, default=False, help="synchronisation des images par temp de seconde")
    args = parser.parse_args()

    reader, topic_names, type_map = extracte_setup.configuration(args.bag)

    main(reader, topic_names, type_map, args.clean)