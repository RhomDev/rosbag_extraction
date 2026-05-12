import os
import argparse
from tqdm import tqdm

import cv2
from cv_bridge import CvBridge, CvBridgeError

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

import extracte_setup

bridge = CvBridge()


# =========================
# CONFIGURATION
# =========================
def configuration(type_map, topic_names):
    import datetime

    timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

    image_topics = [t for t in topic_names if t.endswith("/image_raw")]

    msg_types = [get_message(type_map[t]) for t in image_topics]

    save_dirs = [
        os.path.join(
            "../../out",
            f"data_{timestamp}",
            t.replace("/ez10_gen1/camera_", "").replace("/image_raw", "")
        )
        for t in image_topics
    ]

    for d in save_dirs:
        os.makedirs(d, exist_ok=True)

    return image_topics, msg_types, save_dirs


# =========================
# EXTRACTION
# =========================
def extraction(reader, topics, msg_types, save_dirs, clean=False):
    messages = []

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic in topics:
            messages.append((topic, data))

    for topic, data in tqdm(messages):
        idx = topics.index(topic)
        msg_type = msg_types[idx]

        # Désérialisation immédiate
        msg = deserialize_message(data, msg_type)

        try:
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            stamp = msg.header.stamp

            # Formatage du nom de fichier
            if not clean:
                # On garde les nanosecondes pour la précision du clean
                filename = f"image_{stamp.sec}_{stamp.nanosec:09d}.jpg"
            else:
                filename = f"image_{stamp.sec}.jpg"

            path = os.path.join(save_dirs[idx], filename)
            cv2.imwrite(path, cv_img)

        except Exception as e:
            print(f"Erreur sur le topic {topic}: {e}")

    print(f"Extraction terminée : {len(messages)} images enregistrées.")


# =========================
# CLEAN SYNCHRO
# =========================
def clean(save_dirs):
    file_sets = [set(os.listdir(d)) for d in save_dirs]

    common_files = set.intersection(*file_sets)

    for folder in save_dirs:
        to_remove = [f for f in os.listdir(folder) if f not in common_files]

        for f in tqdm(to_remove, desc=f"Cleaning {os.path.basename(folder)}"):
            os.remove(os.path.join(folder, f))


# =========================
# MAIN
# =========================
def main(bag, clean_mode):
    print("========= Traitement Image ==========")
    print("Configuration de l'image")
    reader, topic_names, type_map = extracte_setup.configuration(bag)
    topics, msg_types, save_dirs = configuration(type_map, topic_names)

    print("extraction des images")
    extraction(reader, topics, msg_types, save_dirs, clean_mode)

    if clean_mode:
        print("synchronisation des images")
        clean(save_dirs)


# =========================
# ENTRYPOINT
# =========================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Extraction d'images depuis un rosbag2"
    )

    parser.add_argument("--bag", type=str, required=True)

    # FIX: bool argparse correct
    parser.add_argument("--clean", action="store_true",
                        help="Synchronisation des images (supprime non communs)")

    args = parser.parse_args()

    main(args.bag, args.clean)