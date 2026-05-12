import os
import time
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import rosbag2_py
from sensor_msgs_py import point_cloud2
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

import extracte_setup


# =========================
# CONFIGURATION
# =========================
def configuration(type_map, topic_name: str):
    root = Path("../../out/save_image_lidar_ring")

    shutil.rmtree(root, ignore_errors=True)
    (root / "face").mkdir(parents=True, exist_ok=True)
    (root / "video").mkdir(parents=True, exist_ok=True)

    msg_type = get_message(type_map[topic_name])
    return msg_type


def read_data_lidar(msg):

    available_fields = [f.name for f in msg.fields]

    required = ["x", "y", "z", "intensity", "ring"]

    for field in required:
        if field not in available_fields:
            raise ValueError(f"Champ manquant: {field}")

    points = np.array(
        list(
            point_cloud2.read_points(
                msg,
                field_names=required,
                skip_nans=False
            )
        ),
        dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('intensity', np.float32),
            ('ring', np.uint16)
        ]
    )

    if points.size == 0:
        return None

    return points

def lidar_to_range_image(
    points,
    width=2048,
    out_height=256,
    fov_deg=60,
    value="intensity"
):
    x = points['x']
    y = points['y']
    z = points['z']

    ring = points['ring']
    intensity = points['intensity']

    # =========================
    # ANGLE HORIZONTAL
    # =========================
    azimuth = np.degrees(np.arctan2(y, x))

    half_fov = fov_deg / 2

    mask = np.abs(azimuth) <= half_fov

    x = x[mask]
    y = y[mask]
    z = z[mask]
    ring = ring[mask]
    intensity = intensity[mask]
    azimuth = azimuth[mask]

    # =========================
    # PROJECTION HORIZONTALE
    # =========================
    u = ((azimuth + half_fov) / fov_deg) * width

    u = np.clip(u.astype(np.int32), 0, width - 1)

    # =========================
    # VERTICAL
    # =========================
    v = ring.astype(np.int32)

    # inversion éventuelle
    v = 31 - v

    # =========================
    # IMAGE
    # =========================
    img = np.zeros((32, width), dtype=np.uint8)

    # =========================
    # VALEURS
    # =========================
    if value == "distance":

        dist = np.sqrt(x**2 + y**2 + z**2)

        values = 255 * (1.0 - dist / 80.0)

    else:

        values = np.log1p(intensity)

        values = 255 * values / np.max(values)

    values = values.astype(np.uint8)

    # =========================
    # REMPLISSAGE
    # =========================
    for i in range(len(u)):
        img[v[i], u[i]] = max(
            img[v[i], u[i]],
            values[i]
        )

    # =========================
    # INTERPOLATION VERTICALE
    # =========================
    img = cv2.resize(
        img,
        (width, out_height),
        interpolation=cv2.INTER_LINEAR
    )

    return img

def filter_fov(
    points,
    fov_deg=60,
    min_distance=0.5,
    max_distance=50.0
):

    x = points['x']
    y = points['y']
    z = points['z']

    # =========================
    # ANGLE HORIZONTAL
    # =========================
    azimuth = np.degrees(np.arctan2(y, x))

    half_fov = fov_deg / 2.0

    fov_mask = np.abs(azimuth) <= half_fov

    # =========================
    # DISTANCE
    # =========================
    distance = np.sqrt(x**2 + y**2 + z**2)

    distance_mask = (
        (distance >= min_distance) &
        (distance <= max_distance)
    )

    # =========================
    # MASQUE FINAL
    # =========================
    mask = fov_mask & distance_mask

    return points[mask]

# =========================
# VIDEO
# =========================
def generate_video():
    folder = Path("../../out/save_image_lidar_ring/face")
    out = Path("../../out/save_image_lidar_ring/video")

    images = sorted([f for f in os.listdir(folder)
                     if f.endswith((".png", ".jpg"))])

    if not images:
        print("Aucune image trouvée.")
        return

    first = cv2.imread(str(folder / images[0]))
    h, w = first.shape[:2]

    video = cv2.VideoWriter(
        str(out / "output.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (w, h)
    )

    for img in tqdm(images, desc="Generation video"):
        frame = cv2.imread(str(folder / img))
        if frame is None:
            continue

        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        video.write(frame)

    video.release()
    print("Vidéo créée avec succès.")



# =========================
# MAIN
# =========================
def main(args):

    output_dir = Path("../../out/save_image_lidar_ring")
    reader, topic_names, type_map = extracte_setup.configuration(args.bag)

    msg_type = configuration(type_map, args.topic)

    messages = []

    print("Chargement des données")

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic == args.topic:
            messages.append((topic, data))

    frame_id = 0

    for topic, data in tqdm(messages, desc="Extraction images"):

        msg = deserialize_message(data, msg_type)
        pc = read_data_lidar(msg)
        if pc is None:
            continue

        pc = filter_fov(
            pc,
            fov_deg=60,
            min_distance=2.0,
            max_distance=30.0
        )
        if pc is None:
            continue

        img = lidar_to_range_image(
            pc
        )

        cv2.imwrite(
            str(output_dir / "face" / f"frame_{frame_id:06d}.png"),
            img
        )

        frame_id += 1

    generate_video()

# =========================
# ENTRYPOINT
# =========================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--topic", default="/ez10_gen1/hesai_front/cloud")
    parser.add_argument("--transf_trans", type=float, nargs=3,default=(1.460, 0.010, 1.920))
    parser.add_argument("--transf_rot", type=float, nargs=3, default=(15.756, -0.229, 3.908))
    parser.add_argument("--enchance", action="store_true")
    parser.add_argument("--video", action="store_true")

    args = parser.parse_args()

    main(args)