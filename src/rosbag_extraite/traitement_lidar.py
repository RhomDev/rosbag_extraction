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
    root = Path("../../out/save_image_lidar")

    shutil.rmtree(root, ignore_errors=True)
    (root / "face").mkdir(parents=True, exist_ok=True)
    (root / "video").mkdir(parents=True, exist_ok=True)

    msg_type = get_message(type_map[topic_name])
    return msg_type


# =========================
# POINTCLOUD PROCESSING
# =========================
def filter_pointcloud(pc, angle_min_deg=-45, angle_max_deg=45,
                      min_dist=0.0, max_dist=10.0):

    x, y, z = pc['x'], pc['y'], pc['z']
    intensity = pc['intensity']

    angles = np.degrees(np.arctan2(y, x))
    distances = np.sqrt(x**2 + y**2)

    mask = (
        (angles >= angle_min_deg) & (angles <= angle_max_deg) &
        (distances >= min_dist) & (distances <= max_dist)
    )

    if not np.any(mask):
        return None

    return {
        'x': x[mask],
        'y': y[mask],
        'z': z[mask],
        'intensity': intensity[mask]
    }


def read_data_lidar(msg):
    points = np.fromiter(
        point_cloud2.read_points(
            msg,
            field_names=("x", "y", "z", "intensity"),
            skip_nans=True
        ),
        dtype=[('x', np.float32), ('y', np.float32),
               ('z', np.float32), ('intensity', np.float32)]
    )

    if len(points) == 0:
        return None

    return {
        'x': points['x'],
        'y': points['y'],
        'z': points['z'],
        'intensity': points['intensity']
    }


def transform_data(pc, tx=0.0, ty=0.0, tz=0.0,
                   rx=0.0, ry=0.0, rz=0.0):

    x, y, z = pc['x'] + tx, pc['y'] + ty, pc['z'] + tz

    # Rotation X
    y, z = (
        y * np.cos(rx) - z * np.sin(rx),
        y * np.sin(rx) + z * np.cos(rx)
    )

    # Rotation Y
    x, z = (
        x * np.cos(ry) + z * np.sin(ry),
        -x * np.sin(ry) + z * np.cos(ry)
    )

    # Rotation Z
    x, y = (
        x * np.cos(rz) - y * np.sin(rz),
        x * np.sin(rz) + y * np.cos(rz)
    )

    return {'x': x, 'y': y, 'z': z, 'intensity': pc['intensity']}


# =========================
# IMAGE PROCESSING
# =========================
def enhance_lidar_image(img,
                        crop=True,
                        use_colormap=True,
                        denoise=True,
                        sharpen=True,
                        contrast=1.5,
                        brightness=10):

    if denoise:
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)

    if crop:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)

        coords = cv2.findNonZero(thresh)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            img = img[y:y + h, x:x + w]

    if use_colormap:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)

    if sharpen:
        kernel = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])
        img = cv2.filter2D(img, -1, kernel)

    return img


def pointcloud_to_image_face(pc, width, height, point_size=2):

    img = np.zeros((height, width, 3), dtype=np.uint8)

    x = -pc['x']
    y = -pc['z']
    z = pc['y']
    intensity = pc.get('intensity', np.ones_like(x) * 255)

    # clamp intensity
    p99 = np.percentile(intensity, 99)
    intensity = np.clip(intensity, 0, p99)

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    scale = min(width / max(x_max - x_min, 1e-6),
                height / max(y_max - y_min, 1e-6)) * 0.9

    cx, cy = width // 2, height // 2 + height // 4

    u = (x * scale + cx).astype(np.int32)
    v = (y * scale + cy).astype(np.int32)

    mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)

    u, v = u[mask], v[mask]
    z = z[mask]
    intensity = intensity[mask]

    import matplotlib.pyplot as plt

    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
    intensity_norm = intensity / (intensity.max() + 1e-6)

    cmap = plt.get_cmap("turbo")
    colors = cmap(z_norm)[:, :3]
    colors = (colors * intensity_norm[:, None] * 255).astype(np.uint8)

    for du in range(-point_size, point_size + 1):
        for dv in range(-point_size, point_size + 1):
            if du * du + dv * dv <= point_size * point_size:
                uu = u + du
                vv = v + dv

                m = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
                img[vv[m], uu[m]] = np.maximum(img[vv[m], uu[m]], colors[m])

    return img


# =========================
# VIDEO
# =========================
def generate_video():
    folder = Path("../../out/save_image_lidar/face")
    out = Path("../../out/save_image_lidar/video")

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
def main(bag: str, topic_name: str, angle, enhance, video):

    output_dir = Path("../../out/save_image_lidar")
    reader, topic_names, type_map = extracte_setup.configuration(bag)

    msg_type = configuration(type_map, topic_name)

    messages = []

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic == topic_name:
            messages.append((topic, data))

    frame_id = 0
    start = time.time()

    for topic, data in tqdm(messages, desc="Extraction images"):

        msg = deserialize_message(data, msg_type)
        pc = read_data_lidar(msg)
        if pc is None:
            continue

        pc = filter_pointcloud(pc, angle[0], angle[1], 1.0, 30.0)
        if pc is None:
            continue

        pc = transform_data(
            pc,
            1.460, 0.010, 1.920,
            np.deg2rad(15.756),
            np.deg2rad(-0.229),
            np.deg2rad(3.908)
        )

        img = pointcloud_to_image_face(pc, 1280, 720)

        if enhance:
            img = enhance_lidar_image(img)

        cv2.imwrite(
            str(output_dir / "face" / f"frame_{frame_id:06d}.png"),
            img
        )

        frame_id += 1

    end = time.time()

    print(f"\n{frame_id} images générées en {end - start:.2f}s")
    print(f"FPS moyen : {frame_id / (end - start):.2f}")

    if video:
        generate_video()


# =========================
# ENTRYPOINT
# =========================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--topic", default="/ez10_gen1/hesai_front/cloud")
    parser.add_argument("--angle", nargs=2, type=float, required=True)
    parser.add_argument("--enhance", action="store_true")
    parser.add_argument("--video", action="store_true")

    args = parser.parse_args()

    main(args.bag, args.topic, args.angle, args.enhance, args.video)