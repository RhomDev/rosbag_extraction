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


def read_data_lidar_by_ring(msg):
    # 1. On lit toutes les données d'un coup, y compris le 'ring'
    # Utiliser 'reshape' ou 'frombuffer' est plus rapide,
    # mais read_points reste pratique si on filtre les champs.
    gen = point_cloud2.read_points(
        msg,
        field_names=("x", "y", "z", "intensity", "ring"),
        skip_nans=True
    )

    # Conversion efficace en tableau structuré
    points = np.array(list(gen), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'f4'), ('ring', 'u2')
    ])

    if points.size == 0:
        return None

    # 2. Séparation par ring
    # On récupère la liste unique des IDs de lasers présents (0, 1, 2...)
    unique_rings = np.unique(points['ring'])

    rings_dict = {}

    for r in unique_rings:
        # Masque pour isoler le ring actuel
        mask = (points['ring'] == r)

        # On stocke sous forme de dictionnaire ou de tableau Nx3
        rings_dict[r] = {
            'x': points['x'][mask],
            'y': points['y'][mask],
            'z': points['z'][mask],
            'intensity': points['intensity'][mask]
        }

    return rings_dict

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


def transform_data(pc, tx, ty, tz,
                   rx, ry, rz):

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
def main(args):
    output_dir = Path("../../out/save_image_lidar")
    shutil.rmtree(output_dir, ignore_errors=True)
    (output_dir / "face").mkdir(parents=True)

    reader, _, type_map = extracte_setup.configuration(args.bag)
    msg_type = get_message(type_map[args.topic])

    frame_id = 0
    start_time = time.time()

    # On ne stocke pas tout en mémoire (messages), on traite au fil de l'eau pour éviter le crash RAM
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic != args.topic: continue

        msg = deserialize_message(data, msg_type)
        pc = read_pc2_fast(msg)

        # Filtrage
        pc = filter_pointcloud(pc, args.angle[0], args.angle[1], 1.0, 30.0)
        if len(pc) == 0: continue

        # Transformation
        pts, intensity = transform_data(pc, args.transf_trans, args.transf_rot)

        # Rendu
        img = pointcloud_to_image_face(pts, intensity, 1280, 720)

        if args.enchance:
            img = enhance_lidar_image(img)

        cv2.imwrite(str(output_dir / "face" / f"frame_{frame_id:06d}.png"), img)
        frame_id += 1

    print(f"Terminé. FPS: {frame_id / (time.time() - start_time):.2f}")
    if args.video: generate_video()

# =========================
# ENTRYPOINT
# =========================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--topic", default="/ez10_gen1/hesai_front/cloud")
    parser.add_argument("--angle", nargs=2, type=float, required=True)
    parser.add_argument("--transf_trans", type=float, nargs=3,default=(1.460, 0.010, 1.920))
    parser.add_argument("--transf_rot", type=float, nargs=3, default=(15.756, -0.229, 3.908))
    parser.add_argument("--enchance", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--ring", action="store_true")

    args = parser.parse_args()

    main(args.bag, args.topic, args.angle, args.enchance, args.video, args.transf_trans, args.transf_rot, args.ring)