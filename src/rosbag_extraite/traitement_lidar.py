import cv2
import numpy as np

from tqdm import tqdm
import time

import os
from pathlib import Path
import argparse
import shutil

import rosbag2_py
from sensor_msgs_py import point_cloud2

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from setuptools.command import rotate

import extracte_setup


# === CONFIGURATION TYPES ===
def configuration(type_map, topic_name:str):
    root = Path("../../out/save_image_lidar")
    shutil.rmtree(root, ignore_errors=True)

    (root / "map").mkdir(parents=True)
    (root / "face").mkdir(parents=True)
    (root / "video").mkdir(parents=True)

    msg_type = get_message(type_map[topic_name])
    return msg_type


def pointcloud_to_image(pc, width=500, height=500, scale=50):
    img = np.zeros((height, width, 3), dtype=np.uint8)

    x =- pc['y']
    y = pc['x']
    z = pc['z']

    # Intensité

    if 'intensity' in pc:
        intensity = pc['intensity']
    else:
        intensity = np.ones_like(x) * 255

    # Normaliser z pour la couleur (0 -> bleu, max -> rouge)
    z_min, z_max = np.min(z), np.max(z)
    if z_max - z_min < 1e-3:
        z_max = z_min + 1e-3  # éviter division par zéro
    z_norm = ((z - z_min) / (z_max - z_min) * 255).astype(np.uint8)

    # Centrer le nuage dans l'image
    cx, cy = width // 2 , height
    u = (x * scale + cx).astype(np.int32)
    v = (y * scale + cy).astype(np.int32)

    # Filtrer points hors image
    mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, intensity, z_norm = u[mask], v[mask], intensity[mask], z_norm[mask]

    # Dessiner les points
    img[v, u] = np.stack([
        255 - z_norm,
        np.clip(intensity, 0, 255),
        z_norm
    ], axis=1)

    #img = np.rot90(img, k=3)

    return img


def filter_pointcloud(pc, angle_min_deg=-45, angle_max_deg=45,
                      min_dist=0.0, max_dist=10.0):
    x = pc['x']
    y = pc['y']
    z = pc['z']
    intensity = pc['intensity']

    # Angle
    angles = np.degrees(np.arctan2(y, x))
    angle_mask = (angles >= angle_min_deg) & (angles <= angle_max_deg)

    # Distance (zoom)
    distances = np.sqrt(x**2 + y**2)
    dist_mask = (distances >= min_dist) & (distances <= max_dist)

    # Masque combiné
    mask = angle_mask & dist_mask

    if np.sum(mask) == 0:
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
        dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32)]
    )

    if len(points) == 0:
        return None

    pc = {
        'x': points['x'],
        'y': points['y'],
        'z': points['z'],
        'intensity': points['intensity']
    }
    return pc

def transform_data(pc, tx=0.0, ty=0.0, tz=0.0, rx=0.0, ry=0.0, rz=0.0):
    x = pc['x']
    y = pc['y']
    z = pc['z']

    # --- Translation ---
    x = x + tx
    y = y + ty
    z = z + tz

    # --- Rotation X ---
    y2 = y * np.cos(rx) - z * np.sin(rx)
    z2 = y * np.sin(rx) + z * np.cos(rx)
    y, z = y2, z2

    # --- Rotation Y ---
    x2 = x * np.cos(ry) + z * np.sin(ry)
    z2 = -x * np.sin(ry) + z * np.cos(ry)
    x, z = x2, z2

    # --- Rotation Z ---
    x2 = x * np.cos(rz) - y * np.sin(rz)
    y2 = x * np.sin(rz) + y * np.cos(rz)
    x, y = x2, y2

    return {'x': x, 'y': y, 'z': z, 'intensity': pc['intensity']}

def pointcloud_to_image_above(pc, width, height, point_size=2 ):
    #===INITIALISATION===
    img = np.zeros((height, width, 3), dtype=np.uint8)

    x = -pc['x']
    y = pc['y']

    if 'intensity' in pc:
        intensity = pc['intensity']
    else:
        intensity = np.ones_like(x) * 255

    #===CONFIG IMAGE===

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    dx = x_max - x_min
    dy = y_max - y_min

    if dx < 1e-6: dx = 1e-6
    if dy < 1e-6: dy = 1e-6

    scale_x = width / dx
    scale_y = height / dy
    scale = min(scale_x, scale_y) * 0.9

    # Centrer le nuage dans l'image
    cx, cy = width // 2 , (height //2) + height//4
    u = (x * scale + cx).astype(np.int32)
    v = (y * scale + cy).astype(np.int32)

    # Filtrer points hors image
    mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, intensity = u[mask], v[mask], intensity[mask]

    # ===GENERATION IMAGE===

    colors = np.stack([
        np.zeros_like(intensity),
        np.clip(intensity, 0, 255),
        np.zeros_like(intensity)
    ], axis=1)

    for du in range(-point_size, point_size + 1):
        for dv in range(-point_size, point_size + 1):
            if du ** 2 + dv ** 2 <= point_size ** 2:  # forme circulaire
                uu = u + du
                vv = v + dv

                mask = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
                current = img[vv[mask], uu[mask]]
                img[vv[mask], uu[mask]] = np.maximum(current, colors[mask])

    return img

def generate_video():
    folder = Path("../../out/save_image_lidar/face")
    out = Path("../../out/save_image_lidar/video")

    images = sorted([img for img in os.listdir(folder) if img.endswith(".png") or img.endswith(".jpg")])

    first_frame = cv2.imread(str(folder / images[0]))
    height, width = first_frame.shape[:2]

    video = cv2.VideoWriter(
        str(out / "output.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (width, height)
    )

    for img in tqdm(images, desc="Generation Video"):
        path = folder / img
        frame = cv2.imread(str(path))

        if frame is None:
            print("Image cassée:", path)
            continue

        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        video.write(frame)

    video.release()
    cv2.destroyAllWindows()

    print("Vidéo créée avec succès !")

import cv2
import numpy as np

def enhance_lidar_image(
    img,
    crop=True,
    use_colormap=True,
    denoise=True,
    sharpen=True,
    contrast=1.5,
    brightness=10
):

    # --- denoise ---
    if denoise:
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    # --- contrast + brightness ---
    img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)

    # --- crop empty black area ---
    if crop:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)

        coords = cv2.findNonZero(thresh)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            img = img[y:y+h, x:x+w]

    # --- convert to grayscale for colormap ---
    if use_colormap:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)

    # --- sharpen ---
    if sharpen:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        img = cv2.filter2D(img, -1, kernel)

    return img

def pointcloud_to_image_face(pc, width, height, point_size=2):
    # === INITIALISATION ===
    img = np.zeros((height, width, 3), dtype=np.uint8)

    x = -pc['x']
    y = -pc['z']
    z = pc['y']  # 👉 vraie profondeur (comme tu l’as dit)

    if 'intensity' in pc:
        intensity = pc['intensity']
    else:
        intensity = np.ones_like(x) * 255

    # --- clamp intensité ---
    p99 = np.percentile(intensity, 99)
    intensity = np.clip(intensity, 0, p99)

    # === CONFIG IMAGE ===
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    dx = max(x_max - x_min, 1e-6)
    dy = max(y_max - y_min, 1e-6)

    scale = min(width / dx, height / dy) * 0.9

    cx, cy = width // 2, (height // 2) + height // 4

    u = (x * scale + cx).astype(np.int32)
    v = (y * scale + cy).astype(np.int32)

    # === MASK IMAGE ===
    mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)

    u = u[mask]
    v = v[mask]
    z = z[mask]
    intensity = intensity[mask]

    # === COLORATION ===
    import matplotlib.pyplot as plt

    # profondeur réelle (IMPORTANT: z, pas x/y)
    z = z.astype(np.float32)

    z_min, z_max = np.min(z), np.max(z)
    z_norm = (z - z_min) / (z_max - z_min + 1e-6)

    intensity_norm = intensity.astype(np.float32)
    intensity_norm = intensity_norm / (np.max(intensity_norm) + 1e-6)

    cmap = plt.get_cmap("turbo")
    base_colors = cmap(z_norm)[:, :3]

    colors = base_colors * intensity_norm[:, None]
    colors = (np.clip(colors, 0, 1) * 255).astype(np.uint8)

    # === DRAW POINTS ===
    for du in range(-point_size, point_size + 1):
        for dv in range(-point_size, point_size + 1):
            if du ** 2 + dv ** 2 <= point_size ** 2:
                uu = u + du
                vv = v + dv

                mask = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)

                current = img[vv[mask], uu[mask]]
                img[vv[mask], uu[mask]] = np.maximum(current, colors[mask])

    return img


def main(bag:str, topic_name:str):


    output_dir = "../../out/save_image_lidar"

    reader, topic_names, type_map = extracte_setup.configuration(bag)
    msg_type = configuration(type_map, topic_name)

    frame_id = 0
    start_time = time.time()

    x = []
    y = []
    z = []

    messages = []
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic in topic_name:
            messages.append((topic, data))

    for topic, data in tqdm(messages, desc="Extraction images"):
        if topic != topic_name:
            continue

        msg = deserialize_message(data, msg_type)

        pc = read_data_lidar(msg)
        if pc is None:
            continue

        x.append(pc['x'])
        y.append(pc['y'])
        z.append(pc['z'])
    # -135 ; -45
        pc_filtree = filter_pointcloud(pc,-135,-45, 1.0, 30.0)
        if pc_filtree is None:
            continue

      #  pc_filtree = transform_data(pc,tz=0.358, rx=np.deg2rad(15.756))
        pc_filtree_face = transform_data(pc_filtree, 1.460, 0.010, 1.920,np.deg2rad(15.756), np.deg2rad(-0.229), np.deg2rad(3.908))


      #  img = pointcloud_to_image(pc_filtree,720,480,50)
        img_map = pointcloud_to_image_above(pc_filtree,1280,720)
        img_face = pointcloud_to_image_face(pc_filtree_face,1280,720)

        img_face = enhance_lidar_image(img_face)

        filename = f"{output_dir}/map/frame_{frame_id:06d}.png"
        cv2.imwrite(filename, img_map)
        filename = f"{output_dir}/face/frame_{frame_id:06d}.png"
        cv2.imwrite(filename, img_face)

        frame_id += 1

    end_time = time.time()
    print(f"\n✅ {frame_id} images générées en {end_time - start_time:.2f}s")
    print(f"⚡ FPS moyen : {frame_id / (end_time - start_time):.2f}")

    print(f"taille : {len(x)}")

    x_concat = np.concatenate(x)

    # Maintenant on peut calculer min/max
    print("Min global x :", np.min(x_concat))
    print("Max global x :", np.max(x_concat))


    y_concat = np.concatenate(y)

    # Maintenant on peut calculer min/max
    print("Min global y :", np.min(y_concat))
    print("Max global y :", np.max(y_concat))


    z_concat = np.concatenate(z)

    # Maintenant on peut calculer min/max
    print("Min global z :", np.min(z_concat))
    print("Max global z :", np.max(z_concat))

    generate_video()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Traitement d'images des ROS2 bag")
    # default="../../rosbag/ez10_gen1_sensors_2026-04-02-16-45-39"
    parser.add_argument("--bag", type=str, required=True,
                        help="Chemin du rosbag")
    # default="/ez10_gen1/hesai_front/cloud"
    parser.add_argument("--topic", type=str, default="/ez10_gen1/hesai_front/cloud",
                        help="nom du topic lidar")

    parser.add_argument("--enchance", type=bool, default=False,
                        help="netoyer et améliorer la qualité d'image")

    parser.add_argument("--video", type=bool, default=False,
                        help="genrer une video")

    args = parser.parse_args()

    main(args.bag, args.topic)


