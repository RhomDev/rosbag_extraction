import pyvista as pv
import numpy as np
import time
import argparse
import shutil
from pathlib import Path
import cv2
from tqdm import tqdm

import rosbag2_py
from sensor_msgs_py import point_cloud2
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

import extracte_setup

# ==================
#   CONFIGURATION ET RECUPERATION DE DONNE
# ==================

def configuration(type_map, topic_name: str):
    root = Path("../../../out/save_image_lidar")

    shutil.rmtree(root, ignore_errors=True)
    (root / "face").mkdir(parents=True, exist_ok=True)
    (root / "video").mkdir(parents=True, exist_ok=True)

    msg_type = get_message(type_map[topic_name])
    return msg_type

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

    return np.vstack((points['x'], points['y'], points['z'])).T.astype(np.float64)

def update_global_map(global_map, new_points, voxel_size=0.1):

    if new_points is None or len(new_points) == 0:
        return global_map

    # convertir en point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(new_points)

    # fusion
    global_map += pcd

    # compression (ESSENTIEL)
    global_map = global_map.voxel_down_sample(voxel_size)

    return global_map


# ==================
#   GNSS
# ==================

R = 6378137.0  # rayon Terre (m)

def gnss_to_local(lat, lon, alt, lat0, lon0, alt0):
    lat = np.radians(lat)
    lon = np.radians(lon)
    lat0 = np.radians(lat0)
    lon0 = np.radians(lon0)

    dlat = lat - lat0
    dlon = lon - lon0

    x = dlon * R * np.cos(lat0)  # Est
    y = dlat * R                 # Nord
    z = alt - alt0               # Up

    return x, y, z

def quaternion_to_euler(x, y, z, w):
    # Roll (X)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    # Pitch (Y)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    # Yaw (Z)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    return roll_x, pitch_y, yaw_z




# ==================
#   FILTRE ET TRAITEMENT
# ==================
def transform_data(pc, tx, ty, tz, rx, ry, rz):

    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]

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

    # Translation APRÈS rotation
    x += tx
    y += ty
    z += tz

    return np.vstack((x, y, z)).T.astype(np.float64)
def gnss_pose_to_transform(gnss_msg, lat0, lon0, alt0):
    lat = gnss_msg.pose.pose.position.y
    lon = gnss_msg.pose.pose.position.x
    alt = gnss_msg.pose.pose.position.z

    q = gnss_msg.pose.pose.orientation

    # Position locale
    tx, ty, tz = gnss_to_local(lat, lon, alt, lat0, lon0, alt0)

    # Orientation
    rx, ry, rz = quaternion_to_euler(q.x, q.y, q.z, q.w)

    return tx, ty, tz, rx, ry, rz

# ==================
#   GESTION MESH
# ==================

def pointcloud_to_mesh_pyvista(points, voxel_size=0.1, alpha=0.5):
    if points is None or len(points) < 10:
        return None

    # Créer un nuage PyVista
    cloud = pv.PolyData(points)

    # 1. Nettoyage : suppression des doublons et sous-échantillonnage
    cloud = cloud.clean(tolerance=voxel_size / 2)  # fusion des points très proches
    cloud = cloud.subsample(voxel_size)  # équivalent du voxel_down_sample

    # 2. Estimation des normales (nécessaire pour l'extraction de surface)
    cloud = cloud.compute_normals(point_normals=True, auto_orient_normals=True)

    # 3. Reconstruction par Delaunay 3D + extraction de l'enveloppe (alpha shape)
    #    Cette méthode est très efficace pour les nuages denses et non structurés.
    volume = cloud.delaunay_3d(alpha=alpha, progress_bar=True)
    mesh = volume.extract_geometry(progress_bar=True)

    # 4. Nettoyage du maillage final
    mesh = mesh.clean()
    mesh = mesh.fill_holes(hole_size=voxel_size * 5)  # bouche les petits trous
    mesh = mesh.smooth(n_iter=20)  # lissage optionnel

    return mesh

def visualize_mesh_pyvista(mesh):
    """Visualise un maillage PyVista."""
    if mesh is None or mesh.n_points == 0:
        print("Aucun maillage à afficher.")
        return

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='lightblue', show_edges=False, smooth_shading=True)
    plotter.add_axes()
    plotter.show_grid()
    plotter.show()