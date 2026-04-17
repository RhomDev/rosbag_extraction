import time
import argparse
import shutil
from pathlib import Path

import open3d as o3d

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

def filter_pointcloud(pc, angle_min_deg=-45, angle_max_deg=45,
                      min_dist=0.0, max_dist=10.0):
    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]

    angles = np.degrees(np.arctan2(y, x))
    distances = np.sqrt(x**2 + y**2)

    mask = (
        (angles >= angle_min_deg) & (angles <= angle_max_deg) &
        (distances >= min_dist) & (distances <= max_dist)
    )

    if not np.any(mask):
        return None

    return np.vstack((x, y, z)).T.astype(np.float64)

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


# =========================
# Convertion PointCloud -> Mesh
# =========================
def pointCloud_by_mesh(all_front, all_rear=None):

    # -------------------------
    # Cas 1 : un seul nuage
    # -------------------------
    if all_rear is None or len(all_rear) < 10:

        if all_front is None or len(all_front) < 10:
            return None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_front)

        # gris
        pcd.paint_uniform_color([0.5, 0.5, 0.5])

    # -------------------------
    # Cas 2 : deux nuages
    # -------------------------
    else:
        if all_front is None or len(all_front) < 10:
            return None

        if len(all_rear) < 10:
            return None

        # Front (bleu)
        pcd_front = o3d.geometry.PointCloud()
        pcd_front.points = o3d.utility.Vector3dVector(all_front)
        pcd_front.paint_uniform_color([0, 0, 1])

        # Rear (rouge)
        pcd_rear = o3d.geometry.PointCloud()
        pcd_rear.points = o3d.utility.Vector3dVector(all_rear)
        pcd_rear.paint_uniform_color([1, 0, 0])

        # Fusion
        pcd = pcd_front + pcd_rear

    # -------------------------
    # Traitement commun
    # -------------------------
    pcd = pcd.voxel_down_sample(0.1)

    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(50)

    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)

    radii = [avg_dist, avg_dist * 2]

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector(radii)
    )

    return mesh

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


def clean_and_merge_points(list1, list2=None, voxel_size=0.05, tolerance=None):

    # -------------------------
    # Merge propre
    # -------------------------
    if list2 is not None:
        points = np.vstack((list1, list2))
    else:
        points = list1

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 1. downsample (ESSENTIEL)
    pcd = pcd.voxel_down_sample(voxel_size)

    # 2. outliers removal (TRÈS IMPORTANT pour LiDAR)
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=2.0)

    return np.asarray(pcd.points)

def preprocess_cloud(pc):
    # supprime points trop proches / bruit GNSS
    dist = np.linalg.norm(pc, axis=1)
    pc = pc[dist > 0.2]

    return pc

# =========================
# MAIN
# =========================
def main(bag: str, topic_name_front: str, topic_name_rear: str):

    reader, topic_names, type_map = extracte_setup.configuration(bag)
    msg_type_front = configuration(type_map, topic_name_front)
    msg_type_rear = configuration(type_map, topic_name_rear)


    list_pc_front = []
    list_pc_rear = []
    messages = []

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic == topic_name_front or topic == topic_name_rear:
            messages.append((topic, data))

    global_map = o3d.geometry.PointCloud()

    for topic, data in tqdm(messages, desc="Extraction data lidar front"):
        if topic == topic_name_front:
            msg = deserialize_message(data, msg_type_front)
            pc = read_data_lidar(msg)
            if pc is None:
                continue
            pc = filter_pointcloud(pc, -180, 180, 1.0, 30.0)
            if pc is None:
                continue
            pc = transform_data(pc, 1.460, -2, 1.920, 0.275, -0.004, 1.639)
            pc = preprocess_cloud(pc)
            global_map = update_global_map(global_map, pc)
        if topic == topic_name_rear:
            msg = deserialize_message(data, msg_type_rear)
            pc = read_data_lidar(msg)
            if pc is None:
                continue
            pc = filter_pointcloud(pc, -180, 180, 1.0, 30.0)
            if pc is None:
                continue
            pc = transform_data(pc, -1.460, 0, 1.910, 0.254, 0.024, -1.540)
            pc = preprocess_cloud(pc)
            global_map = update_global_map(global_map, pc)

    print("Compactage de donnée")

    # Vérifier si le global_map contient des points
    if len(global_map.points) == 0:
        print("Aucun point dans la map globale")
        return

    # Option 1: Ball Pivoting (plus stable)
    try:
        global_map.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.5, max_nn=30
            )
        )
        global_map.orient_normals_consistent_tangent_plane(100)

        distances = global_map.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)

        radii = [avg_dist * 2, avg_dist * 4, avg_dist * 8]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            global_map,
            o3d.utility.DoubleVector(radii)
        )
        print("Mesh créé avec Ball Pivoting")

    except Exception as e:
        print(f"Ball Pivoting a échoué: {e}")
        # Option 2: Poisson reconstruction
        try:
            global_map.estimate_normals()
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                global_map,
                depth=8,
                scale=1.1,
                linear_fit=False
            )

            # Nettoyer les vertices avec faible densité
            if len(densities) > 0:
                threshold = np.quantile(densities, 0.05)
                vertices_to_remove = densities < threshold
                mesh.remove_vertices_by_mask(vertices_to_remove)

            print("Mesh créé avec Poisson reconstruction")

        except Exception as e:
            print(f"Poisson reconstruction a échoué: {e}")
            # Fallback: juste afficher le nuage de points
            mesh = global_map
            print("Affichage du nuage de points uniquement")

    # Visualisation
    if mesh is not None and (isinstance(mesh, o3d.geometry.TriangleMesh) or len(mesh.points) > 0):
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        if isinstance(mesh, o3d.geometry.TriangleMesh):
            mesh.compute_vertex_normals()
            vis.add_geometry(mesh)
            opt = vis.get_render_option()
            opt.mesh_show_back_face = True
            opt.mesh_show_wireframe = False
        else:
            vis.add_geometry(mesh)
            opt = vis.get_render_option()
            opt.point_size = 2.0

        opt.light_on = True
        opt.background_color = np.array([0.1, 0.1, 0.1])

        vis.run()
        vis.destroy_window()
    else:
        print("Mesh vide")



# =========================
# ENTRYPOINT
# =========================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)

    parser.add_argument("--topic_front", default="/ez10_gen1/hesai_front/cloud")
    parser.add_argument("--topic_rear", default="/ez10_gen1/hesai_rear/cloud")

    args = parser.parse_args()

    main(args.bag, args.topic_front, args.topic_rear)