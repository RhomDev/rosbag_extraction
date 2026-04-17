import numpy as np
import argparse
import shutil
from pathlib import Path
import datetime
from bisect import bisect_left

from tqdm import tqdm
import open3d as o3d
from scipy.spatial.transform import Rotation as Rot

import rosbag2_py
from sensor_msgs_py import point_cloud2
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

import extracte_setup

# ==================
#   CONFIGURATION
# ==================

ROOT = Path("../../out/save_mesh")

# Paramètres de filtrage spatial (à adapter à votre scène)
Z_MIN, Z_MAX     = -2.0, 5.0      # Hauteur min/max (élimine sol bruité + toit trop haut)
DIST_MIN         = 0.5             # Rayon mort autour du capteur (bruit proche)
DIST_MAX         = 50.0            # Distance max utile (évite points trop lointains)
VOXEL_FRAME      = 0.50            # Voxel par frame (↑ = moins de points, plus rapide)
VOXEL_FINAL      = 0.05            # Voxel final avant mesh
VOXEL_MESH       = 0.15            # Voxel dédié mesh (> VOXEL_FINAL pour réduire la RAM)
MAX_PTS_BUFFER   = 5_000_000       # Seuil downsampling du buffer global
POISSON_DEPTH    = 9               # Profondeur Poisson (8 = plus rapide, moins détaillé)

# ==================
#   LECTURE POINTS (RAPIDE)
# ==================

def read_lidar_points_fast(msg) -> np.ndarray | None:
    """
    Lecture vectorisée via read_points_numpy (évite fromiter + vstack).
    Retourne un tableau (N, 3) float32.
    """
    try:
        pts = point_cloud2.read_points_numpy(
            msg, field_names=("x", "y", "z"), skip_nans=True
        )  # shape (N, 3) ou (N,) structuré selon la version
        if pts.dtype.names:                        # dtype structuré
            pts = np.column_stack([pts["x"], pts["y"], pts["z"]])
        pts = pts.astype(np.float32)
        return pts if pts.shape[0] > 0 else None
    except Exception:
        # Fallback si read_points_numpy indisponible
        gen = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        arr = np.array(list(gen), dtype=np.float32)
        return arr if arr.shape[0] > 0 else None

# ==================
#   FILTRAGE SPATIAL
# ==================

def spatial_filter(pts: np.ndarray) -> np.ndarray:
    """
    Filtre combiné distance + hauteur.
    Beaucoup plus rapide qu'un filtre en deux passes séparées.
    """
    dist2 = pts[:, 0] ** 2 + pts[:, 1] ** 2 + pts[:, 2] ** 2
    mask = (
        (dist2 > DIST_MIN ** 2) &
        (dist2 < DIST_MAX ** 2) &
        (pts[:, 2] > Z_MIN) &
        (pts[:, 2] < Z_MAX)
    )
    return pts[mask]

def filter_pointcloud(pc, angle_min_deg=-180, angle_max_deg=180,
                      min_dist=0.0, max_dist=10.0):
    if pc is None or pc.shape[0] == 0:
        return None

    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]

    angles = np.degrees(np.arctan2(y, x))
    distances = np.sqrt(x**2 + y**2)

    mask = (
        (angles >= angle_min_deg) & (angles <= angle_max_deg) &
        (distances >= min_dist) & (distances <= max_dist)
    )

    if not np.any(mask):
        return None

    return pc[mask].astype(np.float32)

# ==================
#   GNSS & TRANSFORMATIONS
# ==================

R_EARTH = 6378137.0

def gnss_to_local(lat, lon, alt, lat0, lon0, alt0):
    lat_r, lon_r   = np.radians(lat), np.radians(lon)
    lat0_r, lon0_r = np.radians(lat0), np.radians(lon0)
    x = (lon_r - lon0_r) * R_EARTH * np.cos(lat0_r)
    y = (lat_r - lat0_r) * R_EARTH
    z = alt - alt0
    return x, y, z

def gnss_pose_to_transform(gnss_msg, lat0, lon0, alt0) -> np.ndarray:
    lon, lat, alt = (gnss_msg.pose.pose.position.x,
                     gnss_msg.pose.pose.position.y,
                     gnss_msg.pose.pose.position.z)
    tx, ty, tz = gnss_to_local(lat, lon, alt, lat0, lon0, alt0)

    q = gnss_msg.pose.pose.orientation
    R = Rot.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3]  = [tx, ty, tz]
    return T

def transform_points(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Transformation vectorisée (évite hstack/T inutile)."""
    return (pts.astype(np.float64) @ T[:3, :3].T) + T[:3, 3]

# ==================
#   SYNCHRONISATION TEMPORELLE
# ==================

def build_timestamp_index(msgs_with_ts: list[tuple]) -> np.ndarray:
    """Construit un tableau de timestamps numpy pour bisect."""
    return np.array([ts for _, ts in msgs_with_ts], dtype=np.int64)

def closest_gnss(gnss_timestamps: np.ndarray, lidar_ts: int) -> int:
    """Retourne l'index GNSS le plus proche en temps (O(log n))."""
    idx = bisect_left(gnss_timestamps, lidar_ts)
    if idx == 0:
        return 0
    if idx >= len(gnss_timestamps):
        return len(gnss_timestamps) - 1
    before = gnss_timestamps[idx - 1]
    after  = gnss_timestamps[idx]
    return idx - 1 if abs(lidar_ts - before) <= abs(lidar_ts - after) else idx

# ==================
#   EXTRACTION ROSBAG (avec timestamps)
# ==================

def extract_messages_with_ts(reader, topic_names: dict, type_map: dict):
    """
    Extrait les messages avec leurs timestamps.
    Retourne des listes de tuples (msg, timestamp_ns).
    """
    front_msgs, rear_msgs, gnss_msgs = [], [], []

    front_type = get_message(type_map[topic_names['front']])
    rear_type  = get_message(type_map[topic_names['rear']])
    gnss_type  = get_message(type_map[topic_names['gnss']])

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic == topic_names['front']:
            front_msgs.append((deserialize_message(data, front_type), t))
        elif topic == topic_names['rear']:
            rear_msgs.append((deserialize_message(data, rear_type), t))
        elif topic == topic_names['gnss']:
            gnss_msgs.append((deserialize_message(data, gnss_type), t))

    return front_msgs, rear_msgs, gnss_msgs

def get_gnss_reference(gnss_msgs_with_ts):
    for msg, _ in gnss_msgs_with_ts:
        lat = msg.pose.pose.position.y
        lon = msg.pose.pose.position.x
        alt = msg.pose.pose.position.z
        if not any(np.isnan(v) for v in (lat, lon, alt)):
            return lat, lon, alt
    raise ValueError("Aucun message GNSS valide trouvé pour la référence")

# ==================
#   ACCUMULATION (OPTIMISÉE)
# ==================

def accumulate_clouds_optimized(
    lidar_msgs_ts: list[tuple],
    gnss_msgs_ts:  list[tuple],
    lat0: float, lon0: float, alt0: float,
    voxel_frame: float = VOXEL_FRAME,
    desc: str = "LiDAR"
) -> o3d.geometry.PointCloud:
    """
    Accumule les nuages LiDAR avec synchronisation temporelle précise.
    Filtrage spatial + downsampling par frame pour minimiser la mémoire.
    """
    gnss_timestamps = np.array([ts for _, ts in gnss_msgs_ts], dtype=np.int64)
    gnss_list       = [msg for msg, _ in gnss_msgs_ts]

    # Buffer de points (concat numpy puis conversion O3D une seule fois)
    all_pts_chunks: list[np.ndarray] = []
    total_pts = 0

    combined = o3d.geometry.PointCloud()

    for lidar_msg, lidar_ts in tqdm(lidar_msgs_ts, desc=desc):
        # 1. Lecture vectorisée
        pts = read_lidar_points_fast(lidar_msg)
        if pts is None:
            continue
        pts = filter_pointcloud(pts, min_dist=1.0, max_dist=20.0)
        if pts is None:
            continue

        # 2. Filtrage spatial AVANT transformation (repère capteur → moins de calculs)
        pts = spatial_filter(pts)
        if pts.shape[0] == 0:
            continue

        # 3. Pose GNSS la plus proche temporellement
        gnss_idx = closest_gnss(gnss_timestamps, lidar_ts)
        T = gnss_pose_to_transform(gnss_list[gnss_idx], lat0, lon0, alt0)

        # 4. Transformation
        pts_world = transform_points(pts, T)

        # 5. Downsampling rapide par frame via Open3D
        pcd_frame = o3d.geometry.PointCloud()
        pcd_frame.points = o3d.utility.Vector3dVector(pts_world)
        pcd_frame = pcd_frame.voxel_down_sample(voxel_frame)

        combined += pcd_frame
        total_pts += len(pcd_frame.points)

        # 6. Contrôle mémoire proactif
        if total_pts > MAX_PTS_BUFFER:
            combined = combined.voxel_down_sample(voxel_frame)
            total_pts = len(combined.points)

    return combined

# ==================
#   RECONSTRUCTION MAILLAGE (OPTIMISÉE)
# ==================

def reconstruct_mesh(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = VOXEL_FRAME,
    poisson_depth: int = POISSON_DEPTH
) -> o3d.geometry.TriangleMesh:
    """
    Pipeline de reconstruction avec nettoyage statistique avant Poisson.
    """
    print(f"  Downsampling final ({len(pcd.points)} pts → voxel {voxel_size} m)...")
    pcd_down = pcd.voxel_down_sample(voxel_size)

    print(f"  Suppression outliers statistiques ({len(pcd_down.points)} pts)...")
    pcd_down, _ = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    print(f"  Estimation des normales ({len(pcd_down.points)} pts)...")
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 3, max_nn=30
        )
    )
    pcd_down.orient_normals_consistent_tangent_plane(k=15)

    print(f"  Reconstruction Poisson (depth={poisson_depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_down, depth=poisson_depth, n_threads=-1   # multi-thread
    )

    densities = np.asarray(densities)
    thresh = np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(densities < thresh)
    mesh = mesh.filter_smooth_simple(number_of_iterations=3)
    mesh.compute_vertex_normals()

    return mesh

# ==================
#   MAIN
# ==================

def main(bag_path: str, topic_front: str, topic_rear: str, topic_gnss: str):
    try:
        ROOT.mkdir(parents=True, exist_ok=True)
        print("LANCEMENT DU PROGRAMME")

        # 1. Ouverture
        reader,type_name, type_map = extracte_setup.configuration(bag_path)
        topic_names = {'front': topic_front, 'rear': topic_rear, 'gnss': topic_gnss}

        # 2. Extraction avec timestamps
        print("Extraction des messages...")
        msgs_front, msgs_rear, msgs_gnss = extract_messages_with_ts(
            reader, topic_names, type_map
        )
        print(f"  {len(msgs_front)} frames avant | {len(msgs_rear)} frames arrière | "
              f"{len(msgs_gnss)} poses GNSS")

        if not msgs_gnss:
            print("Erreur : aucun message GNSS."); return
        if not msgs_front and not msgs_rear:
            print("Erreur : aucun message LiDAR."); return

        # 3. Référence GNSS
        lat0, lon0, alt0 = get_gnss_reference(msgs_gnss)
        print(f"Référence GNSS : lat={lat0:.6f}, lon={lon0:.6f}, alt={alt0:.2f} m")

        # 4. Accumulation (sync temporelle)
        cloud_front = accumulate_clouds_optimized(
            msgs_front, msgs_gnss, lat0, lon0, alt0, desc="LiDAR avant"
        )
        print(f"  → {len(cloud_front.points)} points (avant)")

        cloud_rear = accumulate_clouds_optimized(
            msgs_rear, msgs_gnss, lat0, lon0, alt0, desc="LiDAR arrière"
        )
        print(f"  → {len(cloud_rear.points)} points (arrière)")

        # 5. Fusion + downsampling final
        combined = cloud_front + cloud_rear
        print(f"Fusion : {len(combined.points)} points → downsampling final...")
        combined = combined.voxel_down_sample(VOXEL_FRAME)
        print(f"Après downsampling final : {len(combined.points)} points")

        # 6. Maillage
        print("Reconstruction du maillage...")
        mesh = reconstruct_mesh(combined)
        print(f"Maillage : {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

        # 7. Sauvegarde
        now = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        out_path = ROOT / f"output_mesh_{now}.ply"
        o3d.io.write_triangle_mesh(str(out_path), mesh)
        print(f"Maillage sauvegardé : {out_path}")

    except Exception as e:
        import traceback
        print(f"\n--- ERREUR ---\n{type(e).__name__}: {e}\n")
        traceback.print_exc()
    finally:
        input("\nAppuyez sur Entrée pour fermer...")

# ==================
#   ENTRYPOINT
# ==================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Génération de maillage 3D LiDAR+GNSS depuis rosbag2 (version optimisée)"
    )
    parser.add_argument("--bag",          required=True)
    parser.add_argument("--topic_front",  default="/ez10_gen1/hesai_front/cloud")
    parser.add_argument("--topic_rear",   default="/ez10_gen1/hesai_rear/cloud")
    parser.add_argument("--topic_gnss",   default="/ez10_gen1/gnss_main/pose_lpz")
    args = parser.parse_args()
    main(args.bag, args.topic_front, args.topic_rear, args.topic_gnss)