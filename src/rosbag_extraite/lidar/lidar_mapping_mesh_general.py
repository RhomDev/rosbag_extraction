"""
Pipeline de mapping LiDAR → Mesh 3D depuis un ROS2 Bag
=======================================================
Données attendues dans le bag :
  - Topics LiDAR   : sensor_msgs/msg/PointCloud2  (avant + arrière)
  - Topic GNSS     : geometry_msgs/msg/PoseWithCovarianceStamped
  - Topic FWS      : four_wheel_steering_msgs/msg/FourWheelSteeringStamped
  - (optionnel)    : sensor_msgs/msg/Imu (orientation)

Sortie : fichier .ply (nuage) + .obj / .ply (mesh Poisson)

Dépendances :
    pip install rosbag2-py open3d pyproj numpy scipy
    (rclpy fourni par la distribution ROS2 installée)
"""

import sys
import os
import math
import datetime
import numpy as np
from pathlib import Path
from typing import Optional


def quat_to_euler(x, y, z, w):
    roll  = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - z * x))
    yaw   = math.atan2(2 * w * z + x * y, 1 - 2 * (y * y + z * z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BAG_PATH = "~/Documents/stage/bag_files/ez10_gen1_sensors_2026-04-02-16-45-39/"

TOPIC_LIDAR_FRONT = "/ez10_gen1/hesai_front/cloud"
TOPIC_LIDAR_REAR  = "/ez10_gen1/hesai_rear/cloud"
TOPIC_GNSS        = "/ez10_gen1/gnss_main/pose_lpz"
TOPIC_FWS         = "/ez10_gen1/received_raw_four_wheel_steering"  # ← nouveau

OUTPUT_DIR    = Path("../../../out/save_mesh")
MIN_SPEED_MS  = 0.2      # seuil de mouvement (m/s)
VOXEL_SIZE    = 0.05
POISSON_DEPTH = 10
NORMAL_RADIUS = 0.2

SCAN_WHEN_MOVING = False  # True = scan en mouvement, False = scan à l'arrêt

# Calibration extrinsèque LiDAR → base véhicule
roll_front, pitch_front, yaw_front = quat_to_euler(0.095, 0.099, 0.724, 0.676)
EXTRINSIC_FRONT = [1.460,  0.010, 1.920, roll_front, pitch_front, yaw_front]

roll_rear, pitch_rear, yaw_rear = quat_to_euler(0.099, -0.080, -0.692, 0.711)
EXTRINSIC_REAR  = [-1.460, -0.010, 1.910, roll_rear, pitch_rear, yaw_rear]

# ─────────────────────────────────────────────────────────────────────────────


def deg2rad(d): return d * np.pi / 180.0


def extrinsic_to_matrix(params):
    """[x, y, z, roll°, pitch°, yaw°] → matrice homogène 4×4."""
    x, y, z, ro, pi, ya = params
    ro, pi, ya = deg2rad(ro), deg2rad(pi), deg2rad(ya)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(ro), -np.sin(ro)],
                   [0, np.sin(ro),  np.cos(ro)]])
    Ry = np.array([[ np.cos(pi), 0, np.sin(pi)],
                   [0,           1, 0],
                   [-np.sin(pi), 0, np.cos(pi)]])
    Rz = np.array([[np.cos(ya), -np.sin(ya), 0],
                   [np.sin(ya),  np.cos(ya), 0],
                   [0,           0,          1]])
    T = np.eye(4)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3]  = [x, y, z]
    return T


# ─────────────────────────────────────────────────────────────────────────────
# 1. LECTURE DU BAG ROS2
# ─────────────────────────────────────────────────────────────────────────────

def read_bag(bag_path: str):
    """
    Lit le bag et retourne :
        gnss_msgs   : [(ts_ns, PoseWithCovarianceStamped), ...]
        lidar_front : [(ts_ns, PointCloud2), ...]
        lidar_rear  : [(ts_ns, PointCloud2), ...]
        fws_msgs    : [(ts_ns, FourWheelSteeringStamped), ...]
    """
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as e:
        sys.exit(f"[ERREUR] Modules ROS2 introuvables : {e}")

    bag_path = os.path.expanduser(bag_path)
    storage_id = "mcap" if any(f.endswith(".mcap")
                               for f in os.listdir(bag_path)) else "sqlite3"

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr"),
    )

    topic_types = {m.name: m.type for m in reader.get_all_topics_and_types()}

    def make_deserializer(topic):
        t = topic_types.get(topic)
        if t is None:
            print(f"[WARN] Topic '{topic}' absent du bag.")
        return get_message(t) if t else None

    deser_gnss  = make_deserializer(TOPIC_GNSS)
    deser_front = make_deserializer(TOPIC_LIDAR_FRONT)
    deser_rear  = make_deserializer(TOPIC_LIDAR_REAR)
    deser_fws   = make_deserializer(TOPIC_FWS)

    reader.set_filter(rosbag2_py.StorageFilter(
        topics=[TOPIC_GNSS, TOPIC_LIDAR_FRONT, TOPIC_LIDAR_REAR, TOPIC_FWS]))

    gnss_msgs, lidar_front, lidar_rear, fws_msgs = [], [], [], []

    print("[INFO] Lecture du bag en cours…")
    while reader.has_next():
        topic, data, ts = reader.read_next()
        if topic == TOPIC_GNSS and deser_gnss:
            gnss_msgs.append((ts, deserialize_message(data, deser_gnss)))
        elif topic == TOPIC_LIDAR_FRONT and deser_front:
            lidar_front.append((ts, deserialize_message(data, deser_front)))
        elif topic == TOPIC_LIDAR_REAR and deser_rear:
            lidar_rear.append((ts, deserialize_message(data, deser_rear)))
        elif topic == TOPIC_FWS and deser_fws:
            fws_msgs.append((ts, deserialize_message(data, deser_fws)))

    print(f"[INFO] GNSS={len(gnss_msgs)}, "
          f"LiDAR avant={len(lidar_front)}, LiDAR arrière={len(lidar_rear)}, "
          f"FWS={len(fws_msgs)}")
    return gnss_msgs, lidar_front, lidar_rear, fws_msgs


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRAJECTOIRE GNSS
# ─────────────────────────────────────────────────────────────────────────────

class GNSSTrajectory:
    """Position + orientation interpolées depuis PoseWithCovarianceStamped."""

    def __init__(self, gnss_msgs):
        if not gnss_msgs:
            sys.exit("[ERREUR] Aucun message GNSS dans le bag.")

        self.timestamps, self.positions, self.quaternions = [], [], []
        self._gnss_speeds = []

        for ts, msg in gnss_msgs:
            try:
                p = msg.pose.pose.position
                self.quaternions.append(msg.pose.pose.orientation)
            except AttributeError:
                sys.exit("[ERREUR] Topic GNSS attendu : PoseWithCovarianceStamped")
            self.timestamps.append(ts * 1e-9)
            self.positions.append((p.x, p.y, p.z))

        self._compute_gnss_speeds()
        print(f"[INFO] Trajectoire GNSS : {len(self.positions)} points.")

    def _compute_gnss_speeds(self):
        ts  = np.array(self.timestamps)
        pos = np.array(self.positions)
        self._gnss_speeds = [0.0]
        for i in range(1, len(ts)):
            dt = ts[i] - ts[i - 1]
            self._gnss_speeds.append(
                np.linalg.norm(pos[i] - pos[i - 1]) / dt if dt > 1e-6
                else self._gnss_speeds[-1])

    def interpolate_pose(self, query_ts_ns: int):
        """Retourne (position_xyz, yaw_rad) interpolés, ou (None, None)."""
        t  = query_ts_ns * 1e-9
        ts = self.timestamps

        if t < ts[0] or t > ts[-1]:
            return None, None

        idx = np.searchsorted(ts, t)
        if idx == 0:
            return np.array(self.positions[0]), self._quat_to_yaw(self.quaternions[0])
        if idx >= len(ts):
            return np.array(self.positions[-1]), self._quat_to_yaw(self.quaternions[-1])

        alpha = (t - ts[idx - 1]) / (ts[idx] - ts[idx - 1] + 1e-12)
        p0, p1 = np.array(self.positions[idx - 1]), np.array(self.positions[idx])
        q = self.quaternions[idx - 1] if alpha < 0.5 else self.quaternions[idx]
        return p0 + alpha * (p1 - p0), self._quat_to_yaw(q)

    def gnss_speed_at(self, query_ts_ns: int) -> float:
        t   = query_ts_ns * 1e-9
        idx = min(np.searchsorted(self.timestamps, t), len(self._gnss_speeds) - 1)
        return self._gnss_speeds[idx]

    @staticmethod
    def _quat_to_yaw(q) -> float:
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))


# ─────────────────────────────────────────────────────────────────────────────
# 3. VITESSE PAR ODOMÈTRE ROUE (FourWheelSteering)
# ─────────────────────────────────────────────────────────────────────────────

class FWSSpeedTracker:
    """
    Extrait la vitesse longitudinale depuis FourWheelSteeringStamped et la
    fusionne avec la vitesse GNSS.

    Pourquoi le FWS est meilleur que ΔPos_GNSS/Δt :
      - Fréquence bien plus élevée (typiquement 50–100 Hz vs 5–10 Hz GNSS)
      - Pas de latence de fix satellite
      - Résolution sous-centimétrique par codeur de roue
      - Insensible aux sauts de position GPS

    Stratégie de fusion :
      1. FWS disponible dans la fenêtre → priorité FWS (interpolé)
      2. Hors plage FWS              → fallback GNSS
      3. Contrôle de cohérence       → avertissement si |v_fws - v_gnss| > seuil
    """

    MAX_DISCREPANCY_MS = 3.0   # m/s — seuil d'alerte incohérence

    def __init__(self, fws_msgs, gnss_traj: GNSSTrajectory):
        self._gnss    = gnss_traj
        self._has_fws = len(fws_msgs) > 0

        if not self._has_fws:
            print("[WARN] Aucun message FWS — vitesse GNSS seule utilisée.")
            return

        self._ts     = np.empty(len(fws_msgs), dtype=np.float64)
        self._speeds = np.empty(len(fws_msgs), dtype=np.float64)

        for i, (ts, msg) in enumerate(fws_msgs):
            self._ts[i] = ts * 1e-9
            # FourWheelSteering.speed est signé (< 0 en marche arrière)
            self._speeds[i] = abs(msg.data.speed)

        self._check_consistency(fws_msgs)

        print(f"[INFO] FWSSpeedTracker : {len(fws_msgs)} mesures | "
              f"v_max={self._speeds.max():.2f} m/s | "
              f"v_moy={self._speeds.mean():.2f} m/s")

    # ── API publique ──────────────────────────────────────────────────────────

    def speed_at(self, query_ts_ns: int) -> float:
        """Meilleure estimation de vitesse (m/s) — FWS prioritaire, GNSS fallback."""
        if not self._has_fws:
            return self._gnss.gnss_speed_at(query_ts_ns)

        t = query_ts_ns * 1e-9
        if t < self._ts[0] or t > self._ts[-1]:
            return self._gnss.gnss_speed_at(query_ts_ns)

        return float(self._interp(t))

    def is_moving(self, query_ts_ns: int) -> bool:
        return self.speed_at(query_ts_ns) >= MIN_SPEED_MS

    # ── Interne ───────────────────────────────────────────────────────────────

    def _interp(self, t_s: float) -> float:
        idx = np.searchsorted(self._ts, t_s)
        if idx == 0:            return self._speeds[0]
        if idx >= len(self._ts): return self._speeds[-1]
        a = (t_s - self._ts[idx-1]) / (self._ts[idx] - self._ts[idx-1] + 1e-12)
        return self._speeds[idx-1] + a * (self._speeds[idx] - self._speeds[idx-1])

    def _check_consistency(self, fws_msgs):
        step = max(1, len(fws_msgs) // 200)
        bad  = 0
        for i in range(0, len(fws_msgs), step):
            ts = fws_msgs[i][0]
            if abs(self._speeds[i] - self._gnss.gnss_speed_at(ts)) > self.MAX_DISCREPANCY_MS:
                bad += 1
        n_checked = len(range(0, len(fws_msgs), step))
        pct = 100 * bad / max(n_checked, 1)
        if bad:
            print(f"[WARN] Cohérence FWS↔GNSS : {bad}/{n_checked} "
                  f"échantillons hors tolérance ({pct:.1f}%)")
            if pct > 20:
                print("[WARN] Incohérence élevée — vérifier synchronisation "
                      "temporelle ou glissement de roue.")
        else:
            print("[INFO] Cohérence FWS↔GNSS : OK")


# ─────────────────────────────────────────────────────────────────────────────
# 4. DÉSÉRIALISATION PointCloud2 → numpy XYZ
# ─────────────────────────────────────────────────────────────────────────────

def fast_pointcloud2_to_xyz(msg) -> Optional[np.ndarray]:
    """Extraction vectorisée des champs XYZ (float32 standard Hesai/Velodyne)."""
    off = {f.name: f.offset for f in msg.fields}
    if not all(k in off for k in ("x", "y", "z")):
        return _slow_pointcloud2_to_xyz(msg)

    n    = msg.height * msg.width
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    rows = data[:n * msg.point_step].reshape(n, msg.point_step)

    xyz = np.column_stack([
        rows[:, off["x"]:off["x"] + 4].view(np.float32).ravel(),
        rows[:, off["y"]:off["y"] + 4].view(np.float32).ravel(),
        rows[:, off["z"]:off["z"] + 4].view(np.float32).ravel(),
    ])
    return xyz[np.isfinite(xyz).all(axis=1)]


def _slow_pointcloud2_to_xyz(msg) -> Optional[np.ndarray]:
    import struct
    fmt_map = {1:"b",2:"B",3:"h",4:"H",5:"i",6:"I",7:"f",8:"d"}
    fields  = {f.name: (f.offset, fmt_map.get(f.datatype, "f")) for f in msg.fields}
    if not all(k in fields for k in ("x","y","z")):
        return None
    xo,xf = fields["x"]; yo,yf = fields["y"]; zo,zf = fields["z"]
    endian = ">" if msg.is_bigendian else "<"
    data   = bytes(msg.data)
    n      = msg.height * msg.width
    xyz    = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        base = (i // msg.width) * msg.row_step + (i % msg.width) * msg.point_step
        xyz[i] = (struct.unpack_from(endian+xf, data, base+xo)[0],
                  struct.unpack_from(endian+yf, data, base+yo)[0],
                  struct.unpack_from(endian+zf, data, base+zo)[0])
    return xyz[np.isfinite(xyz).all(axis=1)]


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRANSFORMATION capteur → repère global
# ─────────────────────────────────────────────────────────────────────────────

def transform_points(xyz: np.ndarray, pose_xyz,
                     yaw_rad: float, extrinsic: np.ndarray) -> np.ndarray:
    ones    = np.ones((len(xyz), 1))
    xyz_veh = (extrinsic @ np.hstack([xyz, ones]).T).T[:, :3]
    c, s    = np.cos(yaw_rad), np.sin(yaw_rad)
    R       = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)
    return ((R @ xyz_veh.T).T + pose_xyz).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 6. ACCUMULATION DU NUAGE GLOBAL
# ─────────────────────────────────────────────────────────────────────────────

def should_keep_scan(is_moving: bool) -> bool:
    return is_moving if SCAN_WHEN_MOVING else not is_moving

def accumulate_pointcloud(gnss_traj: GNSSTrajectory,
                          speed_tracker: FWSSpeedTracker,
                          lidar_msgs_list,
                          extrinsic: np.ndarray,
                          label: str) -> np.ndarray:
    all_pts, skipped_stop, skipped_range = [], 0, 0

    for ts, msg in lidar_msgs_list:

        # Filtre mouvement — FWS prioritaire, GNSS fallback
        if not should_keep_scan(speed_tracker.is_moving(ts)):
            skipped_stop += 1
            continue

        # Pose GNSS interpolée
        pose, yaw = gnss_traj.interpolate_pose(ts)
        if pose is None:
            skipped_range += 1
            continue

        xyz = fast_pointcloud2_to_xyz(msg)
        if xyz is None or len(xyz) == 0:
            continue

        all_pts.append(transform_points(xyz, pose, yaw, extrinsic))

    print(f"[INFO] {label}: {len(all_pts)} scans utilisés | "
          f"{skipped_stop} ignorés (arrêt) | "
          f"{skipped_range} ignorés (hors plage GNSS)")

    return np.vstack(all_pts) if all_pts else np.zeros((0, 3), dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 7. CONSTRUCTION DU MESH (Open3D)
# ─────────────────────────────────────────────────────────────────────────────

def build_mesh(combined_pts: np.ndarray, output_dir: Path) -> None:
    try:
        import open3d as o3d
    except ImportError:
        sys.exit("[ERREUR] open3d non installé : pip install open3d")

    output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

    print(f"\n[INFO] Nuage global brut : {len(combined_pts):,} points")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_pts)

    pcd_ds = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    print(f"[INFO] Après downsampling ({VOXEL_SIZE}m) : {len(pcd_ds.points):,} points")

    pcd_clean, _ = pcd_ds.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"[INFO] Après filtre outliers : {len(pcd_clean.points):,} points")

    cloud_path = output_dir / f"output_cloud_{now}.ply"
    o3d.io.write_point_cloud(str(cloud_path), pcd_clean)
    print(f"[OK]   Nuage enregistré → {cloud_path}")

    print("[INFO] Calcul des normales…")
    pcd_clean.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_RADIUS, max_nn=30))
    pcd_clean.orient_normals_towards_camera_location(
        camera_location=np.array([0., 0., 1000.]))

    print(f"[INFO] Reconstruction Poisson (depth={POISSON_DEPTH})…")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_clean, depth=POISSON_DEPTH, width=0, scale=1.1, linear_fit=False)

    dens = np.asarray(densities)
    mesh.remove_vertices_by_mask(dens < np.percentile(dens, 10))
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh = mesh.filter_smooth_simple(number_of_iterations=3)
    mesh.compute_vertex_normals()

    print(f"[INFO] Mesh : {len(mesh.vertices):,} sommets, "
          f"{len(mesh.triangles):,} triangles")

    for ext in (".ply", ".obj"):
        p = output_dir / f"map_mesh_{now}{ext}"
        o3d.io.write_triangle_mesh(str(p), mesh)
        print(f"[OK]   Mesh enregistré → {p}")

    _visualize(pcd_clean, mesh)


def _visualize(pcd, mesh):
    try:
        import open3d as o3d
        if os.environ.get("DISPLAY") or sys.platform in ("darwin", "win32"):
            pcd.paint_uniform_color([0.4, 0.4, 0.4])
            mesh.paint_uniform_color([0.9, 0.9, 0.9])
            o3d.visualization.draw_geometries(
                [pcd, mesh], window_name="Carte LiDAR 3D",
                width=1280, height=720)
    except Exception as e:
        print(f"[WARN] Visualisation impossible : {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    bag = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else BAG_PATH)

    print("=" * 62)
    print("  Pipeline LiDAR → Mesh 3D  |  ROS2 Bag")
    print("=" * 62)
    print(f"  Bag          : {bag}")
    print(f"  LiDAR avant  : {TOPIC_LIDAR_FRONT}")
    print(f"  LiDAR arrière: {TOPIC_LIDAR_REAR}")
    print(f"  GNSS         : {TOPIC_GNSS}")
    print(f"  FWS          : {TOPIC_FWS}")
    print(f"  Sortie       : {OUTPUT_DIR}")
    print("=" * 62)

    # 1. Lecture
    gnss_msgs, lidar_front, lidar_rear, fws_msgs = read_bag(bag)

    # 2. Trajectoire GNSS
    traj = GNSSTrajectory(gnss_msgs)

    # 3. Tracker de vitesse fusionné FWS + GNSS
    speed_tracker = FWSSpeedTracker(fws_msgs, traj)

    # 4. Matrices extrinsèques
    T_front = extrinsic_to_matrix(EXTRINSIC_FRONT)
    T_rear  = extrinsic_to_matrix(EXTRINSIC_REAR)

    # 5. Accumulation
    pts_front = accumulate_pointcloud(traj, speed_tracker,
                                      lidar_front, T_front, "LiDAR avant")
    pts_rear  = accumulate_pointcloud(traj, speed_tracker,
                                      lidar_rear,  T_rear,  "LiDAR arrière")

    # 6. Fusion
    parts = [p for p in (pts_front, pts_rear) if len(p) > 0]
    if not parts:
        sys.exit("[ERREUR] Aucun point accumulé. Vérifiez les noms de topics.")
    combined = np.vstack(parts)
    print(f"\n[INFO] Nuage fusionné total : {len(combined):,} points")

    # 7. Mesh
    build_mesh(combined, OUTPUT_DIR)
    print("\n[DONE] Pipeline terminé avec succès.")


if __name__ == "__main__":
    main()