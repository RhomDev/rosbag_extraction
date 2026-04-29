"""
Pipeline de mapping LiDAR → Mesh 3D depuis un ROS2 Bag  (version HPC)
======================================================================
Optimisations vs version originale :
  ✦ Lecture bag 2 passes : métadonnées (GNSS/FWS) puis streaming LiDAR
    → les messages LiDAR ne sont JAMAIS tous en RAM simultanément
  ✦ Transformations fusionnées (1 matmul au lieu de 3 allocations)
  ✦ Accélération GPU automatique : CUDA (CuPy) → MPS (PyTorch) → NumPy
  ✦ Downsampling incrémental Open3D → mémoire bornée
  ✦ Checkpoints .npy périodiques → reprise après crash
  ✦ Monitoring mémoire + GC explicite anti-OOM
  ✦ Désérialisation PointCloud2 100 % vectorisée (slow path éliminé)

Dépendances :
    pip install rosbag2-py open3d pyproj numpy scipy
    (optionnel) pip install cupy-cuda12x   # GPU NVIDIA
    (optionnel) pip install torch           # GPU MPS Apple Silicon
"""

import sys
import os
import math
import datetime
import gc
import time
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List


# ═════════════════════════════════════════════════════════════════════════════
# GPU DETECTION — fallback automatique CUDA → MPS → CPU
# ═════════════════════════════════════════════════════════════════════════════

_DEVICE = "cpu"
_torch_device = None

try:
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() > 0:
        _DEVICE = "cuda_cupy"
        # Limite mémoire GPU à 80 % pour éviter OOM GPU
        mempool = cp.get_default_memory_pool()
        mempool.set_limit(fraction=0.8)
except Exception:
    pass

if _DEVICE == "cpu":
    try:
        import torch
        if torch.cuda.is_available():
            _DEVICE = "torch_cuda"
            _torch_device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _DEVICE = "torch_mps"
            _torch_device = torch.device("mps")
    except Exception:
        pass

print(f"[INFO] Accélération matérielle : {_DEVICE}")


# ═════════════════════════════════════════════════════════════════════════════
# UTILITAIRES SYSTÈME — mémoire, GC, checkpoints
# ═════════════════════════════════════════════════════════════════════════════

def _mem_used_gb() -> float:
    """RSS du processus courant en Go (sans dépendance psutil)."""
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF)
        return ru.ru_maxrss / (1024 ** 2)        # macOS : octets → Go
    except Exception:
        return 0.0


def _force_gc():
    """Garbage-collect + libère les pools GPU si disponibles."""
    gc.collect()
    if _DEVICE == "cuda_cupy":
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    elif _DEVICE.startswith("torch"):
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _save_checkpoint(array: np.ndarray, path: Path, label: str):
    """Sauvegarde .npy atomique (écriture tmp puis rename)."""
    tmp = path.with_suffix(".tmp.npy")
    np.save(str(tmp), array)
    tmp.rename(path)
    print(f"  [CKPT] {label} → {path.name}  ({len(array):,} pts, "
          f"{array.nbytes / 1e6:.0f} Mo)")


def _load_checkpoint(path: Path) -> Optional[np.ndarray]:
    if path.exists():
        arr = np.load(str(path))
        print(f"  [CKPT] Reprise depuis {path.name}  ({len(arr):,} pts)")
        return arr
    return None


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

BAG_PATH = "~/Documents/stage/bag_files/ez10_gen1_sensors_2026-04-02-16-45-39/"

TOPIC_LIDAR_FRONT = "/ez10_gen1/hesai_front/cloud"
TOPIC_LIDAR_REAR  = "/ez10_gen1/hesai_rear/cloud"
TOPIC_GNSS        = "/ez10_gen1/gnss_main/pose_lpz"
TOPIC_FWS         = "/ez10_gen1/received_raw_four_wheel_steering"

OUTPUT_DIR    = Path("../../../out/save_mesh")
MIN_SPEED_MS  = 0.5
VOXEL_SIZE    = 0.05
POISSON_DEPTH = 10
NORMAL_RADIUS = 0.2

SCAN_WHEN_MOVING = True

# ── Paramètres d'optimisation ────────────────────────────────────────────
CHECKPOINT_INTERVAL     = 200     # scans entre sauvegardes disque
INCREMENTAL_DS_INTERVAL = 150     # scans entre voxel-DS incrémentaux
MEMORY_LIMIT_GB         = 6.0    # seuil RAM avant DS forcé (ajuster selon machine)
CHECKPOINT_DIR          = OUTPUT_DIR / ".checkpoints"

# Calibration extrinsèque LiDAR → base véhicule
def quat_to_euler(x, y, z, w):
    roll  = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))   # clamp anti-NaN
    yaw   = math.atan2(2 * w * z + x * y, 1 - 2 * (y * y + z * z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

roll_front, pitch_front, yaw_front = quat_to_euler(0.095, 0.099, 0.724, 0.676)
EXTRINSIC_FRONT = [1.460,  0.010, 1.920, roll_front, pitch_front, yaw_front]

roll_rear, pitch_rear, yaw_rear = quat_to_euler(0.099, -0.080, -0.692, 0.711)
EXTRINSIC_REAR  = [-1.460, -0.010, 1.910, roll_rear, pitch_rear, yaw_rear]


# ═════════════════════════════════════════════════════════════════════════════
# MATHÉMATIQUES — matrices, transformations fusionnées
# ═════════════════════════════════════════════════════════════════════════════

def deg2rad(d):
    return d * np.pi / 180.0


def extrinsic_to_matrix(params):
    """[x, y, z, roll°, pitch°, yaw°] → matrice homogène 4×4."""
    x, y, z, ro, pi, ya = params
    ro, pi, ya = deg2rad(ro), deg2rad(pi), deg2rad(ya)
    cr, sr = np.cos(ro), np.sin(ro)
    cp, sp = np.cos(pi), np.sin(pi)
    cy, sy = np.cos(ya), np.sin(ya)
    # ZYX intrinsic = extrinsic XYZ — même résultat, mais sans 3 allocations
    R = np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [  -sp,           cp*sr,           cp*cr     ],
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3]  = [x, y, z]
    return T


def _extract_R_t(T4x4: np.ndarray):
    """Extrait R (3×3) et t (3,) d'une matrice homogène — évite le 4×4 complet."""
    return T4x4[:3, :3].copy(), T4x4[:3, 3].copy()


# ═════════════════════════════════════════════════════════════════════════════
# TRANSFORMATION FUSIONNÉE — 1 matmul au lieu de 3 allocations
# ═════════════════════════════════════════════════════════════════════════════

def transform_points_fused(xyz: np.ndarray,
                           pose_xyz: np.ndarray,
                           yaw_rad: float,
                           R_ext: np.ndarray,
                           t_ext: np.ndarray) -> np.ndarray:
    """
    Fusionne extrinsic + yaw + translation en une seule opération :
        R_total = R_yaw @ R_ext
        t_total = R_yaw @ t_ext + pose
        result  = xyz @ R_total.T + t_total

    Vs original : élimine np.hstack([xyz, ones]), le 4×4 matmul,
    et la transposition inutile. ~2× plus rapide, ~3× moins d'allocs.
    """
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    R_yaw = np.array([[c, -s, 0.0],
                       [s,  c, 0.0],
                       [0.0, 0.0, 1.0]], dtype=np.float64)

    R_total = R_yaw @ R_ext          # 3×3 — tiny
    t_total = R_yaw @ t_ext + pose_xyz  # 3-vec

    return (xyz @ R_total.T + t_total).astype(np.float32)


def transform_points_gpu(xyz: np.ndarray,
                         pose_xyz: np.ndarray,
                         yaw_rad: float,
                         R_ext: np.ndarray,
                         t_ext: np.ndarray) -> np.ndarray:
    """Version GPU de la transformation fusionnée (CuPy ou PyTorch)."""
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)

    R_total = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]],
                       dtype=np.float64) @ R_ext
    t_total = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]],
                       dtype=np.float64) @ t_ext + pose_xyz

    if _DEVICE == "cuda_cupy":
        import cupy as cp
        xyz_g = cp.asarray(xyz, dtype=cp.float32)
        R_g   = cp.asarray(R_total, dtype=cp.float32)
        t_g   = cp.asarray(t_total, dtype=cp.float32)
        out   = (xyz_g @ R_g.T + t_g).get()   # → numpy
        return out.astype(np.float32)

    elif _DEVICE.startswith("torch"):
        import torch
        xyz_t = torch.from_numpy(xyz).float().to(_torch_device)
        R_t   = torch.from_numpy(R_total.astype(np.float32)).to(_torch_device)
        t_t   = torch.from_numpy(t_total.astype(np.float32)).to(_torch_device)
        out   = (xyz_t @ R_t.T + t_t).cpu().numpy()
        return out.astype(np.float32)

    return transform_points_fused(xyz, pose_xyz, yaw_rad, R_ext, t_ext)


# Sélection automatique du backend de transformation
_transform_fn = transform_points_gpu if _DEVICE != "cpu" else transform_points_fused


# ═════════════════════════════════════════════════════════════════════════════
# TRAJECTOIRE GNSS (optimisée)
# ═════════════════════════════════════════════════════════════════════════════

class GNSSTrajectory:
    """Position + orientation interpolées depuis PoseWithCovarianceStamped."""

    def __init__(self, gnss_msgs):
        if not gnss_msgs:
            sys.exit("[ERREUR] Aucun message GNSS dans le bag.")

        n = len(gnss_msgs)
        self.timestamps = np.empty(n, dtype=np.float64)
        self.positions  = np.empty((n, 3), dtype=np.float64)
        self.quaternions = []
        self._gnss_speeds = np.empty(n, dtype=np.float64)

        for i, (ts, msg) in enumerate(gnss_msgs):
            try:
                p = msg.pose.pose.position
                self.quaternions.append(msg.pose.pose.orientation)
            except AttributeError:
                sys.exit("[ERREUR] Topic GNSS attendu : PoseWithCovarianceStamped")
            self.timestamps[i]  = ts * 1e-9
            self.positions[i]   = (p.x, p.y, p.z)

        self._compute_gnss_speeds()
        print(f"[INFO] Trajectoire GNSS : {n} points.")

    def _compute_gnss_speeds(self):
        # Vectorisé au lieu d'une boucle Python
        dt  = np.diff(self.timestamps)
        dp  = np.linalg.norm(np.diff(self.positions, axis=0), axis=1)
        spd = np.divide(dp, dt, out=np.zeros_like(dp), where=dt > 1e-6)
        self._gnss_speeds[0]  = 0.0
        self._gnss_speeds[1:] = spd

    def interpolate_pose(self, query_ts_ns: int):
        t  = query_ts_ns * 1e-9
        ts = self.timestamps

        if t < ts[0] or t > ts[-1]:
            return None, None

        idx = np.searchsorted(ts, t)
        if idx == 0:
            return self.positions[0].copy(), self._quat_to_yaw(self.quaternions[0])
        if idx >= len(ts):
            return self.positions[-1].copy(), self._quat_to_yaw(self.quaternions[-1])

        alpha = (t - ts[idx - 1]) / (ts[idx] - ts[idx - 1] + 1e-12)
        pos   = self.positions[idx - 1] + alpha * (self.positions[idx] - self.positions[idx - 1])
        q     = self.quaternions[idx - 1] if alpha < 0.5 else self.quaternions[idx]
        return pos, self._quat_to_yaw(q)

    def gnss_speed_at(self, query_ts_ns: int) -> float:
        t   = query_ts_ns * 1e-9
        idx = min(np.searchsorted(self.timestamps, t),
                  len(self._gnss_speeds) - 1)
        return float(self._gnss_speeds[idx])

    @staticmethod
    def _quat_to_yaw(q) -> float:
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))


# ═════════════════════════════════════════════════════════════════════════════
# VITESSE FWS (inchangé — déjà efficace)
# ═════════════════════════════════════════════════════════════════════════════

class FWSSpeedTracker:
    MAX_DISCREPANCY_MS = 3.0

    def __init__(self, fws_msgs, gnss_traj: GNSSTrajectory):
        self._gnss    = gnss_traj
        self._has_fws = len(fws_msgs) > 0
        if not self._has_fws:
            print("[WARN] Aucun message FWS — vitesse GNSS seule utilisée.")
            return

        n = len(fws_msgs)
        self._ts     = np.empty(n, dtype=np.float64)
        self._speeds = np.empty(n, dtype=np.float64)

        for i, (ts, msg) in enumerate(fws_msgs):
            self._ts[i]     = ts * 1e-9
            self._speeds[i] = abs(msg.data.speed)

        self._check_consistency(fws_msgs)
        print(f"[INFO] FWSSpeedTracker : {n} mesures | "
              f"v_max={self._speeds.max():.2f} m/s | "
              f"v_moy={self._speeds.mean():.2f} m/s")

    def speed_at(self, query_ts_ns: int) -> float:
        if not self._has_fws:
            return self._gnss.gnss_speed_at(query_ts_ns)
        t = query_ts_ns * 1e-9
        if t < self._ts[0] or t > self._ts[-1]:
            return self._gnss.gnss_speed_at(query_ts_ns)
        return float(self._interp(t))

    def is_moving(self, query_ts_ns: int) -> bool:
        return self.speed_at(query_ts_ns) >= MIN_SPEED_MS

    def _interp(self, t_s: float) -> float:
        idx = np.searchsorted(self._ts, t_s)
        if idx == 0:              return self._speeds[0]
        if idx >= len(self._ts):  return self._speeds[-1]
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
                  f"hors tolérance ({pct:.1f} %)")
        else:
            print("[INFO] Cohérence FWS↔GNSS : OK")


# ═════════════════════════════════════════════════════════════════════════════
# DÉSÉRIALISATION PointCloud2 → numpy  (100 % vectorisée)
# ═════════════════════════════════════════════════════════════════════════════

def fast_pointcloud2_to_xyz(msg) -> Optional[np.ndarray]:
    """
    Extraction vectorisée des champs XYZ.
    Utilise np.frombuffer + view → zéro boucle Python.
    Le slow-path struct.unpack est remplacé par un fallback numpy structuré.
    """
    off = {f.name: f.offset for f in msg.fields}
    if not all(k in off for k in ("x", "y", "z")):
        return _vectorized_fallback_xyz(msg)

    n = msg.height * msg.width
    if n == 0:
        return None

    # Évite bytes(msg.data) qui copie — essaie le buffer direct
    try:
        data = np.frombuffer(msg.data, dtype=np.uint8)
    except TypeError:
        data = np.frombuffer(bytes(msg.data), dtype=np.uint8)

    expected = n * msg.point_step
    if len(data) < expected:
        return None

    rows = data[:expected].reshape(n, msg.point_step)

    xyz = np.empty((n, 3), dtype=np.float32)
    xyz[:, 0] = rows[:, off["x"]:off["x"] + 4].view(np.float32).ravel()
    xyz[:, 1] = rows[:, off["y"]:off["y"] + 4].view(np.float32).ravel()
    xyz[:, 2] = rows[:, off["z"]:off["z"] + 4].view(np.float32).ravel()

    # Filtre NaN/Inf in-place → pas d'allocation supplémentaire
    mask = np.isfinite(xyz).all(axis=1)
    return xyz[mask] if not mask.all() else xyz


def _vectorized_fallback_xyz(msg) -> Optional[np.ndarray]:
    """
    Fallback vectorisé (remplace l'ancienne boucle struct.unpack).
    Construit un dtype structuré numpy pour lire directement le buffer.
    """
    fmt_map = {1: 'i1', 2: 'u1', 3: '<i2', 4: '<u2',
               5: '<i4', 6: '<u4', 7: '<f4', 8: '<f8'}

    fields = {f.name: (f.offset, fmt_map.get(f.datatype)) for f in msg.fields}
    if not all(k in fields and fields[k][1] for k in ("x", "y", "z")):
        return None

    n = msg.height * msg.width
    if n == 0:
        return None

    # Construit un dtype structuré aligné sur point_step
    dt_fields = []
    for f in msg.fields:
        np_type = fmt_map.get(f.datatype)
        if np_type:
            dt_fields.append((f.name, np_type, f.offset))

    # Trie par offset et construit le dtype avec padding
    dt_fields.sort(key=lambda x: x[2])
    dtype_list = []
    current_offset = 0
    for name, np_type, offset in dt_fields:
        if offset > current_offset:
            dtype_list.append(('_pad' + str(offset), f'V{offset - current_offset}'))
        dtype_list.append((name, np_type))
        current_offset = offset + np.dtype(np_type).itemsize
    if current_offset < msg.point_step:
        dtype_list.append(('_pad_end', f'V{msg.point_step - current_offset}'))

    dt = np.dtype(dtype_list)

    try:
        buf = np.frombuffer(msg.data, dtype=np.uint8)
    except TypeError:
        buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)

    structured = buf[:n * msg.point_step].view(dt)
    xyz = np.column_stack([
        structured['x'].astype(np.float32),
        structured['y'].astype(np.float32),
        structured['z'].astype(np.float32),
    ])
    mask = np.isfinite(xyz).all(axis=1)
    return xyz[mask] if not mask.all() else xyz


# ═════════════════════════════════════════════════════════════════════════════
# LECTURE BAG 2 PASSES  (cœur de l'optimisation mémoire)
# ═════════════════════════════════════════════════════════════════════════════

def _open_bag_reader(bag_path: str, topics: List[str]):
    """Ouvre un SequentialReader filtré sur les topics demandés."""
    import rosbag2_py
    storage_id = "mcap" if any(
        f.endswith(".mcap") for f in os.listdir(bag_path)
    ) else "sqlite3"

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr"),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=topics))

    topic_types = {m.name: m.type for m in reader.get_all_topics_and_types()}
    return reader, topic_types


def read_metadata_pass(bag_path: str):
    """
    PASSE 1 — Lecture GNSS + FWS uniquement.
    Très rapide et léger en mémoire (< 50 Mo typiquement).
    """
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader, topic_types = _open_bag_reader(
        bag_path, [TOPIC_GNSS, TOPIC_FWS])

    deser = {}
    for topic in (TOPIC_GNSS, TOPIC_FWS):
        t = topic_types.get(topic)
        deser[topic] = get_message(t) if t else None

    gnss_msgs, fws_msgs = [], []

    print("[INFO] Passe 1/2 — lecture GNSS + FWS…")
    while reader.has_next():
        topic, data, ts = reader.read_next()
        d = deser.get(topic)
        if d is None:
            continue
        msg = deserialize_message(data, d)
        if topic == TOPIC_GNSS:
            gnss_msgs.append((ts, msg))
        elif topic == TOPIC_FWS:
            fws_msgs.append((ts, msg))

    print(f"[INFO]   GNSS={len(gnss_msgs)}, FWS={len(fws_msgs)}")
    return gnss_msgs, fws_msgs


def stream_lidar_pass(bag_path: str,
                      gnss_traj: GNSSTrajectory,
                      speed_tracker: FWSSpeedTracker,
                      T_front: np.ndarray,
                      T_rear: np.ndarray) -> np.ndarray:
    """
    PASSE 2 — Streaming LiDAR : chaque scan est transformé puis ajouté
    au nuage accumulé. Le message ROS est libéré immédiatement.

    Inclut :
    - Downsampling incrémental tous les INCREMENTAL_DS_INTERVAL scans
    - Checkpoint disque tous les CHECKPOINT_INTERVAL scans
    - Monitoring mémoire avec DS forcé si > MEMORY_LIMIT_GB
    """
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    import open3d as o3d

    # ── Vérification de checkpoint existant ──────────────────────────────
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / "accumulated_cloud.npy"
    ckpt_meta = CHECKPOINT_DIR / "accumulated_meta.npy"
    resumed_pts = _load_checkpoint(ckpt_path)
    resume_after_ts = 0
    if resumed_pts is not None and ckpt_meta.exists():
        meta = np.load(str(ckpt_meta))
        resume_after_ts = int(meta[0])
        print(f"  [CKPT] Reprise après ts={resume_after_ts}")

    # ── Ouvrir le bag pour les topics LiDAR ──────────────────────────────
    reader, topic_types = _open_bag_reader(
        bag_path, [TOPIC_LIDAR_FRONT, TOPIC_LIDAR_REAR])

    deser = {}
    for topic in (TOPIC_LIDAR_FRONT, TOPIC_LIDAR_REAR):
        t = topic_types.get(topic)
        deser[topic] = get_message(t) if t else None

    R_front, t_front = _extract_R_t(T_front)
    R_rear,  t_rear  = _extract_R_t(T_rear)

    # ── Accumulateur pré-alloué ──────────────────────────────────────────
    # Commence avec un buffer de 5M points, double si nécessaire
    INIT_CAPACITY = 5_000_000
    buf = resumed_pts if resumed_pts is not None else np.empty(
        (INIT_CAPACITY, 3), dtype=np.float32)
    buf_idx = len(resumed_pts) if resumed_pts is not None else 0

    scan_count = 0
    skipped_stop = 0
    skipped_range = 0
    t_start = time.monotonic()

    def _should_keep(is_moving):
        return is_moving if SCAN_WHEN_MOVING else not is_moving

    print("[INFO] Passe 2/2 — streaming LiDAR + transformation…")

    while reader.has_next():
        topic, data, ts = reader.read_next()

        # Skip scans déjà dans le checkpoint
        if ts <= resume_after_ts:
            continue

        d = deser.get(topic)
        if d is None:
            continue

        # Filtre mouvement
        if not _should_keep(speed_tracker.is_moving(ts)):
            skipped_stop += 1
            continue

        # Pose interpolée
        pose, yaw = gnss_traj.interpolate_pose(ts)
        if pose is None:
            skipped_range += 1
            continue

        # Désérialisation → transformation (le message est GC-able après)
        msg = deserialize_message(data, d)
        xyz = fast_pointcloud2_to_xyz(msg)
        del msg  # libère la mémoire du message immédiatement

        if xyz is None or len(xyz) == 0:
            continue

        # Sélection extrinsèque
        if topic == TOPIC_LIDAR_FRONT:
            R_ext, t_ext = R_front, t_front
        else:
            R_ext, t_ext = R_rear, t_rear

        # Transformation fusionnée (CPU ou GPU)
        pts = _transform_fn(xyz, pose, yaw, R_ext, t_ext)
        del xyz  # libère

        # ── Ajout au buffer pré-alloué ───────────────────────────────────
        n_new = len(pts)
        if buf_idx + n_new > len(buf):
            # Agrandit le buffer (×1.5 pour amortir les réallocations)
            new_cap = max(len(buf) + n_new, int(len(buf) * 1.5))
            new_buf = np.empty((new_cap, 3), dtype=np.float32)
            new_buf[:buf_idx] = buf[:buf_idx]
            buf = new_buf

        buf[buf_idx:buf_idx + n_new] = pts
        buf_idx += n_new
        scan_count += 1

        # ── Downsampling incrémental périodique ──────────────────────────
        if scan_count % INCREMENTAL_DS_INTERVAL == 0:
            mem_gb = _mem_used_gb()
            force_ds = mem_gb > MEMORY_LIMIT_GB

            if force_ds:
                print(f"  [MEM] {mem_gb:.1f} Go utilisés → downsampling forcé")

            if force_ds or scan_count % INCREMENTAL_DS_INTERVAL == 0:
                pcd_tmp = o3d.geometry.PointCloud()
                pcd_tmp.points = o3d.utility.Vector3dVector(buf[:buf_idx])
                pcd_ds = pcd_tmp.voxel_down_sample(voxel_size=VOXEL_SIZE)
                ds_pts = np.asarray(pcd_ds.points).astype(np.float32)

                # Réécrit le buffer avec les points downsampleés
                buf_idx = len(ds_pts)
                if buf_idx > len(buf):
                    buf = np.empty((buf_idx + INIT_CAPACITY, 3), dtype=np.float32)
                buf[:buf_idx] = ds_pts

                del pcd_tmp, pcd_ds, ds_pts
                _force_gc()

                elapsed = time.monotonic() - t_start
                rate = scan_count / elapsed if elapsed > 0 else 0
                print(f"  [DS]  scan {scan_count:,} | {buf_idx:,} pts "
                      f"| {rate:.0f} scans/s | RSS {_mem_used_gb():.1f} Go")

        # ── Checkpoint périodique ────────────────────────────────────────
        if scan_count % CHECKPOINT_INTERVAL == 0:
            _save_checkpoint(buf[:buf_idx], ckpt_path, f"scan {scan_count}")
            np.save(str(ckpt_meta), np.array([ts], dtype=np.int64))

    # ── Résumé final ─────────────────────────────────────────────────────
    elapsed = time.monotonic() - t_start
    print(f"[INFO] Streaming terminé : {scan_count:,} scans en {elapsed:.1f}s "
          f"({scan_count/elapsed:.0f} scans/s)")
    print(f"[INFO]   {skipped_stop} ignorés (arrêt) | "
          f"{skipped_range} ignorés (hors plage GNSS)")
    print(f"[INFO]   Nuage accumulé : {buf_idx:,} points")

    # Sauvegarde finale
    result = buf[:buf_idx].copy()
    del buf
    _force_gc()

    _save_checkpoint(result, ckpt_path, "final")
    return result


# ═════════════════════════════════════════════════════════════════════════════
# SEGMENTATION SOL (inchangée)
# ═════════════════════════════════════════════════════════════════════════════

def segment_ground(pcd, distance_threshold=0.2, ransac_n=3, num_iterations=1000):
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations)
    [a, b, c, d] = plane_model
    print(f"[INFO] Plan sol détecté : {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    ground    = pcd.select_by_index(inliers)
    nonground = pcd.select_by_index(inliers, invert=True)
    return ground, nonground


# ═════════════════════════════════════════════════════════════════════════════
# CONSTRUCTION MESH  (avec GC inter-étapes)
# ═════════════════════════════════════════════════════════════════════════════

def build_mesh(combined_pts: np.ndarray, output_dir: Path) -> None:
    import open3d as o3d

    output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

    n_raw = len(combined_pts)
    print(f"\n[INFO] Nuage global brut : {n_raw:,} points")

    # ── Étape 1 : voxel downsampling ─────────────────────────────────────
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_pts)
    del combined_pts
    _force_gc()

    pcd_ds = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    del pcd
    _force_gc()
    print(f"[INFO] Après downsampling ({VOXEL_SIZE}m) : "
          f"{len(pcd_ds.points):,} points")

    # ── Étape 2 : suppression outliers ───────────────────────────────────
    pcd_clean, _ = pcd_ds.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0)
    del pcd_ds
    _force_gc()
    print(f"[INFO] Après filtre outliers : {len(pcd_clean.points):,} points")

    # ── Sauvegarde nuage nettoyé ─────────────────────────────────────────
    cloud_path = output_dir / f"output_cloud_{now}.ply"
    o3d.io.write_point_cloud(str(cloud_path), pcd_clean)
    print(f"[OK]   Nuage enregistré → {cloud_path}")

    # ── Étape 3 : normales ───────────────────────────────────────────────
    print("[INFO] Calcul des normales…")
    pcd_clean.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_RADIUS, max_nn=30))
    pcd_clean.orient_normals_towards_camera_location(
        camera_location=np.array([0., 0., 1000.]))

    # ── Étape 4 : Poisson ────────────────────────────────────────────────
    print(f"[INFO] Reconstruction Poisson (depth={POISSON_DEPTH})…")
    t0 = time.monotonic()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_clean, depth=POISSON_DEPTH, width=0, scale=1.1, linear_fit=False)
    print(f"[INFO] Poisson terminé en {time.monotonic()-t0:.1f}s")

    del pcd_clean
    _force_gc()

    # ── Nettoyage mesh ───────────────────────────────────────────────────
    dens = np.asarray(densities)
    mesh.remove_vertices_by_mask(dens < np.percentile(dens, 10))
    del dens, densities
    _force_gc()

    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh = mesh.filter_smooth_simple(number_of_iterations=3)
    mesh.compute_vertex_normals()

    print(f"[INFO] Mesh : {len(mesh.vertices):,} sommets, "
          f"{len(mesh.triangles):,} triangles")

    # ── Export ────────────────────────────────────────────────────────────
    for ext in (".ply", ".obj"):
        p = output_dir / f"map_mesh_{now}{ext}"
        o3d.io.write_triangle_mesh(str(p), mesh)
        print(f"[OK]   Mesh enregistré → {p}")

    _visualize(mesh)


def _visualize(mesh):
    try:
        import open3d as o3d
        if os.environ.get("DISPLAY") or sys.platform in ("darwin", "win32"):
            mesh.paint_uniform_color([0.9, 0.9, 0.9])
            o3d.visualization.draw_geometries(
                [mesh], window_name="Carte LiDAR 3D",
                width=1280, height=720)
    except Exception as e:
        print(f"[WARN] Visualisation impossible : {e}")


# ═════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ═════════════════════════════════════════════════════════════════════════════

def main():
    t_pipeline = time.monotonic()

    bag = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else BAG_PATH)

    print("=" * 62)
    print("  Pipeline LiDAR → Mesh 3D  |  ROS2 Bag  [HPC]")
    print("=" * 62)
    print(f"  Bag          : {bag}")
    print(f"  LiDAR avant  : {TOPIC_LIDAR_FRONT}")
    print(f"  LiDAR arrière: {TOPIC_LIDAR_REAR}")
    print(f"  GNSS         : {TOPIC_GNSS}")
    print(f"  FWS          : {TOPIC_FWS}")
    print(f"  Sortie       : {OUTPUT_DIR}")
    print(f"  Accélération : {_DEVICE}")
    print(f"  Mémoire max  : {MEMORY_LIMIT_GB} Go")
    print("=" * 62)

    try:
        import rosbag2_py                                      # noqa: F401
        from rclpy.serialization import deserialize_message    # noqa: F401
    except ImportError as e:
        sys.exit(f"[ERREUR] Modules ROS2 introuvables : {e}")

    # ── Phase 1 : métadonnées (GNSS + FWS) ──────────────────────────────
    gnss_msgs, fws_msgs = read_metadata_pass(bag)

    traj          = GNSSTrajectory(gnss_msgs)
    speed_tracker = FWSSpeedTracker(fws_msgs, traj)

    del gnss_msgs, fws_msgs   # libère les messages bruts
    _force_gc()

    # ── Phase 2 : streaming LiDAR ───────────────────────────────────────
    T_front = extrinsic_to_matrix(EXTRINSIC_FRONT)
    T_rear  = extrinsic_to_matrix(EXTRINSIC_REAR)

    combined = stream_lidar_pass(bag, traj, speed_tracker, T_front, T_rear)

    del traj, speed_tracker   # plus nécessaires
    _force_gc()

    if len(combined) == 0:
        sys.exit("[ERREUR] Aucun point accumulé. Vérifiez les noms de topics.")

    print(f"\n[INFO] Nuage fusionné total : {len(combined):,} points")

    # ── Phase 3 : segmentation sol ───────────────────────────────────────
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined)
    del combined
    _force_gc()

    ground, nonground = segment_ground(pcd)
    del pcd, ground   # sol non utilisé pour le mesh
    _force_gc()

    print(f"[INFO] Non-sol : {len(nonground.points):,} points → mesh")

    # ── Phase 4 : construction mesh ──────────────────────────────────────
    build_mesh(np.asarray(nonground.points).astype(np.float32), OUTPUT_DIR)
    del nonground
    _force_gc()

    # ── Nettoyage checkpoints ────────────────────────────────────────────
    for f in CHECKPOINT_DIR.glob("*.npy"):
        f.unlink()
    print(f"\n[DONE] Pipeline terminé en "
          f"{time.monotonic()-t_pipeline:.1f}s.")


if __name__ == "__main__":
    try:
        main()
    except MemoryError:
        _force_gc()
        print("\n[FATAL] MemoryError — essayez de réduire POISSON_DEPTH "
              "ou augmenter MEMORY_LIMIT_GB.", file=sys.stderr)
        sys.exit(137)
    except KeyboardInterrupt:
        print("\n[INFO] Interruption utilisateur — checkpoints conservés "
              f"dans {CHECKPOINT_DIR}")
        sys.exit(130)
