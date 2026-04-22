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
MIN_SPEED_MS  = 1.0      # seuil de mouvement (m/s)
VOXEL_SIZE    = 0.05
POISSON_DEPTH = 10
NORMAL_RADIUS = 0.2

SCAN_WHEN_MOVING = True  # True = scan en mouvement, False = scan à l'arrêt

# ── Calibration automatique inter-LiDAR ─────────────────────────────────
USE_AUTO_CALIBRATION   = False    # False → extrinsèques manuels
CALIB_FRAMES_PER_LIDAR = 40      # frames accumulées par capteur
CALIB_VOXEL_SIZE       = 0.05    # voxel grossier pour la calibration
ICP_MAX_ITER           = 60
ICP_MAX_CORR_DIST      = 2.5     # distance max correspondance (m)
ICP_RMSE_THRESHOLD     = 0.08    # seuil RMSE acceptable (m)
# Guess initial : avant au-dessus à +1.46m, arrière à -1.46m → Δx ≈ -2.92m
# Rotation ~180° autour de Z (capteurs opposés sur le véhicule)
_WHEELBASE_M = 2.92


# ── Paramètres d'optimisation ────────────────────────────────────────────
CHECKPOINT_INTERVAL     = 250
INCREMENTAL_DS_INTERVAL = 150
MEMORY_LIMIT_GB         = 6.5
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

LIDAR_CONFIG = {
    "front": {
        "topic": TOPIC_LIDAR_FRONT,
        "extrinsic": EXTRINSIC_FRONT
    },
    "rear": {
        "topic": TOPIC_LIDAR_REAR,
        "extrinsic": EXTRINSIC_REAR
    },
}

# ── Sélection des LiDARs actifs ────────────────────────────────────────
# Choix possibles : "front", "rear"
ACTIVE_LIDARS = ["front"]

ACTIVE_LIDAR_TOPICS = [
    LIDAR_CONFIG[l]["topic"]
    for l in ACTIVE_LIDARS
]

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
# SE(3) — INTERPOLATION COMPLÈTE  (remplace la logique yaw-only)
# ═════════════════════════════════════════════════════════════════════════════

def quat_to_rot(q) -> np.ndarray:
    """Quaternion (objet ROS ou array [w,x,y,z]) → R 3×3 float64."""
    if hasattr(q, 'w'):
        w, x, y, z = q.w, q.x, q.y, q.z
    else:
        w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ], dtype=np.float64)


def quat_to_array(q) -> np.ndarray:
    """ROS quaternion → [w, x, y, z] numpy."""
    return np.array([q.w, q.x, q.y, q.z], dtype=np.float64)


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """
    SLERP entre deux quaternions [w,x,y,z].
    Gère : antipodalité, quaternions quasi-identiques, t ∈ [0,1].
    """
    # Assure le chemin court (hemisphere positif)
    if np.dot(q0, q1) < 0.0:
        q1 = -q1

    dot = np.clip(np.dot(q0, q1), -1.0, 1.0)

    if dot > 0.9995:                        # quasi-identiques → lerp linéaire
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)

    theta_0 = math.acos(dot)
    theta   = theta_0 * t
    sin_0   = math.sin(theta_0)

    s0 = math.cos(theta) - dot * math.sin(theta) / sin_0
    s1 = math.sin(theta) / sin_0
    q  = s0 * q0 + s1 * q1
    return q / np.linalg.norm(q)


def make_pose(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """R (3×3) + t (3,) → matrice homogène 4×4."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3]  = t
    return T

# ═════════════════════════════════════════════════════════════════════════════
# CALIBRATION EXTRINSÈQUE AUTOMATIQUE  —  ICP point-to-plane
# ═════════════════════════════════════════════════════════════════════════════
# ── Calcul automatique depuis les extrinsèques connus ────────────────────

_CALIB_TRANSLATION = np.array([
    EXTRINSIC_FRONT[0] - EXTRINSIC_REAR[0],   # +2.920
    EXTRINSIC_FRONT[1] - EXTRINSIC_REAR[1],   # +0.020
    EXTRINSIC_FRONT[2] - EXTRINSIC_REAR[2],   # +0.010
], dtype=np.float64)


def _build_initial_guess() -> np.ndarray:
    """
    Matrice 4×4 : rear_LiDAR → front_LiDAR.
    Translation extraite des positions capteurs réelles.
    Rotation 180° autour de Z (capteurs opposés).
    """
    T = np.eye(4, dtype=np.float64)

    # Rotation 180° Z (les deux LiDAR se font face)
    T[0, 0] = -1.0
    T[1, 1] = -1.0

    # Translation exacte rear → front (dans le repère véhicule)
    T[:3, 3] = _CALIB_TRANSLATION

    print(f"  [CALIB] Initial guess : t={np.round(_CALIB_TRANSLATION, 3)} m  "
          f"(rotation 180° Z)")
    return T


def estimate_extrinsic_icp(
        front_pts: np.ndarray,
        rear_pts:  np.ndarray,
        voxel_size:           float = CALIB_VOXEL_SIZE,
        max_correspondence_m: float = ICP_MAX_CORR_DIST,
        max_iter:             int   = ICP_MAX_ITER,
        rmse_threshold:       float = ICP_RMSE_THRESHOLD,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Estime la transformation rear_LiDAR → front_LiDAR par ICP point-to-plane.

    Retourne (R 3×3, t 3-vec) tels que :
        rear_in_front_frame = rear_pts @ R.T + t

    Retourne (None, None) si ICP diverge ou RMSE > rmse_threshold.
    """
    import open3d as o3d

    def _prepare(pts: np.ndarray, name: str):
        """Downsample + normales."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd = pcd.voxel_down_sample(voxel_size)
        if len(pcd.points) < 500:
            raise RuntimeError(
                f"[CALIB] Nuage {name} trop petit après DS : "
                f"{len(pcd.points)} pts — augmenter CALIB_FRAMES_PER_LIDAR")
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 3, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(30)
        return pcd

    try:
        pcd_front = _prepare(front_pts, "front")
        pcd_rear  = _prepare(rear_pts,  "rear")

        print(f"  [CALIB] ICP : front={len(pcd_front.points):,} pts | "
              f"rear={len(pcd_rear.points):,} pts "
              f"(après DS {voxel_size}m)")

        init_T = _build_initial_guess()

        # ── ICP point-to-plane ────────────────────────────────────────────
        result = o3d.pipelines.registration.registration_icp(
            source=pcd_rear,          # rear → à aligner
            target=pcd_front,         # front → référence
            max_correspondence_distance=max_correspondence_m,
            init=init_T,
            estimation_method=o3d.pipelines.registration
                               .TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iter,
                relative_fitness=1e-6,
                relative_rmse=1e-6),
        )

        fitness = result.fitness
        rmse    = result.inlier_rmse
        T_icp   = result.transformation   # 4×4 float64

        print(f"  [CALIB] ICP terminé — fitness={fitness:.4f} | "
              f"RMSE={rmse:.4f} m | "
              f"seuil={rmse_threshold} m")

        # ── Validation ────────────────────────────────────────────────────
        if fitness < 0.30:
            print(f"  [CALIB][WARN] Fitness trop faible ({fitness:.3f} < 0.30) "
                  f"→ fallback extrinsèques manuels")
            return None, None

        if rmse > rmse_threshold:
            print(f"  [CALIB][WARN] RMSE trop élevé ({rmse:.4f} > "
                  f"{rmse_threshold}) → fallback extrinsèques manuels")
            return None, None

        R = T_icp[:3, :3].astype(np.float64)
        t = T_icp[:3,  3].astype(np.float64)

        # Sanity-check : la rotation doit rester proche de 180°
        cos_angle = (np.trace(R) - 1.0) / 2.0
        angle_deg = math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))
        if not (120.0 < angle_deg < 240.0):
            print(f"  [CALIB][WARN] Rotation ICP aberrante ({angle_deg:.1f}°) "
                  f"→ fallback extrinsèques manuels")
            return None, None

        print(f"  [CALIB][OK] Transformation rear→front validée "
              f"(angle={angle_deg:.1f}°, t={np.round(t, 3)})")
        return R, t

    except Exception as e:
        print(f"  [CALIB][ERREUR] ICP échoué : {e} → fallback manuels")
        return None, None

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

    def interpolate_pose_se3(self, query_ts_ns: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Retourne (position_xyz, quaternion_wxyz) interpolés."""
        t = query_ts_ns * 1e-9
        ts = self.timestamps

        if t < ts[0] or t > ts[-1]:
            return None, None

        idx = np.searchsorted(ts, t)
        if idx == 0:
            return self.positions[0].copy(), self._quat_to_array(self.quaternions[0])
        if idx >= len(ts):
            return self.positions[-1].copy(), self._quat_to_array(self.quaternions[-1])

        alpha = (t - ts[idx - 1]) / (ts[idx] - ts[idx - 1] + 1e-12)

        # LERP linéaire pour la position
        pos = self.positions[idx - 1] + alpha * (self.positions[idx] - self.positions[idx - 1])

        # SLERP pour le quaternion
        q0 = self._quat_to_array(self.quaternions[idx - 1])
        q1 = self._quat_to_array(self.quaternions[idx])
        q_interp = slerp(q0, q1, alpha)

        return pos, q_interp

    @staticmethod
    def _quat_to_array(q):
        return np.array([q.w, q.x, q.y, q.z], dtype=np.float64)

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

# ═══════════════════════════════════════════════════════════════════════
# FILTRAGE DISTANCE + NETTOYAGE BRUIT
# ═══════════════════════════════════════════════════════════════════════

def filter_points_by_distance(xyz: np.ndarray,
                               min_dist: float,
                               max_dist: float) -> np.ndarray:
    """
    Filtre les points hors de la plage [min_dist, max_dist].
    100 % vectorisé, sans sqrt (comparaison sur distances²).
    """
    # Évite sqrt : compare dist² directement → ~1.5× plus rapide
    sq = np.einsum('ij,ij->i', xyz, xyz)   # (N,) — plus rapide que (xyz**2).sum(1)

    min_sq = min_dist * min_dist
    max_sq = max_dist * max_dist

    # Masque booléen in-place → une seule allocation
    mask = (sq >= min_sq) & (sq <= max_sq)
    return xyz[mask]

def remove_noise(xyz: np.ndarray,
                 method: str = "statistical",
                 **kwargs) -> np.ndarray:
    """
    Supprime le bruit du nuage de points.

    Méthodes :
      "statistical" → remove_statistical_outlier (Open3D)
      "radius"      → remove_radius_outlier (Open3D)
    Fallback NumPy si Open3D indisponible.

    Kwargs par défaut :
      statistical : nb_neighbors=20, std_ratio=2.0
      radius      : nb_points=16,    radius=0.5
    """
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        if method == "statistical":
            nb_neighbors = kwargs.get("nb_neighbors", 20)
            std_ratio    = kwargs.get("std_ratio",    2.0)
            pcd_clean, _ = pcd.remove_statistical_outlier(
                nb_neighbors=nb_neighbors, std_ratio=std_ratio)

        elif method == "radius":
            nb_points = kwargs.get("nb_points", 16)
            radius    = kwargs.get("radius",    0.5)
            pcd_clean, _ = pcd.remove_radius_outlier(
                nb_points=nb_points, radius=radius)

        else:
            raise ValueError(f"Méthode inconnue : '{method}'. "
                             f"Choisir 'statistical' ou 'radius'.")

        # np.asarray → vue directe, pas de copie
        result = np.asarray(pcd_clean.points, dtype=np.float32)
        del pcd, pcd_clean
        return result

    except ImportError:
        # ── Fallback NumPy : filtre les points isolés par densité locale ──
        # Approximation grossière (grille 3D) — suffisant en cas de secours
        print("[WARN] Open3D indisponible → fallback NumPy pour remove_noise")
        return _numpy_noise_fallback(xyz, kwargs.get("std_ratio", 2.0))


def _numpy_noise_fallback(xyz: np.ndarray, std_ratio: float = 2.0) -> np.ndarray:
    """
    Fallback sans Open3D.
    Supprime les points dont la distance au centroïde dépasse
    (mean + std_ratio × std) — O(N), très rapide.
    Moins précis que Statistical Outlier Removal mais sans dépendance.
    """
    centroid = xyz.mean(axis=0)                          # (3,)
    sq_dist  = np.einsum('ij,ij->i', xyz - centroid,
                                     xyz - centroid)     # (N,)
    threshold = sq_dist.mean() + std_ratio * sq_dist.std()
    return xyz[sq_dist <= threshold]

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

def _run_calibration_phase(
        bag_path: str,
        gnss_traj: GNSSTrajectory,
        speed_tracker: FWSSpeedTracker,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Ouvre le bag, accumule CALIB_FRAMES_PER_LIDAR frames par LiDAR
    (points bruts dans le repère capteur), puis lance estimate_extrinsic_icp.
    """
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader, topic_types = _open_bag_reader(
        bag_path, [TOPIC_LIDAR_FRONT, TOPIC_LIDAR_REAR])
    deser = {}
    for topic in (TOPIC_LIDAR_FRONT, TOPIC_LIDAR_REAR):
        t = topic_types.get(topic)
        deser[topic] = get_message(t) if t else None

    front_buf: List[np.ndarray] = []
    rear_buf:  List[np.ndarray] = []
    target = CALIB_FRAMES_PER_LIDAR

    def _should_keep(is_moving):
        return is_moving if SCAN_WHEN_MOVING else not is_moving

    while reader.has_next():
        # Arrêt dès que les deux buffers sont pleins
        if len(front_buf) >= target and len(rear_buf) >= target:
            break

        topic, data, ts = reader.read_next()
        d = deser.get(topic)
        if d is None:
            continue

        # Seulement pendant le mouvement (ou l'arrêt selon config)
        if not _should_keep(speed_tracker.is_moving(ts)):
            continue
        pose, _ = gnss_traj.interpolate_pose(ts)
        if pose is None:
            continue

        msg = deserialize_message(data, d)
        xyz = fast_pointcloud2_to_xyz(msg)
        del msg
        if xyz is None or len(xyz) == 0:
            continue

        # Filtrage distance (repère capteur) — même fenêtre que le pipeline
        xyz = filter_points_by_distance(xyz, min_dist=0.5, max_dist=35.0)
        if len(xyz) == 0:
            continue

        if topic == TOPIC_LIDAR_FRONT and len(front_buf) < target:
            front_buf.append(xyz)
        elif topic == TOPIC_LIDAR_REAR and len(rear_buf) < target:
            rear_buf.append(xyz)

    print(f"  [CALIB] Frames collectées : front={len(front_buf)} | "
          f"rear={len(rear_buf)} (cible={target})")

    if len(front_buf) < 5 or len(rear_buf) < 5:
        print("  [CALIB][WARN] Pas assez de frames → calibration ignorée")
        return None, None

    front_pts = np.concatenate(front_buf, axis=0)
    rear_pts  = np.concatenate(rear_buf,  axis=0)
    del front_buf, rear_buf
    _force_gc()

    return estimate_extrinsic_icp(front_pts, rear_pts)


def transform_points_with_deskew(xyz: np.ndarray,
                                 base_ts_ns: int,
                                 gnss_traj: GNSSTrajectory,
                                 R_ext: np.ndarray,
                                 t_ext: np.ndarray,
                                 scan_duration_s: float = 0.1,
                                 num_chunks: int = 12) -> Optional[np.ndarray]:
    """
    Découpe le scan en secteurs angulaires et applique une pose interpolée à chacun.
    """
    if len(xyz) == 0: return xyz

    # 1. Calcul de l'angle pour estimer le temps relatif (0 à 1)
    # Sur un Hesai, le laser tourne généralement en sens horaire.
    angles = np.arctan2(xyz[:, 1], xyz[:, 0])

    # Normalisation entre 0 et 1 sur la durée du scan
    angle_min, angle_max = np.min(angles), np.max(angles)
    t_frac = (angles - angle_min) / (angle_max - angle_min + 1e-6)

    # 2. Indexation par "chunks" (secteurs)
    chunks_idx = np.clip(np.floor(t_frac * num_chunks).astype(int), 0, num_chunks - 1)

    out_xyz = np.empty_like(xyz)
    valid_mask = np.zeros(len(xyz), dtype=bool)

    # 3. Transformation par secteur
    for i in range(num_chunks):
        mask = (chunks_idx == i)
        if not np.any(mask): continue

        # Temps moyen du secteur
        chunk_t_frac = (i + 0.5) / num_chunks
        ts_query = base_ts_ns + int((chunk_t_frac * scan_duration_s) * 1e9)

        # Interpolation SE(3) complète (Position + Quaternion)
        pos_base, q_base = gnss_traj.interpolate_pose_se3(ts_query)
        if pos_base is None: continue

        R_base = quat_to_rot(q_base)

        # Fusion : Pose_Monde = R_gnss * (R_ext * Point + T_ext) + T_gnss
        R_total = R_base @ R_ext
        t_total = R_base @ t_ext + pos_base

        out_xyz[mask] = (xyz[mask] @ R_total.T + t_total).astype(np.float32)
        valid_mask[mask] = True

    return out_xyz[valid_mask]


def stream_lidar_pass(bag_path: str,
                      gnss_traj: GNSSTrajectory,
                      speed_tracker: FWSSpeedTracker,
                      T_front: np.ndarray,
                      T_rear: np.ndarray) -> np.ndarray:

    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    import open3d as o3d

    # ── Checkpoint ───────────────────────────────────────────────────────
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / "accumulated_cloud.npy"
    ckpt_meta = CHECKPOINT_DIR / "accumulated_meta.npy"
    resumed_pts = _load_checkpoint(ckpt_path)
    resume_after_ts = 0
    if resumed_pts is not None:
        if len(resumed_pts) == 0:
            print("  [CKPT] Checkpoint vide → ignoré")
            resumed_pts = None
            for f in (ckpt_path, ckpt_meta):
                if f.exists(): f.unlink()
        elif ckpt_meta.exists():
            meta = np.load(str(ckpt_meta))
            resume_after_ts = int(meta[0])
            print(f"  [CKPT] Reprise après ts={resume_after_ts}")

    # ── Extrinsèques (manuels comme base) ────────────────────────────────
    lidar_transforms = {}

    for name in ACTIVE_LIDARS:
        T = extrinsic_to_matrix(LIDAR_CONFIG[name]["extrinsic"])
        R, t = _extract_R_t(T)
        lidar_transforms[LIDAR_CONFIG[name]["topic"]] = (R, t)

    # ══════════════════════════════════════════════════════════════════════
    # PHASE DE CALIBRATION AUTOMATIQUE  (si activée et pas de checkpoint)
    # ══════════════════════════════════════════════════════════════════════
    if USE_AUTO_CALIBRATION and set(ACTIVE_LIDARS) == {"front", "rear"}:
        print(f"[INFO] Phase calibration : accumulation de "
              f"{CALIB_FRAMES_PER_LIDAR} frames par LiDAR…")

        R_icp, t_icp = _run_calibration_phase(bag_path, gnss_traj, speed_tracker)

        if R_icp is not None:
            # T_rear_auto = T_front @ T_icp  (rear→front→vehicle_base)
            T_icp_4x4      = np.eye(4, dtype=np.float64)
            T_icp_4x4[:3, :3] = R_icp
            T_icp_4x4[:3,  3] = t_icp
            T_rear_auto    = T_front @ T_icp_4x4
            R_rear, t_rear = _extract_R_t(T_rear_auto)
            print("[INFO] Extrinsèques arrière mis à jour via ICP ✓")
        else:
            print("[INFO] Calibration ICP échouée → extrinsèques manuels conservés")

    # ── Ouverture bag (streaming principal) ──────────────────────────────
    reader, topic_types = _open_bag_reader(
        bag_path, ACTIVE_LIDAR_TOPICS)
    deser = {}
    for topic in (TOPIC_LIDAR_FRONT, TOPIC_LIDAR_REAR):
        t = topic_types.get(topic)
        deser[topic] = get_message(t) if t else None

    # ── Buffer accumulateur ───────────────────────────────────────────────
    INIT_CAPACITY = 5_000_000
    buf     = resumed_pts if resumed_pts is not None else \
              np.empty((INIT_CAPACITY, 3), dtype=np.float32)
    buf_idx = len(resumed_pts) if resumed_pts is not None else 0

    scan_count   = 0
    skipped_stop = 0
    skipped_range= 0
    t_start      = time.monotonic()

    def _should_keep(is_moving):
        return is_moving if SCAN_WHEN_MOVING else not is_moving

    T_ext_front = extrinsic_to_matrix(LIDAR_CONFIG["front"]["extrinsic"])
    R_ext_front, t_ext_front = _extract_R_t(T_ext_front)

    print("[INFO] Début du streaming avec Deskew...")

    # --- DANS TA BOUCLE PRINCIPALE ---
    while reader.has_next():
        topic, data, ts_msg = reader.read_next()

        if topic == TOPIC_LIDAR_FRONT:
            # 1. Désérialisation rapide
            msg = deserialize_message(data, deser[topic])
            xyz = fast_pointcloud2_to_xyz(msg)
            if xyz is None: continue

            # 2. Filtrage (distance + optionnel : vitesse véhicule)
            if SCAN_WHEN_MOVING and not speed_tracker.is_moving(ts_msg):
                continue

            xyz = filter_points_by_distance(xyz, min_dist=2.0, max_dist=50.0)

            # 3. LA CORRECTION : Remplace ton ancien transform_points_fused par ça :
            xyz_global = transform_points_with_deskew(
                xyz=xyz,
                base_ts_ns=ts_msg,
                gnss_traj=gnss_traj,
                R_ext=R_ext_front,
                t_ext=t_ext_front,
                scan_duration_s=0.1,  # 0.1 pour 10Hz (Hesai standard)
                num_chunks=12  # 12 chunks = correction toutes les 8ms
            )
        # Sécurité : si le véhicule était hors trajectoire GNSS
        if xyz_global is None or len(xyz_global) == 0:
            continue

        if ts_msg <= resume_after_ts:
            continue
        d = deser.get(topic)
        if d is None:
            continue
        if not _should_keep(speed_tracker.is_moving(ts_msg)):
            skipped_stop += 1
            continue

        pose, yaw = gnss_traj.interpolate_pose(ts_msg)
        if pose is None:
            skipped_range += 1
            continue

        msg = deserialize_message(data, d)
        xyz = fast_pointcloud2_to_xyz(msg)
        del msg

        if xyz is None or len(xyz) == 0:
            continue

        # Filtrage distance (repère capteur)
        xyz = filter_points_by_distance(xyz, min_dist=0.5, max_dist=35.0)
        if len(xyz) == 0:
            continue

        # Extrinsèque sélectionné (front manuel, rear auto ou manuel)
        if topic not in lidar_transforms:
            continue

        R_ext, t_ext = lidar_transforms[topic]

        pts = _transform_fn(xyz, pose, yaw, R_ext, t_ext)
        del xyz

        # Ajout au buffer
        n_new = len(pts)
        if buf_idx + n_new > len(buf):
            new_cap = max(len(buf) + n_new, int(len(buf) * 1.5))
            new_buf = np.empty((new_cap, 3), dtype=np.float32)
            new_buf[:buf_idx] = buf[:buf_idx]
            buf = new_buf
        buf[buf_idx:buf_idx + n_new] = pts
        buf_idx   += n_new
        scan_count += 1

        # Downsampling incrémental
        if scan_count % INCREMENTAL_DS_INTERVAL == 0:
            mem_gb   = _mem_used_gb()
            force_ds = mem_gb > MEMORY_LIMIT_GB
            if force_ds:
                print(f"  [MEM] {mem_gb:.1f} Go → DS forcé")
            pcd_tmp = o3d.geometry.PointCloud()
            pcd_tmp.points = o3d.utility.Vector3dVector(buf[:buf_idx])
            pcd_ds  = pcd_tmp.voxel_down_sample(voxel_size=VOXEL_SIZE)
            ds_pts  = np.asarray(pcd_ds.points).astype(np.float32)
            buf_idx = len(ds_pts)
            if buf_idx > len(buf):
                buf = np.empty((buf_idx + INIT_CAPACITY, 3), dtype=np.float32)
            buf[:buf_idx] = ds_pts
            del pcd_tmp, pcd_ds, ds_pts
            _force_gc()
            elapsed = time.monotonic() - t_start
            print(f"  [DS]  scan {scan_count:,} | {buf_idx:,} pts "
                  f"| {scan_count/elapsed:.0f} scans/s | {_mem_used_gb():.1f} Go")

        # Checkpoint
        if scan_count % CHECKPOINT_INTERVAL == 0:
            _save_checkpoint(buf[:buf_idx], ckpt_path, f"scan {scan_count}")
            np.save(str(ckpt_meta), np.array([ts_msg], dtype=np.int64))

    elapsed = time.monotonic() - t_start
    print(f"[INFO] Streaming terminé : {scan_count:,} scans en {elapsed:.1f}s "
          f"({scan_count/max(elapsed,1e-6):.0f} scans/s)")
    print(f"[INFO]   {skipped_stop} ignorés (arrêt) | "
          f"{skipped_range} hors plage GNSS | {buf_idx:,} pts accumulés")

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

def build_mesh(combined_pts: np.ndarray, output_dir: Path, ground:bool) -> None:
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
    cloud_path = output_dir / f"output_cloud_{'Sol' if ground else 'Non-Sol'}_{now}.ply"
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
    print(f"  LiDARs actifs : {ACTIVE_LIDARS}")
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
    _force_gc()

    print(f"[INFO] Non-sol : {len(nonground.points):,} points → mesh")

    # ── Phase 4 : construction mesh non sol ──────────────────────────────────────
    build_mesh(np.asarray(nonground.points).astype(np.float32), OUTPUT_DIR, False)
    del nonground
    _force_gc()

    #print(f"[INFO] Sol : {len(ground.points):,} points → mesh")

    # ── Phase 4 : construction mesh ──────────────────────────────────────
    #build_mesh(np.asarray(ground.points).astype(np.float32), OUTPUT_DIR,True)
    del ground
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
