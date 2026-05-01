from lidar_traitement.optimisation.utils import Odometry, FourWheelSteeringStamped, GNSSTrajectory
import numpy as np
from tqdm import tqdm
from typing import Optional

import math


# ═════════════════════════════════════════════════════════════════════════════
# CINÉMATIQUE 4WS  →  Objet Odometry
# ═════════════════════════════════════════════════════════════════════════════

def FWSS_by_Odometry(fwss, delta_t, pred, current_gnss_z=None, current_gnss_yaw=None, L=2.0):
    delta_s = (delta_t[1] - delta_t[0]) / 1_000_000_000.0
    if delta_s <= 0:
        return pred

    odom = pred
    tan_f = math.tan(fwss.front_steering_angle)  # math.tan ~2× plus rapide que np.tan scalaire
    tan_r = math.tan(fwss.rear_steering_angle)
    beta  = math.atan((tan_f + tan_r) / 2.0)

    yaw_rate = (fwss.speed * math.cos(beta) / L) * (tan_f - tan_r)
    odom.twist.angular.z = yaw_rate

    odom.current_yaw += yaw_rate * delta_s
    if current_gnss_yaw is not None:
        odom.current_yaw = current_gnss_yaw

    cos_yaw_beta = math.cos(odom.current_yaw + beta)
    sin_yaw_beta = math.sin(odom.current_yaw + beta)
    vx_global = fwss.speed * cos_yaw_beta
    vy_global = fwss.speed * sin_yaw_beta

    odom.twist.linear.x = vx_global
    odom.twist.linear.y = vy_global
    odom.pose.position.x += vx_global * delta_s
    odom.pose.position.y += vy_global * delta_s
    if current_gnss_z is not None:
        odom.pose.position.z = current_gnss_z

    half_yaw = odom.current_yaw / 2.0
    odom.pose.orientation.x = 0.0
    odom.pose.orientation.y = 0.0
    odom.pose.orientation.z = math.sin(half_yaw)
    odom.pose.orientation.w = math.cos(half_yaw)
    odom.deplacement(delta_s)
    return odom


# ═════════════════════════════════════════════════════════════════════════════
# MATRICE DE TRANSFORMATION SE(3) 4×4
# ═════════════════════════════════════════════════════════════════════════════

def get_transformation_matrix_from_odom(odom: Odometry) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)

    # Utilisation directe du Yaw calculé par ton odométrie
    yaw = odom.pose.orientation.z
    cp, sp = np.cos(yaw), np.sin(yaw)

    # Rotation 2D (Z) + Position
    T[:3, :3] = [
        [cp, -sp, 0],
        [sp, cp, 0],
        [0, 0, 1]
    ]
    T[0, 3] = odom.pose.position.x
    T[1, 3] = odom.pose.position.y
    # T[2, 3] = -odom.pose.position.z

    # --- AJOUT DU BRAS DE LEVIER (Extrinsèques) ---
    # Exemple : LiDAR est à 1.5m à l'avant et 2.0m de haut
    T_lidar_to_base = np.eye(4)
    T_lidar_to_base[:3, 3] = [1.5, 0.0, 2.0]

    # La pose réelle des points est : Pose_Robot @ Pose_LiDAR_dans_Robot
    return T @ T_lidar_to_base

def get_transformation_matrix(
    fwss: FourWheelSteeringStamped,
    delta_t: float,
    L: float = 1.0,
    dz: float = 0.0,
    q_gnss: Optional[np.ndarray] = None   # [w, x, y, z] depuis interpolate_pose_se3
) -> np.ndarray:

    delta_t_sec = delta_t / 1_000_000_000.0
    if delta_t_sec <= 0.0:
        return np.eye(4, dtype=np.float64)

    # ── Cinématique 4WS ────────────────────────────────────────────────────
    tan_f = np.tan(fwss.front_steering_angle)
    tan_r = np.tan(fwss.rear_steering_angle)
    beta  = np.arctan((tan_f + tan_r) / 2.0)

    vx       = fwss.speed * np.cos(beta)
    vy       = fwss.speed * np.sin(beta)
    yaw_rate = (vx / L) * (tan_f - tan_r)
    d_psi    = yaw_rate * delta_t_sec

    # ── Intégration exacte sur arc de cercle ───────────────────────────────
    if abs(d_psi) > 1e-7:
        s, c = np.sin(d_psi), np.cos(d_psi)
        inv  = 1.0 / yaw_rate
        dx   = ( vx * s          + vy * (c - 1.0)) * inv
        dy   = (-vx * (c - 1.0)  + vy * s        ) * inv
    else:
        # Développement de Taylor O(2) — évite la division par zéro
        half = d_psi * delta_t_sec / 2.0
        dx   = vx * delta_t_sec - vy * half
        dy   = vy * delta_t_sec + vx * half

    # ── Matrice de rotation ────────────────────────────────────────────────
    if q_gnss is not None:
        w, x, y, z = q_gnss
        R = np.array([
            [1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y)   ],
            [    2*(x*y + w*z), 1 - 2*(x*x + z*z),   2*(y*z - w*x)   ],
            [    2*(x*z - w*y),   2*(y*z + w*x),   1 - 2*(x*x + y*y) ],
        ], dtype=np.float64)
    else:
        # Fallback : pitch estimé depuis la pente dz/ds, roll = 0
        ds_h  = np.sqrt(dx**2 + dy**2)
        pitch = np.arctan2(dz, ds_h) if ds_h > 1e-6 else 0.0
        cp, sp = np.cos(d_psi), np.sin(d_psi)
        ct, st = np.cos(pitch), np.sin(pitch)
        R = np.array([
            [ cp*ct,  -sp,   cp*st ],
            [ sp*ct,   cp,   sp*st ],
            [-st,      0.0,  ct    ],
        ], dtype=np.float64)

    # ── Assemblage SE(3) ───────────────────────────────────────────────────
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = [dx, dy, dz]
    return T


# ═════════════════════════════════════════════════════════════════════════════
# EXTRACTION ODOMÉTRIE — véhicule EN MOUVEMENT  (à utiliser pour le mapping)
# ═════════════════════════════════════════════════════════════════════════════

def extraire_FWSS_moving(data_fwss, data_gnss):
    timestamps_out, odom_out, T_list = [], [], []
    gnss_helper = GNSSTrajectory(data_gnss)
    odom_pred = Odometry()
    time_pred = data_fwss[0][0]

    # Initialisation Z
    pos_init, _ = gnss_helper.interpolate_pose(time_pred)
    z_pred = pos_init[2] if pos_init is not None else 0.0

    for ts, msg in tqdm(data_fwss, desc="Odométrie"):
        fwss = FourWheelSteeringStamped()
        fwss._conver_MSG(msg)

        if fwss.speed != 0:
            # Interpolation GNSS pour recaler l'altitude et le cap
            pos_curr, q_gnss = gnss_helper.interpolate_pose_se3(ts)
            z_curr = pos_curr[2] if pos_curr is not None else z_pred

            yaw_gnss = None
            if q_gnss is not None:
                yaw_gnss = math.atan2(2.0 * (q_gnss[0] * q_gnss[3] + q_gnss[1] * q_gnss[2]),
                                      1.0 - 2.0 * (q_gnss[2] ** 2 + q_gnss[3] ** 2))

            # Un seul calcul d'odométrie par itération
            data_out = FWSS_by_Odometry(fwss, (time_pred, ts), odom_pred,
                                        current_gnss_z=z_curr, current_gnss_yaw=yaw_gnss)

            # Génération de la matrice avec le bras de levier
            if abs(fwss.speed) < 0.01:
                T_list.append(T_list[-1])
            else:
                T_list.append(get_transformation_matrix_from_odom(data_out))
            timestamps_out.append(ts)
            odom_out.append(data_out)

            odom_pred, z_pred = data_out, z_curr

        time_pred = ts

    return timestamps_out, odom_out, np.array(T_list)

# ═════════════════════════════════════════════════════════════════════════════
# EXTRACTION ODOMÉTRIE — véhicule À L'ARRÊT  (diagnostic / debug uniquement)
# ═════════════════════════════════════════════════════════════════════════════

def extraire_FWSS_move_null(data_fwss, data_gnss):
    """
    Traite uniquement les trames où speed == 0.
    Usage : analyse de la dérive statique, debug — PAS pour le mapping.
    """
    t, odom = [], []
    T_accumulee = [np.eye(4)]

    gnss_helper = GNSSTrajectory(data_gnss)
    time_pred   = data_fwss[0][0]
    odom_pred   = Odometry()

    pos_init, _ = gnss_helper.interpolate_pose(time_pred)
    z_pred = pos_init[2] if pos_init is not None else 0.0

    for ts, msg in tqdm(data_fwss, desc="Odométrie 4WS (arrêt)"):
        fwss = FourWheelSteeringStamped()
        fwss._conver_MSG(msg)

        if fwss.speed == 0 and time_pred != 0:
            delta_t = ts - time_pred

            pos_curr, q_gnss = gnss_helper.interpolate_pose_se3(ts)
            z_curr = pos_curr[2] if pos_curr is not None else z_pred
            dz     = z_curr - z_pred

            yaw_gnss = None
            if q_gnss is not None:
                import math
                w, x, y, z_q = q_gnss
                yaw_gnss = math.atan2(2.0*(w*z_q + x*y), 1.0 - 2.0*(y*y + z_q*z_q))

            data_out = FWSS_by_Odometry(
                fwss, (time_pred, ts), odom_pred,
                current_gnss_z=z_curr,
                current_gnss_yaw=yaw_gnss
            )
            trans_relative = get_transformation_matrix(
                fwss, delta_t, dz=dz, q_gnss=q_gnss
            )

            t.append(ts)
            odom.append(data_out)
            T_accumulee.append(T_accumulee[-1] @ trans_relative)

            odom_pred = data_out
            z_pred    = z_curr

        time_pred = ts

    return t, odom, np.array(T_accumulee)