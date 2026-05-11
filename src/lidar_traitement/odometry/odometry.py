from lidar_traitement.optimisation.utils import Odometry, FourWheelSteeringStamped, GNSSTrajectory
import numpy as np
from tqdm import tqdm
from typing import Optional
import copy

import math


# ═════════════════════════════════════════════════════════════════════════════
# CINÉMATIQUE 4WS  →  Objet Odometry
# ═════════════════════════════════════════════════════════════════════════════

def FWSS_by_Odometry(fwss, delta_t, pred, current_gnss_z=None, current_gnss_yaw=None, L=2.8):
    """
    Calcule l'odométrie pour un véhicule à 4 roues directrices (4WS).
    L : Empattement (Wheelbase) -> 2.8m pour EZ10 Gen1/Gen2.
    """
    # Conversion du temps en secondes
    delta_s = (delta_t[1] - delta_t[0]) / 1_000_000_000.0
    if delta_s <= 0:
        return pred

    odom = copy.deepcopy(pred)

    # Lecture des angles de braquage
    tan_f = math.tan(fwss.front_steering_angle)
    tan_r = math.tan(fwss.rear_steering_angle)

    # 1. Calcul de l'angle de dérive (Slip angle beta) au centre du véhicule
    # C'est l'angle entre l'axe du châssis et le vecteur vitesse réel
    beta = math.atan((tan_f + tan_r) / 2.0)

    # 2. Vitesse de lacet (Yaw rate)
    # On utilise la vitesse des roues projetée sur le vecteur de déplacement du centre
    yaw_rate = (fwss.speed * math.cos(beta) / L) * (tan_f - tan_r)
    odom.twist.angular.z = yaw_rate

    # 3. Mise à jour du Cap (Yaw)
    odom.current_yaw += yaw_rate * delta_s

    # Fusion optionnelle avec le GNSS (Idéalement utiliser un lissage/filtre ici)
    if current_gnss_yaw is not None:
        print("no data")
        diff = current_gnss_yaw - odom.current_yaw

        # Normalisation de l'angle entre -pi et pi (très important !)
        diff = (diff + math.pi) % (2 * math.pi) - math.pi

        # On ne corrige que 1% de l'erreur à chaque frame
        # Cela permet de garder la fluidité locale (poteau net)
        # tout en recalant la carte sur le long terme (GNSS).
        gain = 0.01
        odom.current_yaw += gain * diff

        # 4. Intégration de la position dans le repère GLOBAL
    # On déplace le centre du robot selon l'angle (Cap actuel + angle de dérive beta)
    cos_yaw_beta = math.cos(odom.current_yaw + beta)
    sin_yaw_beta = math.sin(odom.current_yaw + beta)

    vx_global = fwss.speed * cos_yaw_beta
    vy_global = fwss.speed * sin_yaw_beta

    odom.pose.position.x += vx_global * delta_s
    odom.pose.position.y += vy_global * delta_s

    if current_gnss_z is not None:
        odom.pose.position.z = current_gnss_z

    # 5. Mise à jour des vitesses locales (Convention ROS)
    odom.twist.linear.x = fwss.speed * math.cos(beta)
    odom.twist.linear.y = fwss.speed * math.sin(beta)

    # 6. Conversion Yaw -> Quaternion pour la pose
    half_yaw = odom.current_yaw / 2.0
    odom.pose.orientation.z = math.sin(half_yaw)
    odom.pose.orientation.w = math.cos(half_yaw)

    return odom


# ═════════════════════════════════════════════════════════════════════════════
# MATRICE DE TRANSFORMATION SE(3) 4×4
# ═════════════════════════════════════════════════════════════════════════════

def get_transformation_matrix_from_odom(odom: Odometry) -> np.ndarray:
    """
    Génère la matrice SE(3) 4x4 transformant les points du LiDAR vers le repère MONDE.
    """
    T_world_base = np.eye(4, dtype=np.float64)

    # Extraction du cap depuis le quaternion (rotation autour de Z uniquement)
    z = odom.pose.orientation.z
    w = odom.pose.orientation.w
    yaw = 2.0 * math.atan2(z, w)

    cp, sp = np.cos(yaw), np.sin(yaw)

    # Rotation Z
    T_world_base[0, 0], T_world_base[0, 1] = cp, -sp
    T_world_base[1, 0], T_world_base[1, 1] = sp, cp

    # Translation (Position du robot)
    T_world_base[0, 3] = odom.pose.position.x
    T_world_base[1, 3] = odom.pose.position.y
    T_world_base[2, 3] = odom.pose.position.z

    # --- OFFSET LIDAR (Extrinsèques) ---
    # On définit où est le LiDAR par rapport au centre du robot
    T_base_lidar = np.eye(4)

    T_base_lidar[:3, 3] = [1.5, 0.0, 2.0]

    # La transformation finale est la combinaison des deux :
    # Points_Monde = T_world_base * T_base_lidar * Points_LiDAR
    return T_world_base @ T_base_lidar

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
                z_curr = pos_curr[2] if pos_curr is not None else z_pred
                # yaw_gnss = math.atan2(2.0 * (q_gnss[0] * q_gnss[3] + q_gnss[1] * q_gnss[2]),
                #                       1.0 - 2.0 * (q_gnss[2] ** 2 + q_gnss[3] ** 2))

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

        if fwss.speed == 0:
            # Interpolation GNSS pour recaler l'altitude et le cap
            pos_curr, q_gnss = gnss_helper.interpolate_pose_se3(ts)
            z_curr = pos_curr[2] if pos_curr is not None else z_pred

            yaw_gnss = None
            if q_gnss is not None:
                z_curr = pos_curr[2] if pos_curr is not None else z_pred
                # yaw_gnss = math.atan2(2.0 * (q_gnss[0] * q_gnss[3] + q_gnss[1] * q_gnss[2]),
                #                       1.0 - 2.0 * (q_gnss[2] ** 2 + q_gnss[3] ** 2))

            # Un seul calcul d'odométrie par itération
            data_out = FWSS_by_Odometry(fwss, (time_pred, ts), odom_pred,
                                        current_gnss_z=z_curr, current_gnss_yaw=yaw_gnss)

            # Génération de la matrice avec le bras de levier
            if abs(fwss.speed) > 0.01:
                T_list.append(T_list[-1])
            else:
                T_list.append(get_transformation_matrix_from_odom(data_out))
            timestamps_out.append(ts)
            odom_out.append(data_out)

            odom_pred, z_pred = data_out, z_curr

        time_pred = ts

    return timestamps_out, odom_out, np.array(T_list)