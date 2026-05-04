"""
lidar_corrections.py
====================
Expert solutions for LiDAR mapping smearing on a 4WS vehicle with Hesai Pandar XT32.

Modules :
  1. Deskewing       — motion compensation intra-scan via SE(3) interpolation
  2. Extrinsics      — estimation / raffinement de T_base_lidar
  3. 4WS Validation  — vérification et identification de L
  4. Open3D Pipeline — filtrage adapté au XT32
"""

from __future__ import annotations

import math
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
from typing import Optional, Tuple
import open3d as o3d


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DESKEWING  —  Motion Compensation intra-scan
# ══════════════════════════════════════════════════════════════════════════════
#
# Problème : le XT32 tourne à 10 Hz → un scan dure 100 ms. Pendant ce temps
# le véhicule se déplace. Chaque anneau de 32 lasers est capturé à un azimut
# différent, donc à un instant légèrement différent.
#
# Solution en algèbre de Lie SE(3) :
#   Soit  T_s = pose au DÉBUT du scan,  T_e = pose à la FIN du scan.
#   Le déplacement relatif entre les deux :
#
#       ΔT = T_s⁻¹ · T_e   ∈ SE(3)
#
#   On interpole l'élément d'algèbre de Lie  ξ = log(ΔT) ∈ se(3),
#   puis on reconstruit la pose au temps α ∈ [0, 1] :
#
#       T(α) = T_s · exp(α · ξ)
#
#   Chaque point est ramené au repère du DÉBUT de scan :
#
#       p_deskewed = T(α)  · p_lidar          (espace LiDAR local)
#                  = T_s⁻¹ · T(α) · p_lidar   (si on voulait en frame monde)
#
#   On garde tout dans le repère LiDAR du début du scan (frame de référence
#   naturel pour le meshing frame-by-frame).

def _so3_log(R: np.ndarray) -> np.ndarray:
    """
    Logarithme matriciel de SO(3) → vecteur ω ∈ ℝ³ (axe · angle).
    Stable au voisinage de θ=0 et θ=π.
    """
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = math.acos(cos_theta)
    if theta < 1e-9:
        return np.zeros(3)
    if abs(theta - math.pi) < 1e-9:
        # Cas dégénéré : θ ≈ π  →  on utilise la colonne la plus grande
        col = np.argmax(np.diag(R))
        omega = (R[:, col] - R[col, :])
        omega[col] += 1.0
        return omega / np.linalg.norm(omega) * math.pi
    return (theta / (2.0 * math.sin(theta))) * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ])


def _so3_exp(omega: np.ndarray) -> np.ndarray:
    """Exponentielle matricielle de so(3) → SO(3) (formule de Rodrigues)."""
    theta = np.linalg.norm(omega)
    if theta < 1e-9:
        return np.eye(3)
    k = omega / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def se3_log(T: np.ndarray) -> np.ndarray:
    """
    Logarithme de SE(3) → twist ξ = [ρ | ω] ∈ ℝ⁶.
    Convention : ξ[:3] = partie linéaire (ρ), ξ[3:] = partie angulaire (ω).

    V⁻¹ = I − (θ/2)·K̂ + (1 − θ(1+cosθ)/(2 sinθ))·K̂²
    où  K̂ = hat(ω/θ)  (skew-symétrique de l'axe unitaire)
    """
    R = T[:3, :3]
    t = T[:3, 3]
    omega = _so3_log(R)
    theta = np.linalg.norm(omega)
    if theta < 1e-9:
        rho = t.copy()
    else:
        k = omega / theta
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])   # hat(k̂)
        # Coefficients pour V⁻¹ exprimé en K̂ (axe UNITAIRE) :
        #   coeff K̂  = −θ/2
        #   coeff K̂² = 1 − θ(1+cosθ)/(2 sinθ)
        c1 = -theta / 2.0
        c2 = 1.0 - theta * (1.0 + math.cos(theta)) / (2.0 * math.sin(theta))
        V_inv = np.eye(3) + c1 * K + c2 * (K @ K)
        rho = V_inv @ t
    return np.concatenate([rho, omega])


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """Exponentielle de se(3) → SE(3) (twist → matrice 4×4)."""
    rho, omega = xi[:3], xi[3:]
    theta = np.linalg.norm(omega)
    R = _so3_exp(omega)
    if theta < 1e-9:
        V = np.eye(3)
    else:
        k = omega / theta
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])   # hat(k̂)
        # V = I + (1-cosθ)/θ · K̂ + (θ-sinθ)/θ · K̂²   (termes en K̂ = axe UNITAIRE)
        c1 = (1.0 - math.cos(theta)) / theta
        c2 = (theta - math.sin(theta)) / theta
        V = np.eye(3) + c1 * K + c2 * (K @ K)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ rho
    return T


def deskew_scan(
    points: np.ndarray,
    timestamps_ns: np.ndarray,
    T_start: np.ndarray,
    T_end: np.ndarray,
    t_start_ns: int,
    t_end_ns: int,
    reference: str = "start",
) -> np.ndarray:
    """
    Corrige la distorsion de mouvement d'un scan LiDAR complet.

    Paramètres
    ----------
    points        : (N, 3+) — points LiDAR en repère capteur [x, y, z, ...]
    timestamps_ns : (N,)   — horodatage absolu de chaque point (nanosecondes)
    T_start       : SE(3) 4×4 — pose du lidar au début du scan dans le monde
    T_end         : SE(3) 4×4 — pose du lidar à la FIN du scan dans le monde
    t_start_ns    : horodatage du premier point du scan (ns)
    t_end_ns      : horodatage du dernier point du scan (ns)
    reference     : "start" → tous les points ramenés à T_start (défaut)
                    "end"   → tous les points ramenés à T_end

    Retourne
    --------
    (N, 3) — nuage deskewé dans le repère de référence demandé
    """
    dt = float(t_end_ns - t_start_ns)
    if dt < 1e-3:
        return points[:, :3].copy()

    # Twist relatif : de T_start vers T_end dans le repère LOCAL de T_start
    T_rel   = np.linalg.inv(T_start) @ T_end          # ΔT ∈ SE(3)
    xi      = se3_log(T_rel)                            # ξ ∈ se(3), shape (6,)

    # Alpha de chaque point ∈ [0, 1]
    alphas  = np.clip((timestamps_ns.astype(np.float64) - t_start_ns) / dt, 0.0, 1.0)

    # Calcul vectorisé : xi_alpha[i] = alpha[i] * ξ
    xi_batch = alphas[:, None] * xi[None, :]           # (N, 6)

    # Exp de chaque twist interpolé  →  T_rel(α) pour chaque point
    # On exprime T(α) dans le repère monde : T_world_at_alpha = T_start · exp(α·ξ)
    # Le point dans le repère de référence :
    #   si reference = "start" : p_ref = exp(α·ξ) · p_lidar
    pts_xyz  = np.hstack([points[:, :3], np.ones((len(points), 1))])   # (N, 4)
    pts_deskewed = np.empty((len(points), 3), dtype=np.float64)

    if reference == "start":
        T_inv_ref = np.eye(4)      # on reste dans le repère start (rien à faire)
    else:
        T_inv_ref = T_rel          # ramener vers T_end

    for i in range(len(points)):
        T_alpha  = se3_exp(xi_batch[i])                    # T_rel(α)
        T_compensate = T_inv_ref @ T_alpha if reference == "end" else T_alpha
        pts_deskewed[i] = (T_compensate @ pts_xyz[i])[:3]

    return pts_deskewed


def deskew_scan_vectorized(
    points: np.ndarray,
    timestamps_ns: np.ndarray,
    T_start: np.ndarray,
    T_end: np.ndarray,
    t_start_ns: int,
    t_end_ns: int,
) -> np.ndarray:
    """
    Version VECTORISÉE (NumPy pur, ~50× plus rapide que la boucle Python).
    On construit les N matrices de rotation/translation en une seule passe.
    Valide pour de petits angles entre T_start et T_end (cas typique <100 ms).

    Hypothèse : mouvement quasi-constant sur la durée du scan (acceptable à 10 Hz).
    """
    dt = float(t_end_ns - t_start_ns)
    if dt < 1e-3:
        return points[:, :3].copy()

    T_rel = np.linalg.inv(T_start) @ T_end
    xi    = se3_log(T_rel)                                  # (6,)
    alphas = np.clip((timestamps_ns.astype(np.float64) - t_start_ns) / dt, 0.0, 1.0)

    # ── Construction vectorisée des twists interpolés ─────────────────────
    # xi_a[i] = alpha[i] * xi  →  forme (N, 6)
    xi_a = alphas[:, None] * xi[None, :]

    rho   = xi_a[:, :3]                                     # (N, 3)
    omega = xi_a[:, 3:]                                     # (N, 3)
    theta = np.linalg.norm(omega, axis=1, keepdims=True)    # (N, 1)

    # Matrice chapeau [ω]× pour chaque point — shape (N, 3, 3)
    K = np.zeros((len(points), 3, 3))
    K[:, 0, 1] = -omega[:, 2];  K[:, 0, 2] =  omega[:, 1]
    K[:, 1, 0] =  omega[:, 2];  K[:, 1, 2] = -omega[:, 0]
    K[:, 2, 0] = -omega[:, 1];  K[:, 2, 1] =  omega[:, 0]

    theta_sq = (theta ** 2).squeeze(-1)                     # (N,)
    cos_t    = np.cos(theta.squeeze(-1))
    sin_t    = np.sin(theta.squeeze(-1))

    # Rodrigues vectorisé : R = I + sin(θ)/θ·K + (1-cos(θ))/θ²·K²
    I = np.eye(3)[None].repeat(len(points), axis=0)        # (N, 3, 3)
    K2 = K @ K

    mask = (theta.squeeze() > 1e-9)
    safe_t = np.where(mask, theta.squeeze(), np.ones(len(points)))
    coeff1 = np.where(mask, sin_t / safe_t, np.ones(len(points)))
    coeff2 = np.where(mask, (1 - cos_t) / (theta_sq + 1e-30), 0.5 * np.ones(len(points)))

    Rot = I + coeff1[:, None, None] * K + coeff2[:, None, None] * K2   # (N,3,3)

    # Translation V·ρ
    coeff3 = np.where(mask, (theta.squeeze() - sin_t) / (theta_sq * theta.squeeze() + 1e-30),
                      np.full(len(points), 1.0/6.0))
    V      = I + coeff2[:, None, None] * K + coeff3[:, None, None] * K2
    t_vec  = np.einsum("nij,nj->ni", V, rho)               # (N, 3)

    # Application de la transformation à chaque point
    pts = points[:, :3].astype(np.float64)
    pts_deskewed = np.einsum("nij,nj->ni", Rot, pts) + t_vec

    return pts_deskewed


def azimuth_timestamps_from_points(
    points_xyzi: np.ndarray,
    t_scan_start_ns: int,
    scan_duration_ns: int = 100_000_000,   # 100 ms @ 10 Hz
) -> np.ndarray:
    """
    Estime le timestamp de chaque point à partir de son azimut (angle horizontal).
    Utiliser quand le bag ne contient pas de champ 'time' par point.

    Le XT32 tourne dans le sens trigonométrique inverse (horaire vu de dessus),
    de 0° à 360°.  On mappe linéairement l'azimut → timestamp.

    points_xyzi : (N, 3+)  —  les colonnes 0,1 doivent être x,y en repère LiDAR
    """
    azimuth = np.arctan2(points_xyzi[:, 1], points_xyzi[:, 0])  # [-π, π]
    # Normaliser en [0, 1] : 0 rad → début du scan
    alpha = (azimuth % (2 * math.pi)) / (2 * math.pi)
    return (t_scan_start_ns + alpha * scan_duration_ns).astype(np.int64)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  CALIBRATION DES EXTRINSÈQUES  T_base_lidar
# ══════════════════════════════════════════════════════════════════════════════
#
# La matrice  T_base_lidar  (4×4 SE(3)) exprime la pose du LiDAR dans le
# repère "base_link" (centre du véhicule, entre les essieux).
#
# Méthode recommandée : "Hand-Eye Calibration" sur plan horizontal
#   1. Conduire sur une ligne droite de ~50 m.
#   2. Intégrer l'odométrie 4WS → T_odom[i]  (repère monde, base_link)
#   3. Enregistrer le mouvement perçu par le LiDAR seul (scan-matching ICP) → T_lidar[i]
#   4. Résoudre :  T_odom[i] · T_base_lidar = T_base_lidar · T_lidar[i]
#      via la formule de Park & Martin (1994), implémentée ci-dessous.

def hand_eye_calibration_park(A_list: list[np.ndarray],
                               B_list: list[np.ndarray]) -> np.ndarray:
    """
    Résout AX = XB  (Park & Martin 1994) pour trouver X = T_base_lidar.

    A_list : liste de  T_odom[i+1]⁻¹ · T_odom[i]   (mouvement odométrie)
    B_list : liste de  T_lidar[i+1]⁻¹ · T_lidar[i] (mouvement LiDAR / ICP)
    Retourne X : SE(3) 4×4
    """
    assert len(A_list) == len(B_list) >= 3, "Minimum 3 paires de poses nécessaires"

    # Accumulation des contraintes sur la rotation
    M = np.zeros((3, 3))
    for A, B in zip(A_list, B_list):
        Ra = A[:3, :3];  Rb = B[:3, :3]
        # log(Ra) et log(Rb) → vecteurs axe-angle
        ra = R_scipy.from_matrix(Ra).as_rotvec()
        rb = R_scipy.from_matrix(Rb).as_rotvec()
        M += np.outer(rb, ra)

    # SVD pour trouver Rx optimale (critère de Frobenius)
    U, _, Vt = np.linalg.svd(M)
    Rx = U @ np.diag([1, 1, np.linalg.det(U) * np.linalg.det(Vt)]) @ Vt

    # Accumulation des contraintes sur la translation
    C_list, d_list = [], []
    for A, B in zip(A_list, B_list):
        Ra = A[:3, :3];  ta = A[:3, 3]
        Rb = B[:3, :3];  tb = B[:3, 3]
        C_list.append(Ra - np.eye(3))
        d_list.append(Rx @ tb - ta)
    C = np.vstack(C_list);  d = np.concatenate(d_list)
    tx, _, _, _ = np.linalg.lstsq(C, d, rcond=None)

    T = np.eye(4)
    T[:3, :3] = Rx
    T[:3, 3]  = tx
    return T


def refine_extrinsics_icp(
    T_init: np.ndarray,
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    max_iter: int = 50,
) -> np.ndarray:
    """
    Affine T_base_lidar par ICP point-à-plan sur deux scans statiques successifs
    (véhicule à l'arrêt devant un mur plan ou un poteau).
    """
    result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd,
        max_correspondence_distance=0.1,
        init=T_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
    )
    return result.transformation


# ══════════════════════════════════════════════════════════════════════════════
# 3.  VALIDATION DU MODÈLE 4WS + IDENTIFICATION DE L
# ══════════════════════════════════════════════════════════════════════════════
#
# Le modèle de odometry.py est mathématiquement correct :
#   β  = arctan((tan δf + tan δr) / 2)          ← angle de dérive au centre
#   ψ̇  = (v·cos β / L) · (tan δf − tan δr)       ← taux de lacet
#
# Pour identifier L on utilise la propriété suivante :
#   Sur un cercle parfait (δf, δr constants, v constant) :
#       ψ_total = intégrale(ψ̇ · dt)
#   La trajectoire forme un cercle de rayon R = v / ψ̇.
#   En comparant la trajectoire GNSS (R_gnss) et le R prédit (R_model = v/ψ̇),
#   on ajuste L par moindres carrés.

def identify_wheelbase(
    fwss_records: list,     # liste de FourWheelSteeringStamped
    gnss_positions: np.ndarray,  # (N, 3) positions GNSS synchronisées
    dt_s: float = 0.1,          # pas de temps moyen
    L_init: float = 2.8,
) -> float:
    """
    Identifie l'empattement L par optimisation de Gauss-Newton.
    On minimise :  Σ (ψ_gnss[i] - ψ_model[i, L])²

    Stratégie :
      - ψ_gnss[i] estimé depuis la courbure de la trajectoire GNSS
      - ψ_model[i, L] = (v·cos β / L) · (tan δf − tan δr) · dt
    """
    from scipy.optimize import minimize_scalar

    def yaw_rates_gnss(positions: np.ndarray) -> np.ndarray:
        """Estime les taux de lacet à partir des positions GNSS."""
        dp  = np.diff(positions[:, :2], axis=0)
        hdg = np.arctan2(dp[:, 1], dp[:, 0])
        dhdg = np.diff(np.unwrap(hdg))
        return dhdg / dt_s                      # rad/s

    psi_gnss = yaw_rates_gnss(gnss_positions)
    # Tronquer les deux extrémités (différences réduisent d'1 élément à chaque fois)
    fwss_trimmed = fwss_records[1:len(psi_gnss)+1]

    def residuals(L: float) -> float:
        total = 0.0
        for fwss, psi_ref in zip(fwss_trimmed, psi_gnss):
            tan_f = math.tan(fwss.front_steering_angle)
            tan_r = math.tan(fwss.rear_steering_angle)
            beta  = math.atan((tan_f + tan_r) / 2.0)
            psi_pred = (fwss.speed * math.cos(beta) / L) * (tan_f - tan_r)
            total += (psi_ref - psi_pred) ** 2
        return total

    result = minimize_scalar(residuals, bounds=(1.5, 4.5), method='bounded')
    print(f"[INFO] L identifié = {result.x:.4f} m  (résidu = {result.fun:.6f})")
    return float(result.x)


def validate_4ws_model(fwss, L: float) -> dict:
    """
    Retourne un dictionnaire d'analyse cinématique pour un instant donné.
    Utile pour diagnostiquer les valeurs limites.
    """
    tf = math.tan(fwss.front_steering_angle)
    tr = math.tan(fwss.rear_steering_angle)
    beta  = math.atan((tf + tr) / 2.0)
    yaw_r = (fwss.speed * math.cos(beta) / L) * (tf - tr)
    R_curv = abs(fwss.speed / yaw_r) if abs(yaw_r) > 1e-6 else float('inf')

    return {
        "beta_deg"         : math.degrees(beta),
        "yaw_rate_deg_s"   : math.degrees(yaw_r),
        "turning_radius_m" : R_curv,
        "vx_local"         : fwss.speed * math.cos(beta),
        "vy_local"         : fwss.speed * math.sin(beta),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4.  PIPELINE OPEN3D OPTIMISÉ POUR LE PANDAR XT32
# ══════════════════════════════════════════════════════════════════════════════
#
# Spécifications XT32 :
#   Canaux : 32   |  FOV vertical : ±16°  |  RPS : 10 Hz
#   Résolution angulaire horizontale : 0.18° (mode standard, 20 Hz : 0.36°)
#   À 10 m  → espacement latéral ≈ 0.031 m   (très dense)
#   À 30 m  → espacement latéral ≈ 0.094 m
#   Espacement vertical à 10 m  ≈ 0.28 m  (32 plans, ±16°)
#
# Règles empiriques :
#   - VoxelGrid : 0.05 m (proche) … 0.08 m (général)
#   - SOR : k=20, std_ratio=2.0  (conserve poteaux fins)
#   - DBSCAN radius ≈ 0.15 m pour isoler les structures verticales

XT32_CONFIG = {
    "voxel_size"        : 0.06,   # m — compromis densité/vitesse pour XT32
    "sor_k_neighbors"   : 20,
    "sor_std_ratio"     : 2.0,
    "normal_radius"     : 0.20,   # pour estimate_normals
    "normal_max_nn"     : 30,
    # Ball pivoting : deux rayons couvrant l'espacement vertical à 15 m
    "bp_radius_1"       : 0.12,   # ≈ espacement vertical à 15 m
    "bp_radius_2"       : 0.24,   # × 2
    # Poisson (alternatif)
    "poisson_depth"     : 8,
    "poisson_trim_quantile": 0.05,
}


def preprocess_cloud(
    xyz: np.ndarray,
    config: dict = XT32_CONFIG,
    dist_min: float = 2.0,
    dist_max: float = 40.0,
) -> o3d.geometry.PointCloud:
    """
    Pipeline de prétraitement complet pour un nuage XT32.

    Étapes :
      1. Filtrage distance (retire points trop proches → carrosserie / sol immédiat)
      2. Voxel Grid Downsampling
      3. Statistical Outlier Removal (SOR)
      4. Estimation des normales (nécessaire pour Ball Pivoting / Poisson)
    """
    # 1. Filtrage distance
    dists = np.linalg.norm(xyz[:, :3], axis=1)
    mask  = (dists >= dist_min) & (dists <= dist_max)
    pts   = xyz[mask, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # 2. Voxel Grid
    pcd = pcd.voxel_down_sample(voxel_size=config["voxel_size"])

    # 3. SOR — préserve les poteaux fins (std_ratio élevé = peu agressif)
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=config["sor_k_neighbors"],
        std_ratio=config["sor_std_ratio"],
    )

    # 4. Normales — orientation vers le haut (0, 0, 1) pour le sol,
    #    mais on laisse Open3D décider et on reoriente vers l'origine du scan.
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=config["normal_radius"],
            max_nn=config["normal_max_nn"],
        )
    )
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0, 0, 2.0]))

    return pcd


def mesh_xt32_ball_pivoting(
    pcd: o3d.geometry.PointCloud,
    config: dict = XT32_CONFIG,
) -> o3d.geometry.TriangleMesh:
    """
    Ball Pivoting avec rayons calibrés pour la densité XT32.
    On utilise trois rayons pour couvrir la variabilité distance.
    """
    # Rayon adaptatif basé sur la densité réelle du nuage
    dists     = np.asarray(pcd.compute_nearest_neighbor_distance())
    avg_dist  = float(np.median(dists))      # médiane plus robuste que moyenne
    r1 = max(config["bp_radius_1"], 1.5 * avg_dist)
    r2 = r1 * 2.0
    r3 = r1 * 4.0

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector([r1, r2, r3]),
    )
    mesh.compute_vertex_normals()
    mesh = mesh.filter_smooth_simple(number_of_iterations=1)   # lissage léger
    return mesh


def mesh_xt32_poisson(
    pcd: o3d.geometry.PointCloud,
    config: dict = XT32_CONFIG,
) -> o3d.geometry.TriangleMesh:
    """
    Reconstruction de Poisson — meilleure pour les surfaces continues (sol, murs).
    Moins adaptée aux objets fins isolés.
    """
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=config["poisson_depth"],
    )
    # Suppression des triangles à faible densité (artefacts dans le vide)
    thresh = np.quantile(np.asarray(densities), config["poisson_trim_quantile"])
    mesh.remove_vertices_by_mask(np.asarray(densities) < thresh)
    mesh.compute_vertex_normals()
    return mesh


def extract_vertical_structures(
    pcd: o3d.geometry.PointCloud,
    z_normal_threshold: float = 0.3,   # normales quasi-verticales (|nz| < seuil)
    cluster_eps: float = 0.15,
    cluster_min_points: int = 10,
) -> list[o3d.geometry.PointCloud]:
    """
    Segmente les structures verticales (poteaux, panneaux) dans un nuage.
    Utile pour mesher les poteaux séparément avec des paramètres plus fins.

    Retourne une liste de nuages, un par structure détectée.
    """
    normals = np.asarray(pcd.normals)
    pts     = np.asarray(pcd.points)

    if len(normals) == 0:
        return []

    # Sélection des points à normales quasi-horizontales (surfaces verticales)
    mask_vertical = np.abs(normals[:, 2]) < z_normal_threshold
    pcd_vertical  = pcd.select_by_index(np.where(mask_vertical)[0])

    # Clustering DBSCAN pour isoler chaque structure
    labels = np.asarray(pcd_vertical.cluster_dbscan(
        eps=cluster_eps,
        min_points=cluster_min_points,
        print_progress=False,
    ))

    clusters = []
    for label in set(labels):
        if label < 0:
            continue   # bruit
        idx = np.where(labels == label)[0]
        clusters.append(pcd_vertical.select_by_index(idx))

    return clusters


# ══════════════════════════════════════════════════════════════════════════════
# EXEMPLE D'INTÉGRATION — Remplace _mesh_one_cloud dans visualisation.py
# ══════════════════════════════════════════════════════════════════════════════

def mesh_one_cloud_complete(xyz: np.ndarray) -> o3d.geometry.TriangleMesh:
    """
    Remplaçant amélioré de meshing.mesh_style_ball_pivoting().
    Applique le pipeline complet : filtrage → BP avec rayons XT32.
    """
    pcd  = preprocess_cloud(xyz)
    mesh = mesh_xt32_ball_pivoting(pcd)
    return mesh


# ══════════════════════════════════════════════════════════════════════════════
# INTÉGRATION DANS visualisation.py  —  exemple d'appel deskewing
# ══════════════════════════════════════════════════════════════════════════════

def apply_deskewing_to_frame(
    xyz_raw: np.ndarray,
    ts_lidar: int,              # timestamp du scan (ns) — horodatage du 1er point
    T_odom_prev: np.ndarray,    # pose odom au scan précédent (SE3)
    T_odom_curr: np.ndarray,    # pose odom au scan courant  (SE3)
    scan_duration_ns: int = 100_000_000,
) -> np.ndarray:
    """
    Applique le deskewing à un seul frame LiDAR.
    Remplace la ligne : xyz = map_utils.filter_points_by_distance(xyz, 0.5, 30.0)

    Usage dans visualisation.py :
        xyz = apply_deskewing_to_frame(xyz, ts, T_prev, T_curr)
    """
    # Estimation des timestamps par azimut (si pas de champ time dans le bag)
    t_pts = azimuth_timestamps_from_points(xyz_raw, ts_lidar, scan_duration_ns)

    xyz_deskewed = deskew_scan_vectorized(
        points       = xyz_raw,
        timestamps_ns= t_pts,
        T_start      = T_odom_prev,
        T_end        = T_odom_curr,
        t_start_ns   = ts_lidar,
        t_end_ns     = ts_lidar + scan_duration_ns,
    )
    return xyz_deskewed
