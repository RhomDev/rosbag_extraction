import numpy as np
from typing import Optional
from scipy.spatial import KDTree

# ═════════════════════════════════════════════════════════════════════════════
# DÉSÉRIALISATION PointCloud2 → numpy  (100 % vectorisée)
# ═════════════════════════════════════════════════════════════════════════════

def fast_pointcloud2_to_xyz(msg) -> Optional[np.ndarray]:
    """
    Extraction ultra-rapide des champs XYZ.
    Utilise les offsets du dtype NumPy pour éviter les slices et les copies manuelles.
    """
    n_points = msg.width * msg.height
    if n_points == 0:
        return None

    # 1. On crée un dictionnaire d'offsets pour extraire uniquement X, Y, et Z
    # On mappe les types ROS (datatype) vers les types NumPy
    ros_to_numpy_types = {
        7: '<f4',  # FLOAT32
        8: '<f8',  # FLOAT64
    }

    try:
        # On récupère les types et offsets réels du message
        fields = {f.name: f for f in msg.fields if f.name in ('x', 'y', 'z')}

        # Configuration du dtype structuré avec offsets
        # Cela permet de sauter les octets inutiles (RGB, Intensity, etc.) directement
        dt = np.dtype({
            'names': ['x', 'y', 'z'],
            'formats': [ros_to_numpy_types.get(fields[f].datatype, '<f4') for f in ('x', 'y', 'z')],
            'offsets': [fields[f].offset for f in ('x', 'y', 'z')],
            'itemsize': msg.point_step  # Crucial : définit la taille totale d'un point
        })

        # 2. Création de la vue (Zero-copy si possible)
        # Utilisation de memoryview pour éviter toute copie de buffer ROS
        data_view = np.frombuffer(msg.data, dtype=dt, count=n_points)

        # 3. Conversion en tableau 2D standard (N, 3)
        # astype(np.float32) crée une copie propre, indispensable pour les calculs futurs
        xyz = np.stack([data_view['x'], data_view['y'], data_view['z']], axis=1).astype(np.float32)

        # 4. Filtrage des NaN (Optionnel selon ton besoin)
        mask = np.isfinite(xyz).all(axis=1)
        return xyz[mask] if not mask.all() else xyz

    except (KeyError, ValueError):
        # Fallback ou gestion d'erreur si champs manquants
        return None


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
def filter_points_by_distance(xyz: np.ndarray, min_dist: float, max_dist: float) -> np.ndarray:
    # Ton code est déjà optimal ici.
    # On peut juste ajouter une vérification pour éviter de filtrer si les bornes sont infinies.
    sq = np.einsum('ij,ij->i', xyz, xyz)
    mask = (sq >= min_dist ** 2) & (sq <= max_dist ** 2)
    return xyz[mask]


def remove_noise(xyz: np.ndarray, method: str = "statistical", **kwargs) -> np.ndarray:
    try:
        import open3d as o3d
        # Note: Open3D préfère le float64.
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)  # Copie probable ici si xyz est float32

        if method == "statistical":
            pcd_clean, _ = pcd.remove_statistical_outlier(
                nb_neighbors=kwargs.get("nb_neighbors", 20),
                std_ratio=kwargs.get("std_ratio", 2.0)
            )
        elif method == "radius":
            pcd_clean, _ = pcd.remove_radius_outlier(
                nb_points=kwargs.get("nb_points", 16),
                radius=kwargs.get("radius", 0.5)
            )
        else:
            raise ValueError(f"Méthode {method} inconnue")

        return np.asarray(pcd_clean.points, dtype=np.float32)

    except ImportError:
        return _scipy_noise_fallback(xyz, **kwargs)


def _scipy_noise_fallback(xyz: np.ndarray, **kwargs) -> np.ndarray:
    """
    Fallback plus intelligent utilisant un KD-Tree pour une analyse LOCALE.
    Reste en O(N log N), bien meilleur que le centre global.
    """
    k = kwargs.get("nb_neighbors", 20)
    std_ratio = kwargs.get("std_ratio", 2.0)

    tree = KDTree(xyz)
    # On calcule la distance au k-ième voisin pour chaque point
    dists, _ = tree.query(xyz, k=k)
    avg_dists = dists.mean(axis=1)

    # Seuil statistique local
    mean_dist = np.mean(avg_dists)
    std_dist = np.std(avg_dists)

    mask = avg_dists < (mean_dist + std_ratio * std_dist)
    return xyz[mask]