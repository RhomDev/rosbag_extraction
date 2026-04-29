import gc
import os

from lidar_traitement.mapping import utils as map_utils
from lidar_traitement.odometry import odometry
from lidar_traitement.extraction import extrait_rosbag
from lidar_traitement.optimisation.utils import GNSSTrajectory, Odometry, PoseWithCovariance
from lidar_traitement.optimisation import memoire_gpu as utils

import open3d as o3d
import numpy as np
from tqdm import tqdm
from pathlib import Path

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

BAG = '/home/rhomdev/Documents/stage/bag_files/ez10_gen1_sensors_2026-04-02-16-45-39'

TOPIC_LOADING = [
    "/ez10_gen1/received_raw_four_wheel_steering",
    "/ez10_gen1/gnss_main/pose_lpz",
    "/ez10_gen1/hesai_front/cloud",
]

PATH_CHECKPOINT = Path("/home/rhomdev/Documents/stage/rosbag_extraction/out/._checkpoint")
MAX_RAM_ALLOWED = 11.0
PATH_MAP_CHUNKS = PATH_CHECKPOINT / "map_chunks"
os.makedirs(PATH_MAP_CHUNKS, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ═════════════════════════════════════════════════════════════════════════════

def visualiser_nuage(xyz: np.ndarray):
    """Affiche un nuage colorisé par la hauteur Z."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[:, :3])
    pcd.estimate_normals()

    z_values = xyz[:, 2]
    z_range  = z_values.max() - z_values.min()
    colors   = np.zeros((len(xyz), 3), dtype=np.float32)
    if z_range > 1e-6:
        colors[:, 0] = (z_values - z_values.min()) / z_range
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd], window_name="Visualisation LiDAR Front")


def assembler_map_finale(directory: Path) -> o3d.geometry.PointCloud:
    """Fusionne tous les chunks PCD en réduisant la densité à la volée."""
    final_pcd = o3d.geometry.PointCloud()
    files = sorted(f for f in os.listdir(directory) if f.endswith('.pcd'))

    for f in tqdm(files, desc="Assemblage final"):
        temp_pcd   = o3d.io.read_point_cloud(str(directory / f))
        final_pcd += temp_pcd
        final_pcd  = final_pcd.voxel_down_sample(voxel_size=0.05)

    return final_pcd


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("==== LANCEMENT DE L'ÉTUDE MAPPING ====")

    reader, topics_type = extrait_rosbag._open_bag_reader(BAG, TOPIC_LOADING)
    data_reader = extrait_rosbag._open_bag_extrait(reader, topics_type, TOPIC_LOADING)
    del reader, topics_type
    utils._force_gc()

    print("==== Initialisation du système d'optimisation ====")
    utils.init()

    # ── Étape 1 : Odométrie ────────────────────────────────────────────────
    print("==== Récupération des données FWSS ====")

    # BUG 1 CORRIGÉ : on appelle extraire_FWSS_moving (speed != 0)
    # et NON extraire_FWSS_move_null (speed == 0) qui ne produisait
    # que des matrices identité → tous les scans empilés au même endroit.
    t_odom_raw, odom, T_mov = odometry.extraire_FWSS_moving(
        data_reader[TOPIC_LOADING[0]],
        data_reader[TOPIC_LOADING[1]]
    )

    # BUG 4 CORRIGÉ : _save_checkpoint appelé avec array 3D (N,4,4).
    # On passe le shape complet en label plutôt que len() qui ne donne que N.
    print(f"  [INFO] T_mov shape : {T_mov.shape}  "
          f"({T_mov.nbytes / 1e6:.0f} Mo)")
    np.save(str(PATH_CHECKPOINT / "odom_transform_.npy"), T_mov)
    del odom, T_mov

    t_odom = (np.array(t_odom_raw) // 1_000).astype(np.int64)
    del t_odom_raw
    utils._force_gc()

    # ── Étape 2 : Association LiDAR ↔ timestamp odométrique ───────────────
    print(f"==== Traitement des données LiDAR Front (n_odom = {len(t_odom)}) ====")
    points_index_ = []

    for i, (ts, msg) in enumerate(tqdm(data_reader[TOPIC_LOADING[2]], desc="LiDAR Front")):
        t_curr = int(ts // 1_000)   # nanosecondes → microsecondes

        est_proche = np.isclose(t_odom, t_curr, atol=10.0)
        if not np.any(est_proche):
            continue
        index_ = int(np.argmax(est_proche))
        del est_proche

        # Extraction XYZ → (N, 4) avec w=1 déjà positionné
        xyz = map_utils.fast_pointcloud2_to_xyz(msg)
        if xyz is None or len(xyz) == 0:
            continue

        # BUG 2 CORRIGÉ (dans map_utils) : filtre distance n'utilise que [:, :3]
        xyz = map_utils.filter_points_by_distance(xyz, 0.5, 30.0)
        if xyz is None or len(xyz) == 0:
            continue

        np.save(str(PATH_CHECKPOINT / f"data_point_{i}.npy"), xyz)
        del xyz

        points_index_.append(index_)

    utils._force_gc()

    # ── Étape 3 : Application des transformations ──────────────────────────
    print(f"==== Application des transformations (n_scans = {len(points_index_)}) ====")

    trans_odom = np.load(str(PATH_CHECKPOINT / "odom_transform_.npy"))
    T_odom     = trans_odom[points_index_]

    mapped   = o3d.geometry.PointCloud()
    chunk_id = 0

    for i, T in enumerate(tqdm(T_odom, desc="Transformation LiDAR → monde")):
        xyz = np.load(str(PATH_CHECKPOINT / f"data_point_{i}.npy"))
        if xyz is None or len(xyz) == 0:
            continue

        # xyz est (N, 4) avec w=1  →  xyz @ T.T donne (N, 4) correct
        # result[k, :3] = R @ xyz[k, :3] + t   (la colonne w reste ≈1)
        xyz_world = (xyz @ T.T)[:, :3]

        pcd     = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_world)
        downpcd = pcd.voxel_down_sample(voxel_size=0.05)
        mapped += downpcd

        # Surveillance RAM tous les 10 scans
        if i % 10 == 0:
            current_usage = utils.get_current_ram_gb()

            if current_usage > MAX_RAM_ALLOWED:
                print(f"\n⚠️  RAM ({current_usage:.2f} Go) > limite. "
                      f"Sauvegarde chunk {chunk_id}…")
                chunk_path = PATH_MAP_CHUNKS / f"chunk_{chunk_id}.pcd"
                o3d.io.write_point_cloud(str(chunk_path), mapped)

                del mapped
                gc.collect()

                mapped    = o3d.geometry.PointCloud()
                chunk_id += 1
                print("✅ RAM libérée. Reprise…")

    # Sauvegarde du dernier chunk
    if len(mapped.points) > 0:
        o3d.io.write_point_cloud(
            str(PATH_MAP_CHUNKS / f"chunk_{chunk_id}.pcd"), mapped
        )

    # ── Étape 4 : Assemblage final ─────────────────────────────────────────
    print("==== Assemblage de la map finale ====")
    map_complete = assembler_map_finale(PATH_MAP_CHUNKS)

    output_path = "map_totale.pcd"
    o3d.io.write_point_cloud(output_path, map_complete)
    print(f"✅ Map sauvegardée → {output_path}  "
          f"({len(map_complete.points):,} points)")

    map_complete = o3d.io.read_point_cloud(output_path)

    print("Affichage (appuie sur [Q] ou [Esc] pour fermer)…")
    o3d.visualization.draw_geometries(
        [map_complete],
        window_name="Map Totale",
        width=1280, height=720,
        left=50, top=50,
        mesh_show_back_face=False
    )