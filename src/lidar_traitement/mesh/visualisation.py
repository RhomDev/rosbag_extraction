import meshing
import gc
import os
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from lidar_traitement.mesh.lidar_corrections import azimuth_timestamps_from_points, deskew_scan_vectorized

from lidar_traitement.mapping import utils as map_utils
from lidar_traitement.odometry import odometry
from lidar_traitement.extraction import extrait_rosbag
from lidar_traitement.optimisation import memoire_gpu as utils
from lidar_traitement.optimisation.gpu_transform import batch_transform_gpu

import open3d as o3d
import numpy as np
from tqdm import tqdm

BAG = '/home/rhomdev/Documents/stage/bag_files/ez10_gen1_sensors_2026-04-02-16-45-39'
TOPIC_LOADING = [
    "/ez10_gen1/received_raw_four_wheel_steering",
    "/ez10_gen1/gnss_main/pose_lpz",
    "/ez10_gen1/hesai_front/cloud",
]
MAX_RAM_ALLOWED = 11.0


def _find_odom_index(t_odom: np.ndarray, t_curr: float, max_gap: float = 0.1) -> int:
    idx = np.searchsorted(t_odom, t_curr)
    candidates = [i for i in (idx - 1, idx) if 0 <= i < len(t_odom)]
    if not candidates:
        return -1
    best = min(candidates, key=lambda i: abs(t_odom[i] - t_curr))
    return best if abs(t_odom[best] - t_curr) <= max_gap else -1


# ════════════════════════════════════════════════════════════════════

def _mesh_one_cloud(xyz: np.ndarray):
    """Les points arrivent déjà transformés depuis le GPU — pas de .transform()."""
    mesh = meshing.mesh_style_ball_pivoting(xyz)
    return mesh
# ════════════════════════════════════════════════════════════════════
def _mesh_worker_picklable(xyz):
    """ Calcule le mesh et retourne uniquement des tableaux NumPy """
    # 1. On appelle ta fonction existante qui génère le TriangleMesh
    mesh = _mesh_one_cloud(xyz)

    # 2. On extrait les données en format NumPy (Picklable)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # Optionnel : si ton mesh a des couleurs ou des normales
    colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
    normals = np.asarray(mesh.vertex_normals) if mesh.has_vertex_normals() else None

    return vertices, triangles, colors, normals

# ════════════════════════════════════════════════════════════════════

def mesh_one_frame(xyz: np.ndarray, method: str = 'ball_pivoting') -> o3d.geometry.TriangleMesh:
    """
    Crée un mesh à partir d'une seule frame/nuage de points.
    
    Args:
        xyz: Points 3D (N, 3)
        method: 'ball_pivoting', 'poisson', ou 'alpha' (défaut: 'ball_pivoting')
    
    Returns:
        TriangleMesh Open3D
    """
    if xyz is None or len(xyz) == 0:
        print("Erreur : nuage de points vide ou None")
        return None
    
    if method == 'ball_pivoting':
        mesh = meshing.mesh_style_ball_pivoting(xyz)
    elif method == 'poisson':
        mesh = meshing.mesh_style_poisson(xyz)
    elif method == 'alpha':
        mesh = meshing.mesh_style_alpha(xyz, alpha=0.1)
    else:
        raise ValueError(f"Méthode inconnue : {method}")
    
    return mesh


def mesh_mapping(xyz_list: list, method: str = 'ball_pivoting', visualize: bool = True) -> o3d.geometry.TriangleMesh:
    """
    Crée et combine les meshes de plusieurs frames pour former un mapping global.
    
    Args:
        xyz_list: Liste de nuages de points (chacun de shape (N, 3))
        method: 'ball_pivoting', 'poisson', ou 'alpha'
        visualize: Si True, visualise l'évolution du mesh en temps réel
    
    Returns:
        TriangleMesh combiné (mesh global)
    """
    if not xyz_list or all(xyz is None or len(xyz) == 0 for xyz in xyz_list):
        print("Erreur : aucun nuage de points valide")
        return None
    
    meshes = []
    
    # Meshing parallèle de chaque frame
    MAX_WORKERS = min(10, os.cpu_count() or 1)
    ctx = mp.get_context("spawn")
    
    print(f"Meshing {len(xyz_list)} frames avec {MAX_WORKERS} workers...")
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=ctx) as executor:
        futures = {executor.submit(_mesh_worker_picklable, xyz): i
                   for i, xyz in enumerate(xyz_list) if xyz is not None and len(xyz) > 0}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Meshing"):
            try:
                vertices, triangles, colors, normals = future.result()
                
                # Reconstruction du mesh Open3D
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(vertices)
                mesh.triangles = o3d.utility.Vector3iVector(triangles)
                
                if colors is not None and len(colors) > 0:
                    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
                if normals is not None and len(normals) > 0:
                    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
                
                meshes.append(mesh)
                
            except Exception as e:
                print(f"Erreur meshing : {e}")
    
    # Fusion des meshes
    print("\nFusion des meshes...")
    final_mesh = o3d.geometry.TriangleMesh()
    for mesh in meshes:
        final_mesh += mesh
    
    final_mesh.compute_vertex_normals()
    
    # Visualisation optionnelle
    if visualize and len(meshes) > 0:
        print("Visualisation du mapping en cours...")
        import time
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Mapping 3D", width=1280, height=720)
        
        display_mesh = o3d.geometry.TriangleMesh()
        vis.add_geometry(display_mesh)
        
        view_initialized = False
        
        for i, mesh in enumerate(meshes):
            display_mesh += mesh
            display_mesh.compute_vertex_normals()
            display_mesh.paint_uniform_color([0.6, 0.6, 0.6])
            
            vis.update_geometry(display_mesh)
            
            if not view_initialized:
                vis.reset_view_point(True)
                view_initialized = True
            
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.05)
            
            if i % max(1, len(meshes) // 10) == 0:
                print(f"Frame {i}/{len(meshes)} affichée...")
        
        print("Fin de la reconstruction. Appuyez sur Q pour fermer.")
        vis.run()
        vis.destroy_window()
    
    return final_mesh

if __name__ == '__main__':
    print("==== LANCEMENT DE L'ÉTUDE MESHING (GPU) ====")

    reader, topics_type = extrait_rosbag._open_bag_reader(BAG, TOPIC_LOADING)
    data_reader = extrait_rosbag._open_bag_extrait(reader, topics_type, TOPIC_LOADING)
    del reader, topics_type
    utils._force_gc()

    print("==== Odométrie ====")
    t_odom_raw, odom, T_mov = odometry.extraire_FWSS_moving(
        data_reader[TOPIC_LOADING[0]],
        data_reader[TOPIC_LOADING[1]]
    )
    del odom
    t_odom = (np.array(t_odom_raw) / 1e9).astype(np.float64)
    del t_odom_raw
    utils._force_gc()

    print("==== Collecte LiDAR ====")
    mesh_tasks = []   # stocke (xyz brut, T) le temps de la collecte

    for ts, msg in tqdm(data_reader[TOPIC_LOADING[2]], desc="LiDAR Front"):
        t_curr = ts / 1e9
        idx = _find_odom_index(t_odom, t_curr, max_gap=0.1)
        if idx < 0:
            continue

        xyz = map_utils.fast_pointcloud2_to_xyz(msg)
        if xyz is None or len(xyz) == 0:
            continue
        xyz = map_utils.filter_points_by_distance(xyz, 0.5, 30.0)
        if xyz is None or len(xyz) == 0:
            continue

        # Pose précédente pour interpoler le mouvement intra-scan
        T_prev = T_mov[max(0, idx - 1)]
        T_curr = T_mov[idx]

        t_pts = azimuth_timestamps_from_points(xyz, ts)
        xyz = deskew_scan_vectorized(
            xyz, t_pts, T_prev, T_curr,
            t_start_ns=ts,
            t_end_ns=ts + 100_000_000,  # 100 ms @ 10 Hz
        )
        mesh_tasks.append((xyz, T_curr))

    print(f"Frames valides : {len(mesh_tasks)}")

    # ════════════════════════════════════════════════════════════════

    print("==== Transformation GPU (batch) ====")
    clouds_bruts  = [xyz for xyz, _  in mesh_tasks[:30]]
    transforms    = [T   for _,   T  in mesh_tasks[:30]]

    # batch_transform_gpu : tous les nuages → GPU → retour CPU en une passe
    clouds_transformes = batch_transform_gpu(clouds_bruts, transforms)

    del clouds_bruts, mesh_tasks   # libère RAM immédiatement
    utils._force_gc()

    # ════════════════════════════════════════════════════════════════

    # ════════════════════════════════════════════════════════════════

    print("==== Meshing parallèle ====")
    meshes = []
    MAX_WORKERS = min(10, os.cpu_count() or 1)
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=ctx) as executor:
        # On utilise notre nouvelle fonction "picklable"
        futures = {executor.submit(_mesh_worker_picklable, xyz): i
                   for i, xyz in enumerate(clouds_transformes)}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Meshing"):
            try:
                # On récupère les tableaux NumPy
                vertices, triangles, colors, normals = future.result()

                # On reconstruit l'objet Open3D dans le processus principal
                reconstructed_mesh = o3d.geometry.TriangleMesh()
                reconstructed_mesh.vertices = o3d.utility.Vector3dVector(vertices)
                reconstructed_mesh.triangles = o3d.utility.Vector3iVector(triangles)

                if colors is not None and len(colors) > 0:
                    reconstructed_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
                if normals is not None and len(normals) > 0:
                    reconstructed_mesh.vertex_normals = o3d.utility.Vector3dVector(normals)

                meshes.append(reconstructed_mesh)

            except Exception as e:
                print(f"Erreur meshing : {e}")
    # ════════════════════════════════════════════════════════════════

    print("==== Visualisation de l'évolution du Mesh ====")
    import time
    if meshes:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Évolution du Mapping 3D", width=1280, height=720)

        # Initialisation du mesh global (vide au début)
        final_mesh = o3d.geometry.TriangleMesh()
        vis.add_geometry(final_mesh)

        # Pour centrer la vue sur le premier élément
        view_initialized = False

        pred_Y = transforms[0][1, 3]

        for i, m in enumerate(meshes):
            # 1. On ajoute le nouveau morceau au mesh global
            final_mesh += m

            # 2. On rafraîchit les normales pour le relief
            final_mesh.compute_vertex_normals()
            final_mesh.paint_uniform_color([0.6, 0.6, 0.6])

            # 3. Mise à jour de la fenêtre
            vis.update_geometry(final_mesh)

            # Si c'est le premier frame, on cadre la caméra
            if not view_initialized:
                vis.reset_view_point(True)
                view_initialized = True

            # 4. On laisse Open3D gérer les événements (clics, rotation) et on dessine
            vis.poll_events()
            vis.update_renderer()

            # Petit délai pour avoir le temps de voir (ex: 50ms par frame)

            time.sleep(0.05)

            if i % 10 == 0:
                print(f"Frame {i}/{len(meshes)} affichée...")

        print("Fin de la reconstruction. Vous pouvez explorer la carte.")
        vis.run()  # Garde la fenêtre ouverte à la fin
        vis.destroy_window()

    # print("==== Fusion et Rendu des meshes ====")
    # if meshes:
    #     # 1. Fusionner tous les morceaux
    #     final_mesh = meshes[0]
    #     for m in tqdm(meshes[1:], desc="Fusion"):
    #         final_mesh += m
    #
    #     # 2. Couleur Grise (0.5 partout pour un gris neutre, 0.7 pour un gris clair)
    #     final_mesh.paint_uniform_color([0.6, 0.6, 0.6])
    #
    #     # 3. CALCUL DU RELIEF (Essentiel)
    #     # On recalcule les normales pour que la lumière crée des ombres sur les surfaces
    #     final_mesh.compute_vertex_normals()
    #
    #     # 4. Affichage avec éclairage
    #     o3d.visualization.draw_geometries(
    #         [final_mesh],
    #         window_name="Carte 3D - Rendu Relief",
    #         mesh_show_back_face=True,
    #         mesh_show_wireframe=False  # On veut voir les ombres, pas la grille
    #     )