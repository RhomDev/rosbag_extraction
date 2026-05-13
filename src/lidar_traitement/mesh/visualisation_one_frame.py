import os
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import threading
import queue
import shutil

import open3d as o3d
import numpy as np
from tqdm import tqdm
import cv2

from lidar_traitement.mapping import utils as map_utils
from lidar_traitement.odometry import odometry
from lidar_traitement.extraction import extrait_rosbag
from lidar_traitement.optimisation import memoire_gpu as utils

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

# ════════════════════════════════════════════════════════════════════
# WORKER FUSIONNÉ : MESH + RENDU dans le même processus
# ════════════════════════════════════════════════════════════════════

def _mesh_render_worker(task):
    idx, xyz, img_path, width, height, eye_pos, angles, fov = task

    try:
        # ── 1. Meshing ──────────────────────────────────────────────
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz[:, :3])
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(15)

        distances = pcd.compute_nearest_neighbor_distance()
        radius = 1.5 * np.mean(distances)

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector([radius, radius * 2])
        )
        mesh.compute_vertex_normals()

        # ── 2. Caméra ───────────────────────────────────────────────
        rad = np.radians(angles)
        rx, ry, rz = rad
        Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
        Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
        Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
        R  = Rz @ Ry @ Rx

        center     = np.zeros(3)
        final_eye  = R @ (np.array(eye_pos) - center) + center
        rotated_up = R @ np.array([0.0, 0.0, 1.0])

        # ── 3. Rendu ────────────────────────────────────────────────
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"
        renderer.scene.add_geometry("mesh", mesh, mat)
        renderer.setup_camera(fov, center.tolist(), final_eye.tolist(), rotated_up.tolist())

        img_array = np.asarray(renderer.render_to_image())
        del renderer

        # ── 4. Sauvegarde ───────────────────────────────────────────
        img_bgr = cv2.cvtColor(np.rot90(img_array, k=1), cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_path, img_bgr)

        return idx, img_path

    except Exception as e:
        print(f"\n[Worker {idx}] ⚠ Erreur : {e}")
        return idx, None   # None signale l'échec sans bloquer le thread vidéo

# ════════════════════════════════════════════════════════════════════

def _find_odom_index(t_odom: np.ndarray, t_curr: float, max_gap: float = 0.1) -> int:
    idx = np.searchsorted(t_odom, t_curr)
    candidates = [i for i in (idx - 1, idx) if 0 <= i < len(t_odom)]
    if not candidates:
        return -1
    best = min(candidates, key=lambda i: abs(t_odom[i] - t_curr))
    return best if abs(t_odom[best] - t_curr) <= max_gap else -1

# ════════════════════════════════════════════════════════════════════
# THREAD VIDÉO
# ════════════════════════════════════════════════════════════════════

def _video_writer_thread(video_path, vid_w, vid_h, fps, frame_queue, total):
    """
    Encode en streaming dans l'ordre.
    Compte TOUS les résultats (succès + échecs) pour ne jamais se bloquer.
    """
    os.makedirs(os.path.dirname(str(video_path)), exist_ok=True)
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (vid_w, vid_h)
    )

    pending    = {}   # idx → img_path | None
    next_frame = 0
    received   = 0    # ← compte tout (succès ET échecs)

    with tqdm(total=total, desc="Encodage vidéo ", position=1, leave=True) as pbar:
        while received < total:
            try:
                idx, img_path = frame_queue.get(timeout=60)
            except queue.Empty:
                print("\n[Vidéo] Timeout — arrêt de l'encodeur")
                break

            received += 1
            # None = worker échoué : on l'enregistre quand même pour
            # ne pas bloquer les frames suivantes dans le buffer pending
            pending[idx] = img_path

            while next_frame in pending:
                path = pending.pop(next_frame)
                if path is not None:        # on saute les frames échouées
                    frame = cv2.imread(path)
                    if frame is not None:
                        writer.write(frame)
                next_frame += 1
                pbar.update(1)

    writer.release()

# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

def main(args):
    root      = Path("../out/save_img_mesh_lidar")
    img_dir   = root / "images"
    video_dir = root / "video"
    mesh_dir  = root / "mesh"

    shutil.rmtree(root, ignore_errors=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    if args.save_meshes:
        mesh_dir.mkdir(parents=True, exist_ok=True)

    CAM_EYE    = [0.0, 2.0, 2.0]
    CAM_ANGLES = [90.0, 90.0, 135.0]
    CAM_FOV    = 120
    IMG_W, IMG_H = 1080, 1920
    FPS          = args.fps
    VID_W, VID_H = IMG_H, IMG_W   # dimensions après rot90

    # ── Lecture bag ─────────────────────────────────────────────────
    print("📦 Chargement du bag...")
    reader, topics_type = extrait_rosbag._open_bag_reader(args.bag, args.topic)
    data_reader = extrait_rosbag._open_bag_extrait(reader, topics_type, args.topic)
    del reader, topics_type
    utils._force_gc()

    t_odom_raw, odom, T_mov = odometry.extraire_FWSS_move_null(
        data_reader[args.topic[0]],
        data_reader[args.topic[1]]
    )
    del odom, T_mov
    t_odom = (np.array(t_odom_raw) / 1e9).astype(np.float64)
    del t_odom_raw
    utils._force_gc()

    # ── Collecte des scans valides ───────────────────────────────────
    # BUG CORRIGÉ : mesh_tasks est maintenant une liste PLATE de xyz,
    # pas une liste de listes. L'odométrie sert uniquement de filtre.
    mesh_groups = []  # liste de groupes (chaque groupe = liste de xyz)
    current_group = []  # groupe en cours de construction

    for ts, msg in tqdm(data_reader[args.topic[2]], desc="Lecture LiDAR  ", position=0):
        t_curr = ts / 1e9
        has_odom = _find_odom_index(t_odom, t_curr, max_gap=0.1) >= 0

        if not has_odom:
            # Fin d'un arrêt → on clôt le groupe courant s'il est non vide
            if current_group:
                mesh_groups.append(current_group)
                current_group = []
            continue

        xyz = map_utils.fast_pointcloud2_to_xyz(msg)
        if xyz is None or len(xyz) < 10:
            continue
        xyz = map_utils.filter_points_by_distance(xyz, 2.0, 60.0)
        if xyz is None or len(xyz) == 0:
            continue

        current_group.append(xyz)

    # Dernier groupe non clôturé
    if current_group:
        mesh_groups.append(current_group)

        # ── Construction des tâches : fusion des points par groupe ───────
        # Tous les scans d'un même arrêt sont concaténés → nuage plus dense
        # → meilleure qualité de mesh
        tasks = []
        for i, group in enumerate(mesh_groups):
            xyz_merged = np.concatenate(group, axis=0)  # ← fusion ici
            tasks.append((
                i, xyz_merged,
                str(img_dir / f"arret_{i:05d}.jpg"),
                IMG_W, IMG_H, CAM_EYE, CAM_ANGLES, CAM_FOV
            ))

        n_scans = len(tasks)
        total_cores = os.cpu_count() or 4
        max_workers = min(args.max_workers, total_cores)

        if n_scans == 0:
            print("⚠ Aucun groupe valide — vérifiez le topic ou le max_gap.")
            return

        pts_per_group = [len(np.concatenate(g)) for g in mesh_groups]
        print(f"\n🔢 {n_scans} arrêts  |  points/arrêt : "
              f"min={min(pts_per_group):,}  max={max(pts_per_group):,}  "
              f"moy={int(np.mean(pts_per_group)):,}")
        print(f"⚙  {max_workers}/{total_cores} cœurs\n")

        # ── Thread vidéo ─────────────────────────────────────────────────
        frame_queue = queue.Queue(maxsize=max_workers * 3)
        video_path = video_dir / "output.mp4"

        vt = threading.Thread(
            target=_video_writer_thread,
            args=(video_path, VID_W, VID_H, FPS, frame_queue, n_scans),
            daemon=True
        )
        vt.start()

        # ── Pool parallèle ───────────────────────────────────────────────
        ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = {executor.submit(_mesh_render_worker, t): t[0] for t in tasks}

        with tqdm(total=n_scans, desc="Mesh + Rendu   ", position=0, leave=True) as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    frame_queue.put(result)
                pbar.update(1)

    print("\n⏳ Finalisation de la vidéo...")
    vt.join()

    print(f"\n✅ Images : {img_dir}")
    print(f"✅ Vidéo  : {video_path}")


if __name__ == "__main__":
    BAG_DEFAULT = '/home/rhomdev/Documents/stage/bag_files/ez10_gen1_sensors_2026-04-02-16-45-39'
    TOPIC_LOADING = [
        "/ez10_gen1/received_raw_four_wheel_steering",
        "/ez10_gen1/gnss_main/pose_lpz",
        "/ez10_gen1/hesai_front/cloud",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--bag",         type=str,  default=BAG_DEFAULT)
    parser.add_argument("--topic",       nargs=3,   default=TOPIC_LOADING)
    parser.add_argument("--max_workers", type=int,  default=os.cpu_count())
    parser.add_argument("--fps",         type=int,  default=30)
    parser.add_argument("--save_meshes", action="store_true")
    args = parser.parse_args()
    main(args)