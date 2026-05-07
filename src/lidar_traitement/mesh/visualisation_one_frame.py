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
from lidar_traitement.extraction import extrait_rosbag

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

# ════════════════════════════════════════════════════════════════════
# WORKER FUSIONNÉ : MESH + RENDU dans le même processus
# → Zéro sérialisation de mesh entre processus
# ════════════════════════════════════════════════════════════════════

def _mesh_render_worker(task):
    """
    Effectue le meshing ET le rendu dans le même processus.
    Reçoit/renvoie uniquement du NumPy + chemins (pickle léger).
    """
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

        center      = np.zeros(3)
        final_eye   = R @ (np.array(eye_pos) - center) + center
        rotated_up  = R @ np.array([0.0, 0.0, 1.0])

        # ── 3. Rendu ────────────────────────────────────────────────
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"
        renderer.scene.add_geometry("mesh", mesh, mat)
        renderer.setup_camera(fov, center.tolist(), final_eye.tolist(), rotated_up.tolist())

        img_array = np.asarray(renderer.render_to_image())
        del renderer  # libération immédiate GPU/mémoire

        # ── 4. Sauvegarde image ─────────────────────────────────────
        img_bgr = cv2.cvtColor(np.rot90(img_array, k=1), cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_path, img_bgr)

        return idx, img_path

    except Exception as e:
        print(f"\n[Worker {idx}] ⚠ Erreur : {e}")
        return idx, None


# ════════════════════════════════════════════════════════════════════
# THREAD VIDÉO : encode en streaming, dans l'ordre, sans attendre la fin
# ════════════════════════════════════════════════════════════════════

def _video_writer_thread(video_path, vid_w, vid_h, fps, frame_queue, total):
    """
    Consomme la queue au fur et à mesure.
    Réassemble les frames dans l'ordre grâce à un buffer pending.
    """
    os.makedirs(os.path.dirname(str(video_path)), exist_ok=True)
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (vid_w, vid_h)
    )

    pending      = {}   # idx → img_path (buffer pour réordonner)
    next_frame   = 0    # prochain idx attendu
    written      = 0

    with tqdm(total=total, desc="Encodage vidéo ", position=1, leave=True) as pbar:
        while written < total:
            try:
                idx, img_path = frame_queue.get(timeout=60)
            except queue.Empty:
                print("\n[Vidéo] Timeout — arrêt de l'encodeur")
                break

            if img_path:
                pending[idx] = img_path

            # Écrire toutes les frames consécutives disponibles
            while next_frame in pending:
                path = pending.pop(next_frame)
                frame = cv2.imread(path)
                if frame is not None:
                    writer.write(frame)
                next_frame += 1
                written    += 1
                pbar.update(1)

    writer.release()


# ════════════════════════════════════════════════════════════════════
# SAUVEGARDE MESH (optionnelle)
# ════════════════════════════════════════════════════════════════════

def save_mesh_ply(mesh, output_path):
    os.makedirs(os.path.dirname(str(output_path)), exist_ok=True)
    ok = o3d.io.write_triangle_mesh(str(output_path), mesh)
    if not ok:
        print(f"❌ Erreur sauvegarde : {output_path}")


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

def main(args):
    root      = Path("../../../out/save_img_mesh_lidar")
    img_dir   = root / "images"
    video_dir = root / "video"
    mesh_dir  = root / "mesh"

    shutil.rmtree(root, ignore_errors=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    if args.save_meshes:
        mesh_dir.mkdir(parents=True, exist_ok=True)

    # ── Config caméra ───────────────────────────────────────────────
    CAM_EYE    = [0.0, 2.0, 2.0]
    CAM_ANGLES = [90.0, 90.0, 135.0]
    CAM_FOV    = 120
    IMG_W, IMG_H = 1080, 1920
    FPS          = 30

    # Après rot90, les dimensions sont inversées pour la vidéo
    VID_W, VID_H = IMG_H, IMG_W   # → 1920 × 1080

    # ── Lecture bag ─────────────────────────────────────────────────
    print("📦 Chargement du bag...")
    reader, topics_type = extrait_rosbag._open_bag_reader(args.bag, [args.topic])
    data_reader = extrait_rosbag._open_bag_extrait(reader, topics_type, [args.topic])

    mesh_tasks = []
    for ts, msg in tqdm(data_reader[args.topic], desc="Lecture LiDAR  ", position=0):
        xyz = map_utils.fast_pointcloud2_to_xyz(msg)
        if xyz is None or len(xyz) < 10:
            continue
        xyz = map_utils.filter_points_by_distance(xyz, 2.0, 60.0)
        mesh_tasks.append(xyz)

    n_scans     = len(mesh_tasks)
    total_cores = os.cpu_count() or 4
    max_workers = min(args.max_workers, total_cores)

    print(f"\n🔢 {n_scans} scans  |  {max_workers}/{total_cores} cœurs\n")

    # ── Thread vidéo (démarre avant le pool pour encoder en streaming) ──
    frame_queue = queue.Queue(maxsize=max_workers * 3)   # back-pressure RAM
    video_path  = video_dir / "output.mp4"

    vt = threading.Thread(
        target=_video_writer_thread,
        args=(video_path, VID_W, VID_H, FPS, frame_queue, n_scans),
        daemon=True
    )
    vt.start()

    # ── Construction des tâches ──────────────────────────────────────
    tasks = [
        (
            i, xyz,
            str(img_dir / f"scan_{i:05d}.jpg"),
            IMG_W, IMG_H, CAM_EYE, CAM_ANGLES, CAM_FOV
        )
        for i, xyz in enumerate(mesh_tasks)
    ]

    # ── Pool parallèle : mesh + rendu ───────────────────────────────
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = {executor.submit(_mesh_render_worker, t): t[0] for t in tasks}

        with tqdm(total=n_scans, desc="Mesh + Rendu   ", position=0, leave=True) as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    frame_queue.put(result)   # alimente le thread vidéo
                pbar.update(1)

    # ── Attente finalisation vidéo ───────────────────────────────────
    print("\n⏳ Finalisation de la vidéo...")
    vt.join()

    print(f"\n✅ Images : {img_dir}")
    print(f"✅ Vidéo  : {video_path}")


if __name__ == "__main__":
    BAG_DEFAULT = '/home/rhomdev/Documents/stage/bag_files/ez10_gen1_sensors_2026-04-02-16-45-39'

    parser = argparse.ArgumentParser()
    parser.add_argument("--bag",         type=str, default=BAG_DEFAULT)
    parser.add_argument("--topic",       type=str, default="/ez10_gen1/hesai_front/cloud")
    parser.add_argument("--max_workers", type=int, default=os.cpu_count(),
                        help="Nombre de cœurs (défaut: tous)")
    parser.add_argument("--fps",         type=int, default=30)
    parser.add_argument("--save_meshes", action="store_true")
    args = parser.parse_args()
    main(args)