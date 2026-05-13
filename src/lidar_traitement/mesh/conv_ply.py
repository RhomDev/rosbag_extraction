import argparse
from pathlib import Path

import open3d as o3d
from tqdm import tqdm
import numpy as np


from lidar_traitement.mapping import utils as map_utils
from lidar_traitement.extraction import extrait_rosbag
from lidar_traitement.optimisation import memoire_gpu as utils

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

def main(args):
    root      = Path("../out/save_img_mesh_lidar")
    mesh_dir  = root / "mesh_ply"

    mesh_dir.mkdir(parents=True, exist_ok=True)


    # ── Lecture bag ─────────────────────────────────────────────────
    print("📦 Chargement du bag...")
    reader, topics_type = extrait_rosbag._open_bag_reader(args.bag, args.topic)
    data_reader = extrait_rosbag._open_bag_extrait(reader, topics_type, args.topic)
    del reader, topics_type
    utils._force_gc()

    mesh_groups = []  # liste de groupes (chaque groupe = liste de xyz)
    current_group = []  # groupe en cours de construction

    for ts, msg in tqdm(data_reader[args.topic[2]], desc="Lecture LiDAR  ", position=0):

        xyz = map_utils.fast_pointcloud2_to_xyz(msg)

        if xyz is None or len(xyz) < 10:
            continue

        xyz = map_utils.filter_points_by_distance(xyz, 2.0, 60.0)

        if xyz is None or len(xyz) == 0:
            continue

        xyz = np.asarray(xyz)

        # tableau structuré ROS
        if xyz.dtype.names is not None:
            xyz = np.vstack([
                xyz['x'],
                xyz['y'],
                xyz['z']
            ]).T

        # si tableau Nx4 -> garder xyz seulement
        if xyz.ndim == 2 and xyz.shape[1] >= 3:
            xyz = xyz[:, :3]

        xyz = xyz.astype(np.float64)

        # suppression NaN / inf
        xyz = xyz[np.isfinite(xyz).all(axis=1)]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        o3d.io.write_point_cloud(
            str(mesh_dir / f"pointsCloud_{ts}_frame.ply"),
            pcd
        )


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

    args = parser.parse_args()
    main(args)
