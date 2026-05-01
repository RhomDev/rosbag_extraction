import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from collections import deque

# ════════════════════════════════════════════════════════════════════
# CHANGEMENT 1/3 — nouveaux imports en haut du fichier
#
# AVANT : rien de nouveau
# APRÈS :
from lidar_traitement.optimisation.gpu_transform import transformation_gpu

try:
    from kiss_icp.algorithms.icp import kiss_icp_pairwise
    _KISS_AVAILABLE = True
    print("[INFO] kiss-icp disponible — ICP accéléré activé.")
except ImportError:
    _KISS_AVAILABLE = False
    print("[WARN] kiss-icp non installé (pip install kiss-icp). Fallback Open3D.")
# ════════════════════════════════════════════════════════════════════


def transformation(pc, T):
    """Conservée pour compatibilité — préférer transformation_gpu() pour N > 10k."""
    ones = np.ones((pc.shape[0], 1), dtype=np.float32)
    points_homo = np.hstack([pc, ones])
    return (T @ points_homo.T).T[:, :3]


class LidarMapper:
    def __init__(self, voxel_size=0.2, icp_window=50, downsample_every=20):
        self.voxel_size = voxel_size
        self.icp_window = icp_window
        self.downsample_every = downsample_every
        self.global_map = o3d.geometry.PointCloud()
        self._frame_count = 0
        self._local_window = deque(maxlen=icp_window)

    def odom_to_transform_matrix(self, position, quat_wxyz):
        matrix = np.eye(4)
        r = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        matrix[:3, :3] = r.as_matrix()
        matrix[:3, 3] = position
        return matrix

    def process_frame(self, points_array: np.ndarray, pos_xyz, quat_wxyz,
                      skip_icp: bool = False):

        transform_matrix = self.odom_to_transform_matrix(pos_xyz, quat_wxyz)

        # ════════════════════════════════════════════════════════════
        # CHANGEMENT 2/3 — transformation GPU au lieu de pcd.transform()
        #
        # AVANT :
        #   pcd = o3d.geometry.PointCloud()
        #   pcd.points = o3d.utility.Vector3dVector(points_array)
        #   pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        #   transform_matrix = self.odom_to_transform_matrix(pos_xyz, quat_wxyz)
        #   pcd.transform(transform_matrix)    # ← CPU ici, ~20ms sur 120k pts
        #
        # APRÈS :
        pts = transformation_gpu(points_array[:, :3], transform_matrix)
        #   ^ GPU si N > 10 000 pts (~2ms), CPU sinon — sélection automatique
        #   Les points sont déjà dans le repère monde : plus de pcd.transform()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        # ════════════════════════════════════════════════════════════

        if self._frame_count == 0 or skip_icp:
            self._local_window.append(pcd)
            self.global_map += pcd

        else:
            local_ref = self._build_local_ref()

            # ════════════════════════════════════════════════════════
            # CHANGEMENT 3/3 — kiss-icp au lieu de registration_icp
            #
            # AVANT :
            #   reg_p2p = o3d.pipelines.registration.registration_icp(
            #       pcd, self.global_map,              # ← carte ENTIÈRE
            #       max_correspondence_distance=self.voxel_size * 2.0,
            #       init=np.eye(4),
            #       estimation_method=o3d.pipelines.registration
            #                           .TransformationEstimationPointToPoint(),
            #       criteria=o3d.pipelines.registration
            #                   .ICPConvergenceCriteria(max_iteration=30)
            #   )
            #   pcd.transform(reg_p2p.transformation)
            #
            # APRÈS :
            if _KISS_AVAILABLE:
                T_icp, _ = kiss_icp_pairwise(
                    source=np.asarray(pcd.points),
                    target=np.asarray(local_ref.points),   # ← fenêtre locale seulement
                    initial_guess=np.eye(4),
                    max_correspondence_dist=self.voxel_size * 2.0
                )
                pcd.transform(T_icp)
            else:
                # Fallback Open3D — fenêtre locale au lieu de la carte entière
                reg = o3d.pipelines.registration.registration_icp(
                    pcd, local_ref,                        # ← local_ref pas global_map
                    max_correspondence_distance=self.voxel_size * 2.0,
                    init=np.eye(4),
                    estimation_method=o3d.pipelines.registration
                                        .TransformationEstimationPointToPoint(),
                    criteria=o3d.pipelines.registration
                                .ICPConvergenceCriteria(max_iteration=30)
                )
                pcd.transform(reg.transformation)
            # ════════════════════════════════════════════════════════

            self._local_window.append(pcd)
            self.global_map += pcd

        self._frame_count += 1

        if self._frame_count % self.downsample_every == 0:
            self.global_map = self.global_map.voxel_down_sample(self.voxel_size)

    def _build_local_ref(self) -> o3d.geometry.PointCloud:
        ref = o3d.geometry.PointCloud()
        for p in self._local_window:
            ref += p
        return ref.voxel_down_sample(self.voxel_size)

    def visualize(self):
        o3d.visualization.draw_geometries([self.global_map])