import numpy as np
import open3d as o3d


def transformation(pc, T):
    ones = np.ones((pc.shape[0], 1), dtype=np.float32)
    points_homo = np.hstack([pc, ones])  # Forme (N, 4)

    # Application de la matrice T_4x4
    # T_4x4 @ points_homo.T donne une matrice (4, N)
    xyz_new = (T @ points_homo.T).T[:, :3]  # On ne garde que les 3 premières colonnes (X, Y, Z)

    return xyz_new


import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R


class LidarMapper:
    def __init__(self, voxel_size=0.2):
        self.voxel_size = voxel_size
        self.global_map = o3d.geometry.PointCloud()
        self.is_first_cloud = True

    def odom_to_transform_matrix(self, position, quat_xyzw):
        """
        Prend la position et votre quaternion [w, x, y, z]
        et retourne une matrice de transformation 4x4.
        """
        matrix = np.eye(4)

        r = R.from_quat(quat_xyzw)
        matrix[:3, :3] = r.as_matrix()
        matrix[:3, 3] = position
        return matrix

    def process_frame(self, points_array, pos_xyz, quat_wxyz):
        """Intègre le nuage à la carte."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_array)
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)

        # Matrice depuis votre GNSSTrajectory
        transform_matrix = self.odom_to_transform_matrix(pos_xyz, quat_wxyz)
        pcd.transform(transform_matrix)

        if self.is_first_cloud:
            self.global_map = pcd
            self.is_first_cloud = False
        else:
            # Recalage ICP (optionnel mais recommandé pour corriger la petite dérive)
            reg_p2p = o3d.pipelines.registration.registration_icp(
                pcd, self.global_map,
                max_correspondence_distance=self.voxel_size * 2.0,
                init=np.eye(4),
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
            )
            pcd.transform(reg_p2p.transformation)

            # Fusion
            self.global_map += pcd
            self.global_map = self.global_map.voxel_down_sample(voxel_size=self.voxel_size)

    def visualize(self):
        o3d.visualization.draw_geometries([self.global_map])