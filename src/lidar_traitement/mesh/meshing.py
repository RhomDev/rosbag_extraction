import open3d as o3d
import numpy as np


def mesh_style_ball_pivoting(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[:, :3])
    pcd.estimate_normals()

    # Calcul automatique du rayon basé sur la densité des points
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist # On ajuste ici la "sensibilité"

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector([radius, radius * 2])
    )
    mesh.compute_vertex_normals()
    return mesh


def mesh_style_poisson(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[:, :3])
    pcd.estimate_normals()

    # depth=9 définit la résolution (plus c'est haut, plus c'est précis mais lent)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

    # On nettoie les triangles créés dans le vide (basse densité)
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    mesh.compute_vertex_normals()
    return mesh


def mesh_style_alpha(xyz, alpha=0.1):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[:, :3])

    # Plus alpha est petit, plus le mesh colle aux points.
    # Plus il est grand, plus il devient "convexe".
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.compute_vertex_normals()
    return mesh