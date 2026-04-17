import open3d as o3d

# 1. Charger le nuage de points
pcd = o3d.io.read_point_cloud("../../out/save_mesh/output_mesh_16_04_2026_16_50_55.ply")

# 2. Afficher des informations de base (optionnel)
print(pcd)  # Affiche le nombre de points

# 3. Visualiser le nuage de points
o3d.visualization.draw_geometries([pcd])