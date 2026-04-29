import open3d as o3d

# 1. Charger le nuage de points
pcd = o3d.io.read_point_cloud("../../out/save_mesh/output_cloud_Non-Sol_29_04_2026_10_59_31.ply")

# 2. Afficher des informations de base (optionnel)
print(pcd)  # Affiche le nombre de points

# 3. Visualiser le nuage de points
o3d.visualization.draw_geometries([pcd])