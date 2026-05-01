from PIL.GimpGradientFile import linear

from lidar_traitement.extraction import extrait_rosbag
import odometry
from lidar_traitement.optimisation.utils import Odometry, FourWheelSteeringStamped, GNSSTrajectory
import matplotlib.pyplot as plt

BAG = '/home/rhomdev/Documents/stage/bag_files/ez10_gen1_sensors_2026-04-02-16-45-39'
TOPIC_LOADING = ["/ez10_gen1/received_raw_four_wheel_steering",
                 "/ez10_gen1/gnss_main/pose_lpz",
                 ]

def plot_speed(timestamps, speeds, name=""):
    plt.plot(timestamps, speeds, label=name)

    plt.xlabel("coord X")
    plt.ylabel("coord Y")
    plt.title("Trajectoire véhicule")
    plt.grid(True)
    plt.legend()


def save_plot():
    import datetime
    import os
    now = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

    out_dir = "../../out/save_coub_lidar"
    os.makedirs(out_dir, exist_ok=True)

    plt.savefig(os.path.join(out_dir, f"vitesse_{now}.png"))


if __name__ == '__main__':
    print("==== LANCEMENT DE L ETUDE ODOMETRIE ====")
    reader, topics_type = extrait_rosbag._open_bag_reader(BAG, TOPIC_LOADING)
    data_reader = extrait_rosbag._open_bag_extrait(reader, topics_type, TOPIC_LOADING)

    # 1. Préparer la trajectoire GNSS (Vérité terrain)
    # GNSSTrajectory attend la liste brute des messages GNSS
    print("==== Initialisation Trajectoire GNSS ====")
    gnss_traj = GNSSTrajectory(data_reader[TOPIC_LOADING[1]])

    print("==== Traitement Odométrie (FWSS) ====")
    # On crée des listes simples pour stocker uniquement les nombres
    pose_X = []
    pose_Y = []
    t_history = []
    last_odom = Odometry()  # On garde une trace de l'objet courant

    for i, (ts, msg) in enumerate(data_reader[TOPIC_LOADING[0]]):
        fwss = FourWheelSteeringStamped()
        fwss._conver_MSG(msg)

        if i == 0:
            initial_pos, initial_yaw = gnss_traj.interpolate_pose(ts)
            if initial_pos is None:
                initial_pos = gnss_traj.positions[0]
                initial_yaw = gnss_traj._quat_to_yaw(gnss_traj.quaternions[0])

            last_odom.pose.position.x = initial_pos[0]
            last_odom.pose.position.y = initial_pos[1]
            last_odom.current_yaw = initial_yaw

            # On utilise une copie pour le premier calcul
            data = odometry.FWSS_by_Odometry(fwss, (float(ts), float(ts)), last_odom)
        else:
            # On passe l'objet mis à jour au tour précédent
            data = odometry.FWSS_by_Odometry(fwss, (float(t_history[i - 1]), float(ts)), last_odom)

        # CRUCIAL : On stocke les VALEURS (nombres), pas l'objet entier
        pose_X.append(data.pose.position.x)
        pose_Y.append(data.pose.position.y)
        t_history.append(ts)
        # On ne fait plus de odom.append(data) pour éviter le bug de référence

    # --- PRINTS PROPRES ---
    print(f"Odom Start : ({pose_X[0]:.4f}, {pose_Y[0]:.4f})")
    print(f"Odom End   : ({pose_X[-1]:.4f}, {pose_Y[-1]:.4f})")

    # Pour le GNSS, on prend les points originaux stockés dans l'objet GNSSTrajectory
    gnss_x = gnss_traj.positions[:, 0]
    gnss_y = gnss_traj.positions[:, 1]

    print(f"GNSS Start : ({gnss_x[0]:.4f}, {gnss_y[0]:.4f})")
    print(f"GNSS End   : ({gnss_x[-1]:.4f}, {gnss_y[-1]:.4f})")

    # 4. Affichage
    plt.figure(figsize=(12, 8))
    plt.plot(gnss_x, gnss_y, 'r-', label="GNSS (Vérité terrain)", alpha=0.8)
    #plt.plot(pose_X, pose_Y, 'b-', label="Odométrie (Modèle FWSS)", linewidth=2)

    # Points de départ
    plt.scatter(gnss_x[0], gnss_y[0], color='green', s=100, label="Départ", zorder=5)

    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.title("Comparaison de Trajectoire : Modèle Cinématique vs GNSS")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")

    plt.show()