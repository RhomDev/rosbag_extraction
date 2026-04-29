from PIL.GimpGradientFile import linear

from extraction import extrait_rosbag
import odometry
from optimisation.utils import Odometry, FourWheelSteeringStamped, GNSSTrajectory
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
    print("==== lANCEMENT DE L ETUDE ODOMETRIE ====")
    reader,topics_type = extrait_rosbag._open_bag_reader(BAG, TOPIC_LOADING)
    data_reader = extrait_rosbag._open_bag_extrait(reader, topics_type, TOPIC_LOADING)

    print("==== Traitement des donnée ====")
    odom = []
    pose_X = []
    pose_Y = []
    t = []

    print("==== Traitement fwss ====")
    for i, (ts, msg) in enumerate(data_reader[TOPIC_LOADING[0]]):
        fwss = FourWheelSteeringStamped()
        fwss._conver_MSG(msg)

        if i == 0:
            data = odometry.FWSS_by_Odometry(fwss, (0.0, ts), Odometry())
        else:
            data = odometry.FWSS_by_Odometry(fwss, (t[i-1], ts), odom[i - 1])

        pose_X.append(data.pose.position.x)
        pose_Y.append(data.pose.position.y)
        t.append(ts)
        odom.append(data)

    print("==== Traitement gnss ====")
    gnss = GNSSTrajectory(data_reader[TOPIC_LOADING[1]])

    print("==== Traitement graphique ====")

    temps = gnss.timestamps - gnss.timestamps[0]  # Pour partir de 0s

    import numpy as np
    quat_array = np.array([[q.x, q.y, q.z, q.w] for q in gnss.quaternions])

    q_x = quat_array[:, 0]
    q_y = quat_array[:, 1]
    q_z = quat_array[:, 2]
    q_w = quat_array[:, 3]

    plt.plot(temps, q_x, label="Variation de hauteur (x)")
    plt.plot(temps, q_y, label="Variation de hauteur (y)")
    plt.plot(temps, q_z, label="Variation de hauteur (z)")
    plt.plot(temps, q_w, label="Variation de hauteur (w)")
    plt.xlabel("Temps (s)")
    plt.ylabel("Altitude (m)")
    plt.title("Profil altimétrique en fonction du temps")
    plt.grid(True)
    plt.legend()
    plt.show()

