import rosbag2_py
import extracte_setup

import os
import argparse

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import matplotlib.pyplot as plt

# === CONFIG ===
def configuration(type_map, topic_name:str):
    msg_type = get_message(type_map[topic_name])
    return msg_type

# === EXTRACTION ===
def extration(reader, msg_type, topic_name, mode):
    timestamps = []
    velocities = []

    while reader.has_next():
        topic, data, t = reader.read_next()

        if topic == topic_name:
            msg = deserialize_message(data, msg_type)

            # temps
            ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            if mode == 0:
                # Odometry
                # vitesse (norme du vecteur linéaire)
                vx = msg.twist.twist.linear.x
                vy = msg.twist.twist.linear.y
                vz = msg.twist.twist.linear.z

                speed = (vx**2 + vy**2 + vz**2)**0.5
            elif mode == 1:
                speed = msg.data.speed

            timestamps.append(ts)
            velocities.append(speed)

    # === NORMALISATION TEMPS ===
    t0 = timestamps[0]
    data_time = timestamps
    timestamps = [t - t0 for t in timestamps]
    return timestamps,data_time, velocities

# === PLOT ===
def printing(timestamps:list, velocities:list):
    plt.figure()
    plt.plot(timestamps, velocities, label="Vitesse")
    plt.xlabel("Temps (s)")
    plt.ylabel("Vitesse (m/s)")
    plt.title("Evolution de la vitesse")
    plt.grid()
    plt.legend()
    plt.show()

def saving():
    import datetime
    maintenant = datetime.datetime.now()
    format_specifique = maintenant.strftime("%d_%m_%Y_%H_%M_%S")
    os.makedirs("../../out/save", exist_ok=True)
    plt.savefig(f"../../out/save/vitesse_{format_specifique}.png")

def main(reader,type_map, topic_name, mode):
    global data, msg_type, timestamps, velocities
    from tqdm import tqdm
    import time

    steps = ["Configuration", "Extraction", "Plot", "Sauvegarde"]
    for step in tqdm(steps, desc="Progression Traitement Vitesse", ncols=80):
        if step == "Configuration":
            msg_type = configuration(type_map,topic_name)
            time.sleep(0.3)
        elif step == "Extraction":
            timestamps, data_time, velocities = extration(reader, msg_type,topic_name, mode)
            time.sleep(0.3)
        elif step == "Plot":
            printing(timestamps, velocities)
            time.sleep(0.3)
        elif step == "Sauvegarde":
            saving()
            time.sleep(0.3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Traitement de vitesse des ROS2 bag")
    #default="../../rosbag/ez10_gen1_sensors_2026-04-02-16-45-39"
    parser.add_argument("--bag", type=str, required=True,
                        help="Chemin du rosbag")
    parser.add_argument("--topic", type=str, default="/ez10_gen1/received_raw_four_wheel_steering",
                        help="Nom des topics")
    parser.add_argument("--mode", type=int,
                        default="1",
                        help="mode de la vitesse (0 : odometry ; 1 : wheel_steering)")
    args = parser.parse_args()

    reader, topic_names, type_map = extracte_setup.configuration(args.bag)

    main(reader,type_map, args.topic, args.mode)