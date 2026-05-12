import os
import argparse
import datetime

import rosbag2_py
import matplotlib.pyplot as plt

from tqdm import tqdm
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

import extracte_setup


# =========================
# CONFIGURATION
# =========================
def configuration(type_map, topic_name: str):
    return get_message(type_map[topic_name])


# =========================
# EXTRACTION
# =========================
def extraction(reader, msg_type, topic_name, mode, use_nanoseconds=True):

    timestamps = []
    speeds = []

    while reader.has_next():
        topic, data, t = reader.read_next()

        if topic != topic_name:
            continue

        msg = deserialize_message(data, msg_type)

        # timestamp
        if use_nanoseconds:
            ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        else:
            ts = msg.header.stamp.sec

        # speed computation
        if mode == 0:
            vx = msg.twist.twist.linear.x
            vy = msg.twist.twist.linear.y
            vz = msg.twist.twist.linear.z
            speed = (vx**2 + vy**2 + vz**2) ** 0.5

        elif mode == 1:
            speed = msg.data.speed

        else:
            raise ValueError("Mode inconnu (0: odometry, 1: wheel steering)")

        timestamps.append(ts)
        speeds.append(speed)

    if not timestamps:
        return [], [], []

    t0 = timestamps[0]
    timestamps_norm = [t - t0 for t in timestamps]

    return timestamps_norm, timestamps, speeds


# =========================
# PLOT
# =========================
def plot_speed(timestamps, speeds):
    plt.figure()
    plt.plot(timestamps, speeds, label="Vitesse")

    plt.xlabel("Temps (s)")
    plt.ylabel("Vitesse (m/s)")
    plt.title("Évolution de la vitesse")
    plt.grid(True)
    plt.legend()


def save_plot():
    now = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

    out_dir = "../../out/save_vitesse_odometry"
    os.makedirs(out_dir, exist_ok=True)

    plt.savefig(os.path.join(out_dir, f"vitesse_{now}_odometry.png"))


# =========================
# MAIN
# =========================
def main(bag, topic_name, mode):

    steps = ["Configuration", "Extraction", "Plot", "Sauvegarde"]

    for step in tqdm(steps, desc="Pipeline vitesse", ncols=80):

        if step == "Configuration":
            reader, topic_names, type_map = extracte_setup.configuration(bag)
            msg_type = configuration(type_map, topic_name)

        elif step == "Extraction":
            t_norm, t_raw, speeds = extraction(reader, msg_type, topic_name, mode)

        elif step == "Plot":
            plot_speed(t_norm, speeds)

        elif step == "Sauvegarde":
            save_plot()


# =========================
# ENTRYPOINT
# =========================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extraction vitesse ROS2 bag")

    parser.add_argument("--bag", type=str, required=True)

    parser.add_argument(
        "--topic",
        type=str,
        default="/ez10_gen1/received_raw_four_wheel_steering"
    )

    parser.add_argument("--mode", type=int, default=1)

    args = parser.parse_args()

    main(args.bag, args.topic, args.mode)