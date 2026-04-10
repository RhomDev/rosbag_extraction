import os
import glob
import shutil
import argparse

from tqdm import tqdm

import traitement_img as ti
import traitement_vitesse as tv
import extracte_setup


# =========================
# DETECTION DES ARRÊTS
# =========================
def timestamp_stop_periods(data_vel):
    periods = []
    current = []

    for time, speed in zip(data_vel[1], data_vel[2]):
        if speed == 0.0:
            current.append(time)
        else:
            if current:
                periods.append(current)
                current = []

    if current:
        periods.append(current)

    return periods


# =========================
# FILTRAGE DES PÉRIODES
# =========================
def filter_periods(periods, min_len=100, reduction_percent=10):
    filtered = []

    for p in periods:
        size = len(p)

        if size == 0:
            continue

        trim = int(size * (reduction_percent / 100))

        # évite slicing négatif trop agressif
        if size > 2 * trim:
            p = p[trim:-trim]

        if len(p) >= min_len:
            filtered.append(p)

    return filtered


# =========================
# CLEAN IMAGES PAR BATCH
# =========================
def clean_batches(periods, base_dir):
    for i, batch in enumerate(tqdm(periods, desc="Déplacement des batches")):

        folder = os.path.join(base_dir, f"batch_{i}")
        os.makedirs(folder, exist_ok=True)

        for ts in batch:
            filename = f"image_{ts}.jpg"
            src = os.path.join(base_dir, filename)
            dst = os.path.join(folder, filename)

            if os.path.exists(src):
                shutil.move(src, dst)

    # supprimer les images restantes
    remaining = glob.glob(os.path.join(base_dir, "image_*.jpg"))

    for f in tqdm(remaining, desc="Suppression des images restantes"):
        os.remove(f)


# =========================
# MAIN PIPELINE
# =========================
def main(bag, topic_vel, mode_vel):

    reader, topic_names, type_map = extracte_setup.configuration(bag)

    config_tv = tv.configuration(type_map, topic_vel)
    config_ti = ti.configuration(type_map, topic_names)

    print("=== TRAITEMENT IMAGES ===")
    ti.main(bag, True)

    print("=== TRAITEMENT VITESSE ===")
    data_vel = tv.extraction(reader, config_tv, topic_vel, mode_vel, False)

    print("=== DETECTION ARRÊTS ===")
    stop_periods = timestamp_stop_periods(data_vel)

    filtered = filter_periods(stop_periods)

    print("=== ORGANISATION IMAGES ===")

    # config_ti[-1] = save_dirs
    for base_dir in config_ti[-1]:
        clean_batches(filtered, base_dir)


# =========================
# ENTRYPOINT
# =========================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Traitement ROS2 bag")

    parser.add_argument("--bag", type=str, required=True)

    parser.add_argument(
        "--topic_vel",
        type=str,
        default="/ez10_gen1/received_raw_four_wheel_steering"
    )

    parser.add_argument("--mode", type=int, default=1)

    args = parser.parse_args()

    main(args.bag, args.topic_vel, args.mode)