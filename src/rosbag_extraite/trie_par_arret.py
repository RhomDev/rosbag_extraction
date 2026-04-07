import traitement_img as ti
import traitement_vitesse as tv
import argparse

import os
import glob

import shutil
from tqdm import tqdm

reduction = 1

def timestamp_stop_periods(data_vel):
    periods = []          # Liste finale de périodes
    current_period = []   # Sous-liste pour la période en cours

    for time, speed in zip(data_vel[0], data_vel[1]):
        if speed == 0.0:
            current_period.append(time)  # On ajoute le timestamp à la période en cours
        else:
            if current_period:           # Si on quitte une période d'arrêt
                periods.append(current_period)  # On sauvegarde la période
                current_period = []      # On réinitialise pour la prochaine période

    # On ajoute la dernière période si elle se termine à la fin
    if current_period:
        periods.append(current_period)

    return periods


def clean_photos(clean, base_dir):
    for i, arret in enumerate(tqdm(clean, desc="Déplacement des batches", unit="batch")):
        folder_path = os.path.join(base_dir, f"batch_{i}")
        os.makedirs(folder_path, exist_ok=True)

        for name in arret:
            img_file = f"image_{name}.jpg"
            img = os.path.join(base_dir, img_file)
            folder_path_img = os.path.join(folder_path, img_file)

            # Vérifie que le fichier existe
            if not os.path.exists(img):
                continue  # passe au suivant

            # Déplace le fichier
            shutil.move(img, folder_path_img)

    remaining_files = glob.glob(os.path.join(base_dir, "image_*.jpg"))
    for fichier in tqdm(remaining_files, desc="Suppression des images restantes", unit="fichier"):
        os.remove(fichier)


def main(bag_path, topic_vel,mode):
    config_tv = tv.configuration(bag_path, topic_vel)
    config_ti = ti.configuration(bag_path)

    print("=== TRAITEMENT DES IMAGES ===")
    ti.main(bag_path)

    print("=== TRAITEMENT DE LA VITESSE ===")
    data_vel = tv.extration(config_tv[0], config_tv[1],topic_vel, mode)


    print("=== RECHERCHE DES ARRETS ===")
    time_stop = timestamp_stop_periods(data_vel)

    clean = []
    reduction = 10  # Exemple, adapte selon ton code

    # Boucle avec barre de progression
    for arret in tqdm(time_stop, desc="Traitement des arrets", unit="arret"):
        size = len(arret)
        red = int(size * (reduction / 100))
        if len(arret[red:-red]) < 100:
            print("coucou")
            continue
        clean.append(arret[red:-red])



    print("=== NETOYAGE DES PHOTOS ===")
    for name_dir in config_ti[-1]:
        clean_photos(clean, name_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Traitement des ROS2 bag")
    parser.add_argument("--bag", type=str, required=True,
                        help="Chemin du rosbag")

    parser.add_argument("--topic_vel", type=str, required=True,
                        help="Nom du topic vitesse")

    parser.add_argument("--mode_vel", type=int, default=1,
                        help="mode de la vitesse (0 : odometry ; 1 : wheel_steering)")

    args = parser.parse_args()
    main(args.bag, args.topic_vel, args.mode_vel)