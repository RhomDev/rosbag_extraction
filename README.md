# 🧠 ROS2 Bag Extraction Toolkit

Toolkit Python pour extraire, traiter et organiser des données issues de rosbag2 (ROS2).

Ce projet permet de manipuler plusieurs types de capteurs :

- 📷 Images multi-caméras  
- 🚗 Vitesse véhicule  
- ☁️ LiDAR (PointCloud2)  
- 🛑 Détection d’arrêts + tri automatique des images  

---

## 🚀 Fonctionnalités

### 📷 Extraction d’images
- Détection automatique des topics `/image_raw`
- Export en `.jpg`
- Organisation par caméra
- Synchronisation inter-caméras (`--clean`)

---

### 🚗 Analyse de vitesse
- Extraction depuis :
  - Odometry (`twist`)
  - Wheel steering
- Calcul de vitesse scalaire
- Visualisation avec graphique temporel
- Export en image (`.png`)

---

### ☁️ Traitement LiDAR
- Lecture de `PointCloud2`
- Filtrage angulaire et distance
- Transformation (translation + rotation)
- Projection en image 2D
- Amélioration visuelle (denoise, contraste, colormap)
- Génération de vidéo `.mp4`

---

### 🛑 Détection d’arrêts (feature avancée)
- Détection automatique des périodes à vitesse nulle
- Filtrage des périodes (durée minimale)
- Découpage des images en **batches d’arrêt**
- Organisation automatique des datasets

---

## 📦 Installation

```bash
git clone https://github.com/RhomDev/rosbag_extraction.git
cd rosbag_extraction
source setup.sh
```


