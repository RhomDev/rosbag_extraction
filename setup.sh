#!/usr/bin/env bash

# ==============================================================================
# CONFIGURATION DU SCRIPT
# ==============================================================================
# On utilise set -e uniquement pendant l'exécution du script.
# On le désactivera à la fin (set +e) pour ne pas casser le shell interactif.
set -e

# Vérification que le script est bien sourcé
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo -e "\033[0;31m[ERROR]\033[0m Ce script doit être sourcé : source setup.sh"
    exit 1
fi

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
# Pour un script sourcé, on utilise return au lieu d'exit pour ne pas fermer le terminal
error() { echo -e "${RED}[ERROR]${NC} $1"; return 1; }

# =========================
# Répertoire du projet
# =========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || error "Impossible d'accéder à $SCRIPT_DIR"

info "=== CONFIGURATION DE L'ENVIRONNEMENT ==="
info "Projet : $SCRIPT_DIR"

# =========================
# Détection ROS2
# =========================
if [ -z "${ROS_DISTRO:-}" ]; then
    warn "ROS_DISTRO non détecté, recherche dans /opt/ros..."
    if [ -d "/opt/ros" ]; then
        # On prend la version la plus récente si plusieurs existent
        ROS_DISTRO=$(ls /opt/ros | sort -r | head -n 1)
        info "ROS2 détecté : $ROS_DISTRO"
    else
        error "ROS2 n'est pas installé dans /opt/ros."
    fi
fi

ROS_SETUP="/opt/ros/$ROS_DISTRO/setup.bash"
if [ -f "$ROS_SETUP" ]; then
    # On évite les traces inutiles lors du sourcing
    export AMENT_TRACE_SETUP_FILES=""
    source "$ROS_SETUP"
    info "ROS2 ($ROS_DISTRO) sourcé."
else
    error "Fichier setup ROS2 introuvable."
fi

# =========================
# Virtualenv Python
# =========================
VENV_DIR=".venv"
VENV_PATH="$SCRIPT_DIR/$VENV_DIR"

if [ ! -d "$VENV_PATH" ]; then
    info "Création du virtualenv..."
    python3 -m venv "$VENV_PATH" || error "Échec création venv"
    source "$VENV_PATH/bin/activate"
    pip install --upgrade pip setuptools wheel
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    fi
else
    source "$VENV_PATH/bin/activate"
    info "Virtualenv activé."
fi

# =========================
# Configuration PYTHONPATH
# =========================
# On construit le PYTHONPATH intelligemment :
# 1. Tes sources projet (priorité haute)
# 2. Le site-packages de ROS2 (pour importer rclpy, etc.)
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
ROS_SITE_PACKAGES="/opt/ros/$ROS_DISTRO/lib/python${PY_VER}/site-packages"

export PYTHONPATH="$SCRIPT_DIR/src:$ROS_SITE_PACKAGES:${PYTHONPATH:-}"

# =========================
# Finalisation
# =========================
# Aller dans le dossier de travail si spécifié
TARGET_DIR="$SCRIPT_DIR/src/rosbag_extraite"
if [ -d "$TARGET_DIR" ]; then
    cd "$TARGET_DIR"
fi

info "✅ Configuration terminée avec succès."
info "Python : $(which python3)"
info "Dossier : $(pwd)"

# ------------------------------------------------------------------------------
# CRUCIAL : Désactivation du mode strict pour le shell interactif
# C'est ce qui empêche le terminal de se fermer lors d'un TAB (auto-complete)
# ------------------------------------------------------------------------------
set +e
set +u
set +o pipefail