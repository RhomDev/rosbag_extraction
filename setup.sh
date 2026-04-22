#!/usr/bin/env bash

set -euo pipefail  # Arrêt sur erreur, variable non définie, échec pipeline

# =========================
# Couleurs pour les messages
# =========================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# =========================
# Répertoire du script
# =========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || error "Impossible d'accéder au répertoire du script"

info "=== SETUP ENV ==="
info "Répertoire de travail : $SCRIPT_DIR"

# =========================
# Vérification de Python
# =========================
if ! command -v python3 &> /dev/null; then
    error "python3 n'est pas installé"
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Python détecté : version $PY_VERSION"

# =========================
# Détection ROS2
# =========================
if [ -z "${ROS_DISTRO:-}" ]; then
    warn "ROS_DISTRO non défini → tentative de détection..."
    if [ -d "/opt/ros" ]; then
        ROS_DISTRO=$(ls /opt/ros | head -n 1)
        [ -z "$ROS_DISTRO" ] && error "Aucune installation ROS2 trouvée dans /opt/ros"
        info "ROS2 détecté : $ROS_DISTRO"
    else
        error "Dossier /opt/ros introuvable. ROS2 est-il installé ?"
    fi
fi

ROS_SETUP="/opt/ros/$ROS_DISTRO/setup.bash"
if [ ! -f "$ROS_SETUP" ]; then
    error "Fichier setup.bash introuvable : $ROS_SETUP"
fi

# Prévention ROS2
export AMENT_TRACE_SETUP_FILES=""
export AMENT_PYTHON_EXECUTABLE="$(which python3)"
export COLCON_TRACE=""
export ROS_PYTHON_VERSION="3"

# Sourcing ROS2
source "$ROS_SETUP"
info "ROS2 sourcé depuis $ROS_SETUP"

# =========================
# Virtualenv Python
# =========================
VENV_DIR="${VENV_DIR:-.venv}"
VENV_PATH="$SCRIPT_DIR/$VENV_DIR"

create_venv() {
    info "Création du virtualenv dans $VENV_PATH"
    python3 -m venv "$VENV_PATH" || error "Échec de création du virtualenv"
    source "$VENV_PATH/bin/activate"

    info "Mise à jour de pip"
    pip install --upgrade pip setuptools wheel

    if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
        info "Installation des dépendances"
        pip install -r requirements.txt || warn "Certaines dépendances ont échoué"
    else
        warn "requirements.txt absent"
    fi
}

if [ ! -d "$VENV_PATH" ]; then
    create_venv
else
    source "$VENV_PATH/bin/activate"
    if [ "${RECREATE_VENV:-0}" = "1" ]; then
        warn "RECREATE_VENV=1 → recréation venv"
        rm -rf "$VENV_PATH"
        create_venv
    fi
fi

# =========================
# PYTHONPATH (IMPORTANT)
# =========================
info "Configuration PYTHONPATH..."

# priorité projet
export PYTHONPATH="$SCRIPT_DIR/src"

# ajout ROS2 si dispo
ROS_PY="/opt/ros/$ROS_DISTRO/lib/python$PY_VERSION/site-packages"

if [ -d "$ROS_PY" ]; then
    export PYTHONPATH="$PYTHONPATH:$ROS_PY"
    info "Ajout ROS2 PYTHONPATH"
else
    warn "ROS2 python path introuvable : $ROS_PY"
fi

# =========================
# Aller dans le projet
# =========================
TARGET_DIR="$SCRIPT_DIR/src/rosbag_extraite"

if [ -d "$TARGET_DIR" ]; then
    cd "$TARGET_DIR" || error "Impossible d'accéder à $TARGET_DIR"
    info "Répertoire courant : $(pwd)"
else
    warn "Dossier rosbag_extraite introuvable"
fi

# =========================
# Résumé
# =========================
info "✅ Environnement prêt"
info "ROS_DISTRO     = $ROS_DISTRO"
info "Python version = $(python3 --version)"
info "Virtualenv     = $VIRTUAL_ENV"
info "PYTHONPATH     = $PYTHONPATH"

# =========================
# Shell interactif (SAFE)
# =========================
if [ "${START_SHELL:-0}" = "1" ]; then
    info "Shell interactif"
    stty sane 2>/dev/null || true
    "${SHELL:-bash}"
fi