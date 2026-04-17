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

# Chemin du setup.bash système
ROS_SETUP="/opt/ros/$ROS_DISTRO/setup.bash"
if [ ! -f "$ROS_SETUP" ]; then
    error "Fichier setup.bash introuvable : $ROS_SETUP"
fi

# Prévention des erreurs "unbound variable" dans les scripts ROS2
export AMENT_TRACE_SETUP_FILES=""
export AMENT_PYTHON_EXECUTABLE="$(which python3)"
export COLCON_TRACE=""
export ROS_PYTHON_VERSION="3"

# Sourçage de ROS2
source "$ROS_SETUP"
info "ROS2 sourcé depuis $ROS_SETUP"

# =========================
# Environnement virtuel Python
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
        info "Installation des dépendances depuis requirements.txt"
        pip install -r requirements.txt || warn "Certaines dépendances n'ont pas pu être installées"
    else
        warn "Fichier requirements.txt absent, aucune dépendance automatique"
    fi
}

if [ ! -d "$VENV_PATH" ]; then
    create_venv
else
    info "Activation du virtualenv existant : $VENV_PATH"
    source "$VENV_PATH/bin/activate"
    if [ "${RECREATE_VENV:-0}" = "1" ]; then
        warn "RECREATE_VENV=1 : suppression et recréation du virtualenv"
        rm -rf "$VENV_PATH"
        create_venv
    fi
fi

# =========================
# Configuration PYTHONPATH (uniquement ROS2 système)
# =========================
info "Configuration PYTHONPATH..."
ROS_PY="/opt/ros/$ROS_DISTRO/lib/python$PY_VERSION/site-packages"

if [ -d "$ROS_PY" ]; then
    export PYTHONPATH="$ROS_PY${PYTHONPATH:+:$PYTHONPATH}"
    info "Ajout de $ROS_PY au PYTHONPATH"
else
    warn "Chemin ROS2 introuvable : $ROS_PY"
fi

# Suppression des doublons éventuels (simple nettoyage)
clean_path() {
    echo -n "$1" | awk -v RS=':' -v ORS=':' '!a[$0]++' | sed 's/:$//'
}
export PYTHONPATH=$(clean_path "$PYTHONPATH")

# =========================
# Aller dans le dossier src/rosbag_extraite (si existant)
# =========================
TARGET_DIR="$SCRIPT_DIR/src/rosbag_extraite"
if [ -d "$TARGET_DIR" ]; then
    cd "$TARGET_DIR" || error "Impossible d'accéder à $TARGET_DIR"
    info "Répertoire de travail actuel : $(pwd)"
else
    warn "Dossier $TARGET_DIR introuvable, reste dans $SCRIPT_DIR"
fi

# =========================
# Résumé final
# =========================
info "✅ Environnement prêt"
info "ROS_DISTRO     = $ROS_DISTRO"
info "Python version = $(python --version 2>&1)"
info "Virtualenv     = $VIRTUAL_ENV"
info "PYTHONPATH     = $PYTHONPATH"

# =========================
# Option : lancer un shell interactif
# =========================
if [ "${START_SHELL:-0}" = "1" ]; then
    info "Lancement d'un shell interactif (tapez exit pour revenir)"
    exec "${SHELL:-bash}"
fi