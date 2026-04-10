#!/usr/bin/env bash

set -e  # stop si erreur

echo "=== SETUP ENV ==="

# =========================
# DETECTION ROS2
# =========================
if [ -z "$ROS_DISTRO" ]; then
    echo "ROS_DISTRO non défini → tentative de détection..."

    if [ -d "/opt/ros" ]; then
        ROS_DISTRO=$(ls /opt/ros | head -n 1)
        echo "ROS détecté: $ROS_DISTRO"
    else
        echo "❌ ROS2 non trouvé dans /opt/ros"
        exit 1
    fi
fi

# Source ROS2
source /opt/ros/$ROS_DISTRO/setup.bash

# =========================
# WORKSPACE (optionnel)
# =========================
WS_PATH=~/Documents/stage/ropradeau_ws_extrait

if [ -f "$WS_PATH/install/setup.bash" ]; then
    echo "Sourcing workspace..."
    source "$WS_PATH/install/setup.bash"
fi

# =========================
# VENV
# =========================
VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Création du virtualenv..."
    python3 -m venv $VENV_DIR

    source $VENV_DIR/bin/activate

    echo "Upgrade pip..."
    pip install --upgrade pip

    # Optionnel : requirements.txt
    if [ -f "requirements.txt" ]; then
        echo "Installation requirements..."
        pip install -r requirements.txt
    fi
else
    echo "Activation du virtualenv..."
    source $VENV_DIR/bin/activate
fi

# =========================
# PYTHONPATH CLEAN (auto)
# =========================
echo "Configuration PYTHONPATH..."

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

ROS_PY="/opt/ros/$ROS_DISTRO/lib/python$PY_VER/site-packages"
WS_PY="$WS_PATH/install/lib/python$PY_VER/site-packages"

if [ -d "$ROS_PY" ]; then
    export PYTHONPATH=$PYTHONPATH:$ROS_PY
fi

if [ -d "$WS_PY" ]; then
    export PYTHONPATH=$PYTHONPATH:$WS_PY
fi

# =========================
# WORKDIR
# =========================
cd src/rosbag_extraite || exit

echo "✅ ENV prêt"
echo "ROS_DISTRO=$ROS_DISTRO"
echo "Python=$(python --version)"