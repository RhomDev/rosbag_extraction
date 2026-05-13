#!/usr/bin/env bash

# ==============================================================================
# CONFIGURATION DU WORKSPACE
# Doit être sourcé : source setup.sh
# ==============================================================================

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

info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ==============================================================================
# RÈGLE FONDAMENTALE pour les scripts sourcés :
#   Ne JAMAIS utiliser `set -euo pipefail` — ces options s'appliquent au shell
#   interactif parent et le feront crasher sur la moindre erreur.
#   On gère les erreurs manuellement avec || { error "..."; return 1; }
# ==============================================================================
_setup_workspace() {

    # -------------------------------------------------------------------------
    # Répertoire du projet
    # -------------------------------------------------------------------------
    local SCRIPT_DIR
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" \
        || { error "Impossible de résoudre le répertoire du script."; return 1; }

    cd "$SCRIPT_DIR" \
        || { error "Impossible d'accéder à : $SCRIPT_DIR"; return 1; }

    info "=== CONFIGURATION DE L'ENVIRONNEMENT ==="
    info "Projet : $SCRIPT_DIR"

    # -------------------------------------------------------------------------
    # Détection ROS2
    # -------------------------------------------------------------------------
    if [ -z "${ROS_DISTRO:-}" ]; then
        warn "ROS_DISTRO non détecté, recherche dans /opt/ros..."
        if [ -d "/opt/ros" ] && [ -n "$(ls /opt/ros 2>/dev/null)" ]; then
            ROS_DISTRO=$(ls /opt/ros | sort -r | head -n 1)
            info "ROS2 détecté : $ROS_DISTRO"
        else
            error "ROS2 n'est pas installé dans /opt/ros."
            return 1
        fi
    fi

    local ROS_SETUP="/opt/ros/$ROS_DISTRO/setup.bash"
    if [ ! -f "$ROS_SETUP" ]; then
        error "Fichier setup ROS2 introuvable : $ROS_SETUP"
        return 1
    fi

    # On source ROS dans un contexte permissif : les scripts ROS retournent
    # régulièrement des codes non-zero, c'est normal et attendu.
    export AMENT_TRACE_SETUP_FILES=""
    # shellcheck source=/dev/null
    source "$ROS_SETUP"
    info "ROS2 ($ROS_DISTRO) sourcé."

    # -------------------------------------------------------------------------
    # Installation des dépendances ROS2 manquantes via rosdep
    # -------------------------------------------------------------------------
    local found_pkg_xml
    found_pkg_xml=$(find "$SCRIPT_DIR" -maxdepth 3 -name "package.xml" 2>/dev/null | head -n 1)

    if [ -n "$found_pkg_xml" ]; then
        info "package.xml détecté, vérification des dépendances ROS2..."
        if command -v rosdep &>/dev/null; then
            if [ ! -f "/etc/ros/rosdep/sources.list.d/20-default.list" ]; then
                warn "rosdep non initialisé — initialisation (sudo requis)..."
                sudo rosdep init 2>/dev/null || true
                rosdep update --quiet || warn "rosdep update a échoué (non bloquant)."
            fi
            rosdep install --from-paths "$SCRIPT_DIR" --ignore-src -r -y --quiet 2>/dev/null \
                && info "Dépendances ROS2 installées/vérifiées." \
                || warn "rosdep a rencontré des erreurs mineures (non bloquant)."
        else
            warn "rosdep introuvable. Pour l'installer : sudo apt install python3-rosdep"
        fi
    fi

    # -------------------------------------------------------------------------
    # Virtualenv Python
    # --system-site-packages : le venv hérite des packages ROS installés via apt
    # -------------------------------------------------------------------------
    local VENV_PATH="$SCRIPT_DIR/.venv"

    if [ ! -d "$VENV_PATH" ]; then
        info "Création du virtualenv (--system-site-packages)..."
        python3 -m venv --system-site-packages "$VENV_PATH" \
            || { error "Échec de la création du virtualenv."; return 1; }

        # shellcheck source=/dev/null
        source "$VENV_PATH/bin/activate" \
            || { error "Échec d'activation du virtualenv."; return 1; }

        pip install --upgrade pip setuptools wheel --quiet
        if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
            info "Installation de requirements.txt..."
            pip install -r "$SCRIPT_DIR/requirements.txt" --quiet \
                || warn "Certaines dépendances n'ont pas pu être installées."
        fi
        info "Virtualenv créé et activé."
    else
        # shellcheck source=/dev/null
        source "$VENV_PATH/bin/activate" \
            || { error "Échec d'activation du virtualenv."; return 1; }
        info "Virtualenv activé."

        if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
            pip install -r "$SCRIPT_DIR/requirements.txt" --quiet \
                || warn "Mise à jour partielle des dépendances (non bloquant)."
        fi
    fi

    # -------------------------------------------------------------------------
    # Configuration PYTHONPATH
    # -------------------------------------------------------------------------
    local PY_VER
    PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    local ROS_SITE_PACKAGES="/opt/ros/$ROS_DISTRO/lib/python${PY_VER}/site-packages"

    export PYTHONPATH="$SCRIPT_DIR/src:$ROS_SITE_PACKAGES:${PYTHONPATH:-}"

    # Déduplication (évite les doublons si on re-source plusieurs fois)
    PYTHONPATH=$(python3 -c "
seen, parts = set(), []
for p in '${PYTHONPATH}'.split(':'):
    if p and p not in seen:
        seen.add(p)
        parts.append(p)
print(':'.join(parts))
")
    export PYTHONPATH

    # -------------------------------------------------------------------------
    # Finalisation
    # -------------------------------------------------------------------------
    local TARGET_DIR="$SCRIPT_DIR/src/rosbag_extraite"
    if [ -d "$TARGET_DIR" ]; then
        cd "$TARGET_DIR"
    fi

    info "✅ Configuration terminée avec succès."
    info "   Python : $(which python3) — $(python3 --version)"
    info "   ROS2   : $ROS_DISTRO"
    info "   Venv   : $VENV_PATH"
    info "   Dossier: $(pwd)"
}

# ==============================================================================
# Exécution
# ==============================================================================
if _setup_workspace; then
    : # succès
else
    error "❌ La configuration a échoué. Consultez les messages ci-dessus."
fi

# Nettoyage de la fonction helper (ne doit pas rester dans le shell)
unset -f _setup_workspace