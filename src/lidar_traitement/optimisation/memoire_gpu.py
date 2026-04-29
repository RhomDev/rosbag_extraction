import gc
import numpy as np
from pathlib import Path
from typing import Optional


# ═════════════════════════════════════════════════════════════════════════════
# GPU DETECTION — fallback automatique CUDA → MPS → CPU
# ═════════════════════════════════════════════════════════════════════════════

_DEVICE = "cpu"
_torch_device = None

def init():
    """Initialisation du package."""
    global _DEVICE
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() > 0:
            _DEVICE = "cuda_cupy"
            # Limite mémoire GPU à 80 % pour éviter OOM GPU
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(fraction=0.8)
    except Exception:
        pass

    if _DEVICE == "cpu":
        try:
            import torch
            if torch.cuda.is_available():
                _DEVICE = "torch_cuda"
                _torch_device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                _DEVICE = "torch_mps"
                _torch_device = torch.device("mps")
        except Exception:
            pass

    print(f"[INFO] Accélération matérielle : {_DEVICE}")


# ═════════════════════════════════════════════════════════════════════════════
# UTILITAIRES SYSTÈME — mémoire, GC, checkpoints
# ═════════════════════════════════════════════════════════════════════════════

def _mem_used_gb() -> float:
    """RSS du processus courant en Go (sans dépendance psutil)."""
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF)
        return ru.ru_maxrss / (1024 ** 2)        # macOS : octets → Go
    except Exception:
        return 0.0

def get_current_ram_gb() -> float:
    """Lit la RAM réellement utilisée à cet instant (Linux)."""
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    # La ligne ressemble à : VmRSS:  123456 kB
                    kb = float(line.split()[1])
                    return kb / (1024 ** 2) # Ko -> Go
    except:
        return 0.0

def _force_gc():
    """Garbage-collect + libère les pools GPU si disponibles."""
    gc.collect()
    if _DEVICE == "cuda_cupy":
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    elif _DEVICE.startswith("torch"):
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _save_checkpoint(array: np.ndarray, path: Path, label: str):
    """Sauvegarde .npy atomique (écriture tmp puis rename)."""
    tmp = path.with_suffix(".tmp.npy")
    np.save(str(tmp), array)
    tmp.rename(path)
    print(f"  [CKPT] {label} → {path.name}  ({len(array):,} pts, "
          f"{array.nbytes / 1e6:.0f} Mo)")


def _load_checkpoint(path: Path) -> Optional[np.ndarray]:
    if path.exists():
        arr = np.load(str(path))
        print(f"  [CKPT] Reprise depuis {path.name}  ({len(arr):,} pts)")
        return arr
    return None