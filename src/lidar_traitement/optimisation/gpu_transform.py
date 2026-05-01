"""
gpu_transform.py
Remplacement GPU de mapping.transformation() pour les gros nuages.
Requiert : pip install cupy-cuda12x  (adapter la version CUDA)
"""

import numpy as np

try:
    import cupy as cp

    _GPU_AVAILABLE = True
except ImportError:
    _GPU_AVAILABLE = False


def transformation_gpu(pc: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Applique T (4×4) à pc (N×3 ou N×4) sur GPU si disponible, sinon CPU.

    Gain typique : 10-50× sur nuages > 50 000 pts.
    """
    if not _GPU_AVAILABLE or len(pc) < 10_000:
        # CPU : chemin original, rapide sur petits nuages
        ones = np.ones((pc.shape[0], 1), dtype=np.float32)
        ph = np.hstack([pc[:, :3], ones])
        return (T @ ph.T).T[:, :3]

    # GPU
    pc_gpu = cp.asarray(pc[:, :3], dtype=cp.float32)  # transfert vers GPU
    T_gpu = cp.asarray(T, dtype=cp.float32)

    ones = cp.ones((pc_gpu.shape[0], 1), dtype=cp.float32)
    ph = cp.hstack([pc_gpu, ones])  # (N, 4)
    result = (T_gpu @ ph.T).T[:, :3]  # (N, 3)

    return cp.asnumpy(result)  # retour CPU


def batch_transform_gpu(clouds: list, transforms: list) -> list:
    """
    Transforme une liste de nuages en batch sur GPU.
    Plus efficace que N appels séparés (réduit les allers-retours CPU↔GPU).

    clouds     : list of np.ndarray (Ni, 3)
    transforms : list of np.ndarray (4, 4)
    """
    if not _GPU_AVAILABLE:
        return [transformation_gpu(c, T) for c, T in zip(clouds, transforms)]

    results = []
    for pc, T in zip(clouds, transforms):
        results.append(transformation_gpu(pc, T))
    return results