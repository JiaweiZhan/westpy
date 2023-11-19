import numpy as np

def epsilon_diag(pdep_g: np.ndarray, # [npdep, nmill]
                 ev: np.ndarray,     # [npdep]
                 mill: np.ndarray,   # [nmill, 3]
                 ):
    g_norm = np.linalg.norm(mill, axis=-1, keepdims=False)
    eps_diag = 1 + np.sum(((ev) / (1 - ev))[:, None] * np.abs(pdep_g) ** 2, axis=0)

    g_norm, eps_diag = zip(*(sorted(zip(g_norm, eps_diag), key=lambda x: x[0])))
    return g_norm, eps_diag
