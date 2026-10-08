"""From the simulation's cells to what the water shader draws: the surface height carried a cell or two into the dry
ground (pressed down under it, so the terrain's own depth makes a clean waterline), the foam, the crest (steep,
raised water, where the low sun shines through), and the current."""
import numpy as np

SEA = 0.0


def _max_nb(a):
    p = np.pad(a, 1, mode='constant', constant_values=-np.inf)
    return np.maximum(np.maximum(p[1:-1, 2:], p[1:-1, :-2]), np.maximum(p[2:, 1:-1], p[:-2, 1:-1]))


def surface(eta, h, b, foam, u, v, dx, wet_h=0.03):
    wet = h > wet_h
    E = np.where(wet, eta, -np.inf)
    nb = _max_nb(E)
    edge = ~wet & np.isfinite(nb)
    eta_d = np.where(wet, eta, np.where(edge, np.minimum(nb, b - 0.05), b - 2.0))
    h_d = np.where(wet, h, np.where(edge, 0.3, 0.0))
    gy, gx = np.gradient(eta_d, dx)
    slope = np.hypot(gx, gy) * wet
    crest = np.clip(slope / 0.7, 0.0, 1.0) * np.clip((eta - SEA) / 2.5, 0.0, 1.0) * wet
    state = np.stack([eta_d, h_d, foam * wet, crest], -1).astype(np.float32)
    vel = np.stack([u * wet, v * wet], -1).astype(np.float32)
    return state, vel
