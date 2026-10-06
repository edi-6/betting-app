"""Smoke from the campfire and the chimneys (and the campfire's flames): deterministic puffs, a pure function of the
video's time, so any frame can be rendered on its own."""
import numpy as np


def _h(k, s, salt):
    x = np.sin(k * 12.9898 + s * 78.233 + salt * 37.719) * 43758.5453
    return x - np.floor(x)


def smoke(t, sources, rate=4.0, life=5.0):
    """sources: (n, 4) x, y, z, strength. Returns puffs (m, 8): pos3, size, alpha, fire, variant, age."""
    out = []
    for si, (x, y, z, st) in enumerate(np.asarray(sources, float)):
        r = rate * (0.6 + 0.4 * st)
        k1 = int(np.floor(t * r))
        k0 = int(np.floor((t - life) * r)) + 1
        ks = np.arange(k0, k1 + 1, dtype=float)
        if not len(ks):
            continue
        age = t - ks / r
        a = age / life
        jx, jy = _h(ks, si, 1) - 0.5, _h(ks, si, 2) - 0.5
        # rises, slowing down; the wind takes it east-south-east; it wobbles and spreads
        rise = 2.4 * life * (1 - np.exp(-age / 2.2)) * (0.8 + 0.4 * _h(ks, si, 3))
        wx = 0.42 * age + 0.25 * np.sin(age * 1.3 + ks) + jx * 0.2
        wy = -0.12 * age + 0.2 * np.cos(age * 1.1 + ks * 0.7) + jy * 0.2
        size = (0.22 + 0.9 * a ** 0.7) * (0.6 + 0.4 * st)
        alpha = np.clip(age / 0.35, 0, 1) * np.clip((1 - a) / 0.45, 0, 1) * 0.42 * (0.55 + 0.45 * st)
        p = np.zeros((len(ks), 8))
        p[:, 0] = x + wx
        p[:, 1] = y + wy
        p[:, 2] = z + 0.15 + rise
        p[:, 3] = size
        p[:, 4] = alpha
        p[:, 5] = 0.0
        p[:, 6] = np.floor(_h(ks, si, 4) * 4)
        p[:, 7] = age
        out.append(p)
    return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 8), np.float32)


def flames(t, x, y, z):
    """A few flickering flame licks over the campfire."""
    n = 4
    p = np.zeros((n, 8))
    for k in range(n):
        ph = t * (7.0 + k * 1.3) + k * 2.1
        p[k, 0] = x + 0.18 * np.sin(k * 2.4) + 0.03 * np.sin(ph)
        p[k, 1] = y + 0.18 * np.cos(k * 2.4) + 0.03 * np.cos(ph * 1.3)
        p[k, 2] = z + 0.22 + 0.08 * (0.5 + 0.5 * np.sin(ph))
        p[k, 3] = 0.34 + 0.08 * np.sin(ph * 1.7)
        p[k, 4] = 0.55
        p[k, 5] = 0.75 + 0.2 * np.sin(ph * 2.3)
        p[k, 6] = k % 4
        p[k, 7] = 0.2
    return p.astype(np.float32)
