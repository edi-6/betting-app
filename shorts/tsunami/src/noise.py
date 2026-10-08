"""Vectorised gradient noise helpers (Perlin style) used for textures, sky and terrain."""
import numpy as np


def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def perlin2d(x, y, seed=0, period=None):
    """Classic 2D Perlin noise evaluated at arrays x, y. Returns values in about [-1, 1].

    If `period` is given the noise tiles with that period (in lattice units) on both axes.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(256)
    perm = np.concatenate([perm, perm])
    ang = rng.random(256) * 2 * np.pi
    gx, gy = np.cos(ang), np.sin(ang)

    xi = np.floor(x).astype(np.int64)
    yi = np.floor(y).astype(np.int64)
    xf = x - xi
    yf = y - yi

    def h(ix, iy):
        if period is not None:
            ix = np.mod(ix, period)
            iy = np.mod(iy, period)
        return perm[(perm[ix & 255] + iy) & 255]

    def grad(ix, iy, dx, dy):
        g = h(ix, iy)
        return gx[g] * dx + gy[g] * dy

    n00 = grad(xi, yi, xf, yf)
    n10 = grad(xi + 1, yi, xf - 1, yf)
    n01 = grad(xi, yi + 1, xf, yf - 1)
    n11 = grad(xi + 1, yi + 1, xf - 1, yf - 1)
    u = _fade(xf)
    v = _fade(yf)
    nx0 = n00 + u * (n10 - n00)
    nx1 = n01 + u * (n11 - n01)
    return (nx0 + v * (nx1 - nx0)) * 1.41


def fbm2d(x, y, octaves=5, lacunarity=2.0, gain=0.5, seed=0, period=None):
    total = np.zeros(np.broadcast(x, y).shape, dtype=np.float64)
    amp = 1.0
    norm = 0.0
    freq = 1.0
    for o in range(octaves):
        p = None if period is None else int(round(period * freq))
        total += amp * perlin2d(x * freq, y * freq, seed=seed + 101 * o, period=p)
        norm += amp
        amp *= gain
        freq *= lacunarity
    return total / norm


def value_noise_grid(shape, seed=0):
    """Plain per-cell white noise in [0,1)."""
    return np.random.default_rng(seed).random(shape)
