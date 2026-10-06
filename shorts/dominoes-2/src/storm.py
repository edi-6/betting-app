"""The storm that comes in after the run stops: the day darkening, rain, and the lightning that sets the field off.

All functions of simulation time (slow motion slows the rain and stretches the flashes).
"""
import numpy as np

import layout as LY
from creeper import VOXEL_DTYPE


def _ss(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def darkness(t, ev):
    """0 (a sunny day) .. 1: the storm coming in after the run stops, deeper still at the end."""
    d = 0.62 * _ss(ev['stop'] + 0.2, ev['strike'] - 0.1, t)
    d += 0.26 * _ss(ev['done'] - 0.2, ev['done'] + 1.0, t)
    return float(min(d, 0.95))


def flash(t, ev):
    """The lightning's flash on the sky and the land: the first strike (right by the gap) brightest; each strike
    flickers twice. Never more than three flashes a second of video."""
    f = 0.0
    for k, (ts, x) in enumerate(ev['strikes']):
        a = 1.25 if k == 0 else 0.5
        if t >= ts:
            f += a * np.exp(-(t - ts) / 0.07)
        if t >= ts + 0.13:
            f += 0.45 * a * np.exp(-(t - ts - 0.13) / 0.06)
    return float(f)


def rain_amount(t, ev):
    return float(_ss(ev['stop'] + 0.9, ev['strike'] + 0.4, t))


def rain(eye, t, amount, n=1600, seed=5):
    """Rain streaks around the camera: (N, 8) head3 tail3 width alpha, falling at 30 blocks/s with a little wind."""
    if amount <= 0.01:
        return np.zeros((0, 8), np.float32)
    rng = np.random.default_rng(seed)
    box = np.array([24.0, 24.0, 22.0])
    base = rng.random((n, 3)) * box - box / 2
    v = np.array([1.6, 0.8, -30.0]) * rng.uniform(0.85, 1.15, (n, 1))
    p = base + v * t
    # wrap into the box round the camera
    p = (p - np.asarray(eye, float) + box / 2) % box - box / 2 + np.asarray(eye, float)
    p[:, 2] = np.maximum(p[:, 2], 0.05)
    tail = p - v / np.linalg.norm(v, axis=1, keepdims=True) * 0.95
    out = np.zeros((n, 8), np.float32)
    out[:, 0:3] = p
    out[:, 3:6] = tail
    out[:, 6] = 0.022
    out[:, 7] = 0.26 * amount * rng.uniform(0.5, 1.0, n)
    return out


def bolt_voxels(t, t_strike, x, y, seed=13):
    """A lightning bolt: a jagged column of glowing voxels from the clouds to the ground at (x, y), flickering out
    (two strokes)."""
    age = t - t_strike
    if age < 0 or age > 0.5:
        return np.zeros(0, VOXEL_DTYPE)
    if 0.09 < age < 0.13:
        return np.zeros(0, VOXEL_DTYPE)
    rng = np.random.default_rng(seed)
    pts = []
    p = np.array([x + rng.uniform(-10, 10), y + rng.uniform(-6, 8), 110.0])
    target = np.array([x, y, 0.0])
    branches = []
    while p[2] > 0:
        step = (target - p) / max(1.0, p[2] / 2.4)
        step += rng.normal(0, 1.3, 3) * np.array([1, 1, 0.2])
        step[2] = -abs(step[2]) - 1.4
        n = int(np.ceil(np.linalg.norm(step) / 0.18))
        for k in range(n):
            pts.append(p + step * k / n)
        if rng.random() < 0.12 and p[2] > 25:
            branches.append(p.copy())
        p = p + step
    for b in branches:                      # a few short forks
        q = b.copy()
        d = rng.normal(0, 1, 3)
        d[2] = -abs(d[2]) - 1.0
        for _ in range(int(rng.integers(4, 9))):
            st = d * 1.6 + rng.normal(0, 0.7, 3)
            st[2] = -abs(st[2])
            n = int(np.ceil(np.linalg.norm(st) / 0.18))
            for k in range(n):
                pts.append(q + st * k / n)
            q = q + st
    pts = np.array(pts)
    inst = np.zeros(len(pts), VOXEL_DTYPE)
    inst['pos'] = pts
    inst['quat'] = (0, 0, 0, 1)
    inst['scale'] = 0.24 * (1.0 - 0.5 * min(1.0, age / 0.5))
    for k in ('cx', 'cy', 'cz', 'inner'):
        inst[k][:, :3] = (215, 225, 255)
    inst['cx'][:, 3] = 0b111111
    inst['cy'][:, 3] = 0b111111
    inst['cz'][:, 3] = 255
    return inst


def bolts(t, ev):
    """All the bolts in the air now, and their lights ([x, y, z, intensity, radius, r, g, b])."""
    vox, lights = [], []
    for k, (ts, x) in enumerate(ev['strikes']):
        v = bolt_voxels(t, ts, x, LY.FEEDER_Y, seed=13 + 7 * k)
        if len(v):
            vox.append(v)
            a = np.exp(-max(0.0, t - ts) / 0.12)
            lights.append([x, LY.FEEDER_Y + 1.0, 12.0, (160.0 if k == 0 else 90.0) * a, 70.0, 0.75, 0.82, 1.0])
    return vox, lights
