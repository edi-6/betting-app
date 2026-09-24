"""Video structure: the rounds, their anvil formations, camera moves, slow-motion schedules, HUD timing and the
opening hook."""
import numpy as np

import sim as S
from sim import CELL

FPS = 30


def sec(t):
    return int(round(t * FPS))


def _ease(u):
    return u * u * (3 - 2 * u)


# ---------------------------------------------------------------------------------------------
# camera paths: keyframes (time_s, eye, target, fov); eased interpolation between them
# ---------------------------------------------------------------------------------------------
def cam_path(keys, t):
    ts = [k[0] for k in keys]
    if t <= ts[0]:
        k = keys[0]
        return np.array(k[1], float), np.array(k[2], float), float(k[3])
    if t >= ts[-1]:
        k = keys[-1]
        return np.array(k[1], float), np.array(k[2], float), float(k[3])
    i = np.searchsorted(ts, t) - 1
    a, b = keys[i], keys[i + 1]
    u = (t - a[0]) / (b[0] - a[0])
    e = 0.35 * u + 0.65 * _ease(u) if len(keys) == 2 else _ease(u)
    eye = np.array(a[1], float) * (1 - e) + np.array(b[1], float) * e
    tgt = np.array(a[2], float) * (1 - e) + np.array(b[2], float) * e
    fov = a[3] * (1 - e) + b[3] * e
    return eye, tgt, fov


# ---------------------------------------------------------------------------------------------
# slow motion: keyframes (video_time_s, time_scale), eased between keys
# ---------------------------------------------------------------------------------------------
def scale_at(keys, t):
    if not keys:
        return 1.0
    ts = [k[0] for k in keys]
    if t <= ts[0]:
        return float(keys[0][1])
    if t >= ts[-1]:
        return float(keys[-1][1])
    i = np.searchsorted(ts, t) - 1
    u = _ease((t - ts[i]) / (ts[i + 1] - ts[i]))
    return float(keys[i][1] * (1 - u) + keys[i + 1][1] * u)


def frame_scales(keys, n):
    return [scale_at(keys, f / FPS) for f in range(n)]


# ---------------------------------------------------------------------------------------------
# rounds
# ---------------------------------------------------------------------------------------------
def grid(ixs, iys):
    a, b = np.meshgrid(np.asarray(ixs, float), np.asarray(iys, float), indexing='ij')
    return np.stack([a.ravel() * CELL, b.ravel() * CELL], -1)


HEAD_TOP = np.array([0.0, -1.2, 30.4])       # where the sonic boom leaves his body


def round_defs():
    R = []
    # 1 anvil: anvil-cam dive onto his head ---------------------------------------------------------
    rng = np.random.default_rng(21)
    f1 = S.formation(grid([0], [-1]), 54.0, 0.0, vz=30.0, rng=rng, var=0)
    R.append(dict(
        key='r1', label='1 Anvil', formation=f1, sim=dict(seed=201), duration=3.7, hp_loss=1,
        stamp='x', stamp_at=2.8, shake=0.07,
        scale=[(0.0, 1.0), (0.45, 1.0), (0.6, 0.3), (1.5, 0.3), (2.1, 1.0)],
        shots=[dict(start=0.0, end=3.7, anvilcam=dict(anvil=0, back=3.0, off=(3.0, -16.0), fov=66.0,
                                                      stop=12.0, soft=2.5, reveal_eye=(15.0, -23.0, 41.0),
                                                      reveal_tgt=(0.0, -1.5, 27.5), reveal_fov=50.0,
                                                      reveal_time=1.5))],
    ))
    # 10 anvils: a row hanging over his head drops one after another -------------------------------
    rng = np.random.default_rng(22)
    f2 = S.formation(grid(np.arange(-5, 5), [-1]), 45.0, 0.55 + 0.1 * np.arange(10), rng=rng, hover=True)
    R.append(dict(
        key='r2', label='10 Anvils', formation=f2, sim=dict(seed=202), duration=3.8, hp_loss=3,
        stamp='x', stamp_at=3.0, shake=0.09,
        shots=[dict(start=0.0, end=3.8, keys=[(0.0, (17.0, -27.0, 30.0), (0.0, -1.0, 31.0), 58.0),
                                              (3.8, (15.5, -25.0, 29.0), (0.0, -1.0, 29.0), 56.0)])],
    ))
    # 100 anvils: a 10 x 10 slab falls flat on him ---------------------------------------------------
    rng = np.random.default_rng(23)
    xy3 = grid(np.arange(-5, 5), np.arange(-5, 5))
    f3 = S.formation(xy3, 58.0, 0.5 + rng.uniform(0, 0.06, len(xy3)), vz=6.0, rng=rng, hover=True)
    R.append(dict(
        key='r3', label='100 Anvils', formation=f3, sim=dict(seed=203), duration=3.9, hp_loss=8,
        stamp='x', stamp_at=3.0, shake=0.14,
        shots=[dict(start=0.0, end=3.9, keys=[(0.0, (10.0, -18.0, 5.0), (0.0, -2.0, 36.0), 70.0),
                                              (1.4, (11.0, -19.0, 6.0), (0.0, -2.0, 30.0), 66.0),
                                              (3.9, (13.0, -21.5, 10.0), (0.0, -1.0, 21.0), 62.0)])],
    ))
    # 1,000 anvils: a 10 x 10 x 10 cube, the top of it pours off him ----------------------------------
    rng = np.random.default_rng(24)
    xy4 = np.tile(grid(np.arange(-5, 5), np.arange(-5, 5)), (10, 1))
    z4 = 50.0 + np.repeat(np.arange(10), 100) * CELL * 1.02
    f4 = S.formation(xy4, z4, 0.7, vz=8.0, rng=rng, hover=True)
    R.append(dict(
        key='r4', label='1,000 Anvils', formation=f4, sim=dict(seed=204, weight_crush=0.25), duration=5.4,
        hp_loss=17, stamp='x', stamp_at=4.4, shake=0.2,
        shots=[dict(start=0.0, end=5.4, keys=[(0.0, (30.0, -32.0, 28.0), (0.0, 0.0, 49.0), 62.0),
                                              (1.4, (29.0, -31.0, 25.0), (0.0, 0.0, 25.0), 58.0),
                                              (5.4, (26.0, -29.0, 22.0), (0.0, 0.0, 16.0), 56.0)])],
    ))
    # 10,000 anvils: the storm; he fights back with a sonic boom, then he's buried ---------------------
    rng = np.random.default_rng(25)
    n5 = 10000
    r = np.abs(rng.normal(0, 6.5, n5))
    r = np.where(r > 17.0, rng.uniform(0, 17.0, n5), r)
    ang = rng.uniform(0, 2 * np.pi, n5)
    xy5 = np.stack([r * np.cos(ang), r * np.sin(ang)], -1)
    z5 = 62.0 + rng.uniform(0, 50.0, n5)
    # the whole cloud hangs in the sky, then it rains out from the bottom up
    t5 = 0.6 + (z5 - 62.0) / 50.0 * 1.7 + rng.uniform(0, 0.25, n5)
    f5 = S.formation(xy5, z5, t5, vz=30.0, rng=rng, hover=True)
    booms = [(1.35, HEAD_TOP, (0.0, 0.0, 1.0))]
    R.append(dict(
        key='r5', label='10,000 Anvils', formation=f5,
        sim=dict(seed=205, weight_crush=0.8, crush_k=0.0009, crush_max=2.0, booms=booms), duration=8.6,
        hp_loss=20, stamp='check',
        stamp_at=7.3, death_fraction=0.45, shake=0.32, boom_at=None,
        scale=[(0.0, 1.0), (1.2, 1.0), (1.45, 0.25), (2.4, 0.25), (2.9, 1.0)],
        shots=[
            # the sky fills with anvils
            dict(start=0.0, end=1.35, keys=[(0.0, (11.5, -18.5, 3.5), (0.0, 2.0, 50.0), 76.0),
                                            (1.35, (11.0, -18.0, 3.8), (0.0, 2.0, 48.0), 74.0)]),
            # he fights back: a sonic boom rips through them (slow motion)
            dict(start=1.35, end=3.3, keys=[(1.35, (24.0, -26.0, 14.0), (0.0, -8.0, 32.0), 64.0),
                                            (3.3, (23.0, -25.0, 15.0), (0.0, -6.0, 30.0), 62.0)]),
            # but there are too many
            dict(start=3.3, end=6.1, keys=[(3.3, (38.0, -38.0, 24.0), (0.0, 0.0, 18.0), 56.0),
                                           (6.1, (40.0, -41.0, 27.0), (0.0, 0.0, 15.0), 56.0)]),
            # buried
            dict(start=6.1, end=8.6, keys=[(6.1, (46.0, -58.0, 30.0), (0.0, 0.0, 19.0), 52.0),
                                           (8.6, (50.0, -63.0, 33.0), (0.0, 0.0, 19.0), 52.0)]),
        ],
    ))
    return R


def hook_shots():
    """Opening montage: the sky full of anvils, an anvil-cam dive at his head, the sonic boom (no impacts)."""
    return [
        dict(round='r5', sim_start=0.3, dur=0.75, scale=[(0.0, 0.6)],
             keys=[(0.0, (11.5, -18.5, 3.5), (0.0, 2.0, 50.0), 76.0), (0.75, (11.1, -18.1, 3.7), (0.0, 2.0, 49.0), 75.0)]),
        dict(round='r1', sim_start=0.05, dur=0.7, scale=[(0.0, 0.7)],
             anvilcam=dict(anvil=0, back=3.0, off=(3.0, -16.0), fov=66.0)),
        dict(round='r5', sim_start=1.25, dur=0.65, scale=[(0.0, 0.5)],
             keys=[(0.0, (24.0, -26.0, 14.0), (0.0, -8.0, 32.0), 64.0), (0.65, (23.6, -25.6, 14.4), (0.0, -8.0, 32.0), 63.0)]),
    ]
