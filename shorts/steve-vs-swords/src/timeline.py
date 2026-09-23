"""Video structure: the rounds, their sword formations, camera moves, HUD timing, and the opening hook."""
import numpy as np

import sim as S

FPS = 30


def sec(t):
    return int(round(t * FPS))


# ---------------------------------------------------------------------------------------------
# camera paths: keyframes (time_s, eye, target, fov); eased interpolation between them
# ---------------------------------------------------------------------------------------------
def _ease(u):
    return u * u * (3 - 2 * u)


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
    # gentle ease only at the very ends of a multi-key path, linear-ish in between (keeps motion alive)
    if len(keys) == 2:
        e = 0.35 * u + 0.65 * _ease(u)
    else:
        e = _ease(u) if (i == 0 or i == len(keys) - 2) else u
    eye = np.array(a[1], float) * (1 - e) + np.array(b[1], float) * e
    tgt = np.array(a[2], float) * (1 - e) + np.array(b[2], float) * e
    fov = a[3] * (1 - e) + b[3] * e
    return eye, tgt, fov


# ---------------------------------------------------------------------------------------------
# rounds
# ---------------------------------------------------------------------------------------------
def round_defs():
    R = []
    # 1 sword -------------------------------------------------------------------------------------
    f1 = S.grid_formation(1, 1, (0.7, 0.7), (19.2, 19.2), y_start=-12.0, launch=0.6, seed=11,
                          mats=np.array([4]), jitter=0.0, speed=24.0)
    R.append(dict(
        key='r1', label='1 Sword', formation=f1, sim=dict(seed=101, wound_radius=None, loosen=False), duration=3.7,
        hp_loss=1, stamp='x', stamp_at=2.55,
        shots=[dict(start=0.0, end=3.7, keys=[(0.0, (16.5, -20.5, 22.0), (0.0, -5.5, 19.8), 56.0),
                                              (3.7, (14.5, -19.0, 21.6), (0.0, -4.5, 19.8), 54.0)])],
        shake=0.12,
    ))
    # 10 swords -----------------------------------------------------------------------------------
    f2 = S.grid_formation(5, 2, (-2.5, 2.5), (18.0, 20.8), y_start=-14.0, launch=0.55, seed=12, speed=24.0)
    R.append(dict(
        key='r2', label='10 Swords', formation=f2, sim=dict(seed=102), duration=4.0, hp_loss=3,
        stamp='x', stamp_at=2.95,
        shots=[dict(start=0.0, end=4.0, keys=[(0.0, (19.0, -25.0, 25.0), (0.0, -6.0, 19.5), 56.0),
                                              (4.0, (17.0, -24.0, 24.0), (0.0, -5.0, 19.5), 54.0)])],
        shake=0.18,
    ))
    # 100 swords ----------------------------------------------------------------------------------
    f3 = S.grid_formation(10, 10, (-3.3, 3.3), (15.2, 29.3), y_start=-17.0, launch=0.55, seed=13, speed=25.0)
    R.append(dict(
        key='r3', label='100 Swords', formation=f3, sim=dict(seed=103), duration=4.8, hp_loss=9,
        stamp='x', stamp_at=3.75,
        shots=[dict(start=0.0, end=4.8, keys=[(0.0, (27.0, -29.0, 17.0), (0.0, -8.0, 19.5), 58.0),
                                              (4.8, (22.5, -29.0, 17.5), (0.0, -5.0, 19.5), 56.0)])],
        shake=0.3,
    ))
    # 1,000 swords --------------------------------------------------------------------------------
    f4 = S.grid_formation(20, 50, (-3.85, 3.85), (0.7, 31.4), y_start=-21.0, launch=0.6, seed=14,
                          speed=26.0, launch_jitter=0.06)
    R.append(dict(
        key='r4', label='1,000 Swords', formation=f4, sim=dict(seed=104), duration=6.4, hp_loss=17,
        stamp='x', stamp_at=5.2,
        shots=[dict(start=0.0, end=6.4, keys=[(0.0, (36.0, -38.0, 15.0), (0.0, -9.0, 16.5), 60.0),
                                              (6.4, (29.0, -38.0, 16.0), (0.0, -3.0, 16.0), 58.0)])],
        shake=0.55,
    ))
    # 10,000 swords -------------------------------------------------------------------------------
    f5 = S.silhouette_formation(1000, 10, y_start=-28.0, layer_gap=5.0, launch=0.7, seed=15, speed=22.0)
    R.append(dict(
        key='r5', label='10,000 Swords', formation=f5, sim=dict(seed=105, wound_radius=0.4, voxel_cost=1.5),
        duration=8.2, hp_loss=20, stamp='check', stamp_at=6.0, death_fraction=0.8,
        shots=[
            dict(start=0.0, end=4.4, keys=[(0.0, (46.0, -34.0, 16.0), (0.0, -18.0, 16.0), 60.0),
                                           (4.4, (39.0, -37.0, 16.5), (0.0, -7.0, 15.5), 58.0)]),
            dict(start=4.4, end=8.2, keys=[(4.4, (26.0, -40.0, 30.0), (0.0, 0.0, 9.0), 56.0),
                                           (8.2, (21.0, -42.5, 32.5), (0.0, 0.0, 7.5), 54.5)]),
        ],
        shake=0.9,
    ))
    return R


def hook_shots():
    """Opening montage: flashes of the incoming formations (before impact) to tease the finale."""
    return [
        # wide: the giant and the 10,000-sword block rushing in
        dict(round='r5', sim_start=0.72, dur=0.75,
             keys=[(0.0, (44.0, -38.0, 15.0), (0.0, -14.0, 16.0), 62.0), (0.75, (41.0, -38.5, 15.5), (0.0, -12.0, 16.0), 60.0)]),
        # the 1,000 grid sweeping in, seen from beside the giant's shoulder
        dict(round='r4', sim_start=0.62, dur=0.55,
             keys=[(0.0, (15.0, -9.0, 25.0), (-1.0, -16.0, 17.0), 62.0), (0.55, (14.5, -8.5, 25.5), (-1.0, -15.0, 17.0), 61.0)]),
        # low angle: the block's leading edge about to hit
        dict(round='r5', sim_start=1.12, dur=0.66,
             keys=[(0.0, (15.0, -13.0, 5.0), (0.0, -5.0, 19.0), 64.0), (0.66, (14.3, -12.4, 5.4), (0.0, -5.0, 19.0), 63.0)]),
    ]
