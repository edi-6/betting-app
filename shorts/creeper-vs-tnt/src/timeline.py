"""Video structure: the rounds, their TNT formations, camera moves, HUD timing, and the opening hook."""
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
    if len(keys) == 2:
        e = 0.35 * u + 0.65 * _ease(u)
    else:
        e = _ease(u)          # multi-key paths are hold / move / settle choreography: ease every segment
    eye = np.array(a[1], float) * (1 - e) + np.array(b[1], float) * e
    tgt = np.array(a[2], float) * (1 - e) + np.array(b[2], float) * e
    fov = a[3] * (1 - e) + b[3] * e
    return eye, tgt, fov


# ---------------------------------------------------------------------------------------------
# rounds
# ---------------------------------------------------------------------------------------------
def round_defs():
    R = []
    # 1 TNT: straight into the face ---------------------------------------------------------------
    f1 = S.block_formation(1, 1, 1, x_center=0.0, z_bottom=22.0, y_front=-12.0, launch=0.7, speed=22.0, seed=11)
    R.append(dict(
        key='r1', label='1 TNT', formation=f1, sim=dict(seed=101), duration=3.6, hp_loss=1,
        stamp='x', stamp_at=2.5,
        shots=[dict(start=0.0, end=3.6, keys=[(0.0, (11.0, -20.0, 23.5), (-0.6, -8.0, 21.6), 54.0),
                                              (3.6, (9.5, -18.0, 23.0), (-0.3, -6.5, 21.5), 51.0)])],
        shake=0.25,
    ))
    # 10 TNT: two rows across the face ------------------------------------------------------------
    f2 = S.block_formation(5, 2, 1, x_center=0.0, z_bottom=20.4, y_front=-13.0, spacing=1.6, launch=0.6, speed=22.0,
                           seed=12)
    R.append(dict(
        key='r2', label='10 TNT', formation=f2, sim=dict(seed=102), duration=3.9, hp_loss=3,
        stamp='x', stamp_at=2.8,
        shots=[dict(start=0.0, end=3.9, keys=[(0.0, (-12.0, -20.5, 23.0), (0.0, -5.0, 20.0), 54.0),
                                              (3.9, (-10.5, -19.0, 22.5), (0.0, -4.5, 20.0), 52.0)])],
        shake=0.4,
    ))
    # 100 TNT: a 10x10 wall at the chest and face ------------------------------------------------------
    f3 = S.block_formation(10, 10, 1, x_center=0.0, z_bottom=12.0, y_front=-15.0, spacing=1.1, launch=0.6,
                           speed=24.0, seed=13)
    R.append(dict(
        key='r3', label='100 TNT', formation=f3, sim=dict(seed=103), duration=4.6, hp_loss=9,
        stamp='x', stamp_at=3.6,
        shots=[dict(start=0.0, end=4.6, keys=[(0.0, (24.5, -13.5, 20.0), (0.0, -11.0, 16.5), 64.0),
                                              (0.62, (24.3, -13.3, 20.0), (0.0, -10.6, 16.5), 63.0),
                                              (1.15, (23.5, -12.5, 20.0), (0.0, -5.0, 16.5), 57.0),
                                              (4.6, (22.0, -12.0, 21.0), (0.0, -4.0, 16.0), 56.0)])],
        shake=0.6,
    ))
    # 1,000 TNT: a 10x10x10 TNT cube into his upper left side (he keeps standing on what's left) ----------
    f4 = S.block_formation(10, 10, 10, x_center=-4.4, z_bottom=14.0, y_front=-16.0, launch=0.6, speed=18.0,
                           launch_jitter=0.03, seed=14)
    R.append(dict(
        key='r4', label='1,000 TNT', formation=f4, sim=dict(seed=104), duration=6.2, hp_loss=17,
        stamp='x', stamp_at=5.0,
        shots=[dict(start=0.0, end=6.2, keys=[(0.0, (15.0, -24.0, 20.0), (-1.5, -5.0, 17.5), 60.0),
                                              (6.2, (13.0, -22.0, 20.5), (-0.5, -3.0, 17.0), 58.0)])],
        shake=0.85,
    ))
    # 10,000 TNT: a 25x25x16 TNT block ---------------------------------------------------------------
    f5 = S.block_formation(25, 25, 16, x_center=0.0, z_bottom=0.8, y_front=-20.0, launch=0.7, speed=16.0,
                           launch_jitter=0.05, seed=15)
    R.append(dict(
        key='r5', label='10,000 TNT', formation=f5, sim=dict(seed=105), duration=8.4, hp_loss=20,
        stamp='check', stamp_at=6.2, death_fraction=0.8,
        shots=[
            dict(start=0.0, end=3.6, keys=[(0.0, (40.0, -32.0, 13.0), (0.0, -12.0, 15.0), 60.0),
                                           (3.6, (36.0, -31.0, 13.5), (0.0, -7.0, 14.0), 58.0)]),
            dict(start=3.6, end=8.4, keys=[(3.6, (24.0, -38.0, 34.0), (0.0, 2.0, 4.0), 58.0),
                                           (8.4, (19.0, -40.0, 37.0), (0.0, 2.0, 2.0), 56.0)]),
        ],
        shake=1.1,
    ))
    return R


def hook_shots():
    """Opening montage: flashes of the incoming TNT (before impact) to tease the finale."""
    return [
        # the creeper's face and the 10,000-TNT block rushing in
        dict(round='r5', sim_start=0.55, dur=0.72,
             keys=[(0.0, (36.0, -30.0, 12.5), (0.0, -12.0, 15.0), 62.0), (0.72, (34.5, -29.2, 12.8), (0.0, -11.0, 15.0), 60.0)]),
        # the 1,000-TNT cube flying at his face
        dict(round='r4', sim_start=0.45, dur=0.6,
             keys=[(0.0, (-14.0, -24.0, 26.0), (-2.0, -6.0, 16.0), 60.0), (0.6, (-13.2, -23.0, 25.6), (-2.0, -6.0, 16.0), 58.0)]),
        # low angle: the wall's leading edge about to hit
        dict(round='r5', sim_start=0.98, dur=0.62,
             keys=[(0.0, (18.0, 4.0, 1.5), (0.0, -9.0, 18.0), 68.0), (0.62, (17.4, 3.6, 1.8), (0.0, -9.0, 18.0), 67.0)]),
    ]
