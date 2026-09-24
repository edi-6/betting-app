"""Video structure: the rounds, their arrow formations, camera moves, slow-motion schedules, HUD timing and the
opening hook."""
import numpy as np

import sim as S

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
    if len(keys) == 2:
        e = 0.35 * u + 0.65 * _ease(u)
    else:
        e = _ease(u)
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


def sim_time_at(keys, t_video):
    """Simulation time reached after t_video seconds of video with this schedule."""
    return sum(frame_scales(keys, sec(t_video))) / FPS


# ---------------------------------------------------------------------------------------------
# rounds
# ---------------------------------------------------------------------------------------------
def _single(start, target, speed, launch=0.0):
    start = np.asarray([start], float)
    target = np.asarray([target], float)
    T = np.linalg.norm(target - start, axis=1) / speed
    return {'pos': start, 'vel': S.ballistic_velocity(start, target, T), 'launch': np.array([launch]),
            'var': np.array([0]), 'hover': np.array([True]), 'target': target}


def round_defs():
    R = []
    # 1 arrow: arrow-cam headshot -----------------------------------------------------------------
    f1 = _single((3.0, -34.0, 30.6), (0.2, -3.2, 27.9), 42.0)
    R.append(dict(
        key='r1', label='1 Arrow', formation=f1, sim=dict(seed=101), duration=4.2, hp_loss=1,
        stamp='x', stamp_at=3.2, shake=0.06,
        scale=[(0.0, 0.6), (0.9, 0.6), (1.15, 0.3), (1.9, 0.3), (2.5, 1.0)],
        shots=[dict(start=0.0, end=4.2, follow=dict(arrow=0, back=4.5, up=0.8, side=0.6, look=20.0, fov=56.0,
                                                    stop=13.0, soft=2.0, pull=8.0, pull_side=14.0, pull_up=-0.5,
                                                    pull_time=2.6, pull_fov=2.0))],
    ))
    # 10 arrows: a hovering stack of arrows fires one after another ---------------------------------
    rng = np.random.default_rng(12)
    gx = np.array([-1.3, 1.3] * 5)
    gz = np.repeat(np.linspace(29.0, 15.0, 5), 2) + np.tile([0.0, -0.8], 5)
    starts = np.stack([gx, np.full(10, -22.0), gz], -1)
    targets = np.array([[-1.8, -4.0, 29.3], [2.2, -4.0, 27.4], [-6.0, -10.0, 22.3], [1.2, -2.0, 21.4],
                        [-2.0, -2.0, 18.5], [6.1, -9.0, 21.6], [-2.4, -2.0, 13.2], [2.4, -2.0, 16.8],
                        [-1.6, -2.0, 7.5], [2.0, -2.0, 4.6]])
    f2 = S.hover_line(starts, targets, 40.0, 0.55, 0.0, rng)
    f2['launch'] = 0.55 + 0.085 * np.arange(10) + rng.uniform(0, 0.03, 10)
    f2['target'] = targets
    R.append(dict(
        key='r2', label='10 Arrows', formation=f2, sim=dict(seed=102), duration=3.8, hp_loss=3,
        stamp='x', stamp_at=2.8, shake=0.08,
        shots=[dict(start=0.0, end=3.8, keys=[(0.0, (9.0, -38.0, 21.0), (0.0, -6.0, 21.0), 58.0),
                                              (3.8, (7.5, -35.0, 21.0), (0.0, -5.0, 20.0), 56.0)])],
    ))
    # 100 arrows: a side volley arcs in from his right ------------------------------------------
    rng = np.random.default_rng(13)
    o3 = np.array([-95.0, -12.0])
    t3 = S.body_targets(100, rng, miss_frac=0.05, miss_r=(3.0, 8.0), approach=o3 / np.linalg.norm(o3))
    f3 = S.volley(t3, (-95.0, -12.0, 20.0), (4.0, 8.0, 4.0), 2.0, 0.3, 0.25, rng)
    f3['target'] = t3
    R.append(dict(
        key='r3', label='100 Arrows', formation=f3, sim=dict(seed=103), preroll=1.2, duration=4.2, hp_loss=9,
        stamp='x', stamp_at=3.2, shake=0.12,
        shots=[dict(start=0.0, end=4.2, keys=[(0.0, (-21.0, -27.0, 14.0), (-3.0, -3.0, 19.0), 58.0),
                                              (4.2, (-18.0, -25.0, 15.0), (-2.0, -3.0, 18.0), 55.0)])],
    ))
    # 1,000 arrows: a rain from the sky turns him into a pincushion --------------------------------
    rng = np.random.default_rng(14)
    t4 = S.body_targets(1000, rng, miss_frac=0.15, miss_r=(3.0, 12.0), approach=(0.0, -1.0))
    f4 = S.volley(t4, (0.0, -90.0, 18.0), (22.0, 8.0, 4.0), 3.0, 0.2, 0.5, rng)
    f4['target'] = t4
    R.append(dict(
        key='r4', label='1,000 Arrows', formation=f4, sim=dict(seed=104, debris_frac=0.35), preroll=1.9, duration=5.0, hp_loss=17,
        stamp='x', stamp_at=3.9, shake=0.16,
        shots=[dict(start=0.0, end=5.0, keys=[(0.0, (36.0, -14.0, 15.0), (0.0, -26.0, 34.0), 64.0),
                                              (1.0, (35.0, -13.5, 15.2), (0.0, -4.5, 17.0), 60.0),
                                              (5.0, (31.0, -11.0, 16.0), (0.0, -3.5, 16.5), 56.0)])],
    ))
    # 10,000 arrows: ten waves of 1,000, all from the front half so the cameras on his right stay clear of
    # the swarm. Wave 0 covers him head to toe in slow motion, waves 1-4 take him apart piece by piece
    # (arms, head, body, legs), waves 5-9 rain on what is left of him.
    rng = np.random.default_rng(15)
    # (azimuth of the archers, what they aim at, first contacts between t0 and t1 of simulation time, flight time)
    specs = [(-90, 'all', 2.80, 2.92, 2.5), (-65, ('arm_r', 'arm_l'), 3.25, 3.5, 2.8),
             (-115, ('head',), 3.65, 3.9, 2.8), (-55, ('body',), 4.05, 4.3, 2.8),
             (-135, ('leg_r', 'leg_l'), 4.4, 4.65, 2.8),
             (-80, 'ground', 3.9, 4.2, 2.8), (-100, 'ground', 4.2, 4.5, 2.8), (-60, 'ground', 4.5, 4.8, 2.8),
             (-125, 'ground', 4.8, 5.1, 2.8), (-90, 'ground', 5.1, 5.4, 2.8)]
    waves = []
    t5_all = []
    for k, (az, parts, t0, t1, flight) in enumerate(specs):
        a = np.radians(az)
        u = np.array([np.cos(a), np.sin(a)])
        origin = (100.0 * u[0], 100.0 * u[1], 18.0)
        if parts == 'ground':
            # into what is left of him and all around his feet
            tk = np.concatenate([S.ground_targets(500, rng, 5.0), S.ground_targets(500, rng, 13.0, r_min=5.0)])
        else:
            tk = S.body_targets(1000, rng, parts=None if parts == 'all' else list(parts), miss_frac=0.08,
                                miss_r=(2.0, 9.0), approach=u)
        spread = (12.0, 8.0, 3.0) if k == 0 else (14.0, 14.0, 3.0)
        wv = S.arrive_between(S.volley(tk, origin, spread, flight, 0.0, 0.0, rng, time_spread=0.04), t0, t1, rng)
        waves.append(wv)
        t5_all.append(tk)
    f5 = S.merge(*waves)
    f5['target'] = np.concatenate(t5_all)
    R.append(dict(
        key='r5', label='10,000 Arrows', formation=f5,
        sim=dict(seed=105, wound_radius=0.45, voxel_cost=1.6, debris_frac=0.4,
                 severs=[(3.42, 'arm_r', (-1.8, -0.8, 1.2), (0.9, 0.0, 0.3)),
                         (3.48, 'arm_l', (1.8, -0.8, 1.2), (0.9, 0.0, -0.3)),
                         (3.84, 'head', (5.5, 2.5, 3.0), (0.6, -2.4, 0.5))]),
        duration=7.8, hp_loss=20, stamp='check', stamp_at=6.8, death_fraction=0.8, shake=0.3,
        scale=[(0.0, 1.0), (2.5, 1.0), (2.8, 0.22), (4.0, 0.22), (4.5, 1.0)],
        shots=[
            # he waits...
            dict(start=0.0, end=1.0, keys=[(0.0, (16.0, -23.0, 14.0), (0.0, -3.0, 25.0), 68.0),
                                           (1.0, (15.2, -22.2, 14.4), (0.0, -3.0, 25.0), 66.5)]),
            # ...over his shoulder the sky fills with arrows
            dict(start=1.0, end=2.8, keys=[(1.0, (5.2, 34.5, 40.0), (0.0, -40.0, 24.0), 62.0),
                                           (2.8, (4.5, 32.0, 39.5), (0.0, -40.0, 25.0), 60.0)]),
            # slow motion: the first wave lands
            dict(start=2.8, end=4.4, keys=[(2.8, (32.0, -10.0, 18.0), (0.0, -2.5, 19.0), 56.0),
                                           (4.4, (30.0, -9.5, 18.5), (0.0, -2.5, 19.0), 54.0)]),
            # the next waves take him apart
            dict(start=4.4, end=5.9, keys=[(4.4, (48.0, -22.0, 26.0), (0.0, -1.0, 15.0), 54.0),
                                           (5.9, (46.0, -23.0, 26.5), (0.0, -1.0, 14.0), 54.0)]),
            # and keep coming until there is nothing left
            dict(start=5.9, end=7.8, keys=[(5.9, (38.0, -25.0, 30.0), (2.5, 1.0, 7.0), 52.0),
                                           (7.8, (42.0, -28.0, 34.0), (2.5, 1.0, 6.0), 52.0)]),
        ],
    ))
    return R


def pick_arrow(formation, lo, hi, first=None, last=None):
    """Index of the first arrow (within [first, last)) whose target lies in the box lo..hi."""
    t = formation['target']
    idx = np.arange(len(t))
    sel = np.all((t >= lo) & (t <= hi), axis=1)
    if first is not None:
        sel &= idx >= first
    if last is not None:
        sel &= idx < last
    hits = np.nonzero(sel)[0]
    return int(hits[0]) if len(hits) else 0


def hook_shots():
    """Opening montage: the sky filling with arrows, an arrow-cam dive at his head and the rain about to land
    (no impacts - those are saved for the rounds)."""
    f5 = [r for r in round_defs() if r['key'] == 'r5'][0]['formation']
    dive = pick_arrow(f5, (-1.5, -4.5, 26.0), (1.5, 4.0, 31.0), 0, 1000)
    return [
        # over his shoulder: the sky full of arrows coming for him
        dict(round='r5', sim_start=1.3, dur=0.8, scale=[(0.0, 0.55)],
             keys=[(0.0, (5.0, 34.0, 40.0), (0.0, -40.0, 24.0), 62.0), (0.8, (4.6, 32.5, 39.6), (0.0, -40.0, 24.8), 61.0)]),
        # arrow-cam in the middle of the first wave, diving at his head
        dict(round='r5', sim_start=1.95, dur=0.7, scale=[(0.0, 0.8)],
             follow=dict(arrow=dive, back=4.5, up=0.8, side=0.6, look=20.0, fov=60.0)),
        # a thousand arrows about to land on him
        dict(round='r4', sim_start=2.55, dur=0.62, scale=[(0.0, 0.5)],
             keys=[(0.0, (36.0, -14.0, 15.0), (0.0, -4.5, 17.0), 60.0), (0.62, (35.0, -13.6, 15.2), (0.0, -4.5, 17.0), 59.0)]),
    ]
