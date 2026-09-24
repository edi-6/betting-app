"""The battle (what happens when, in simulation time) and the edit (which shots show it, with their cameras and
slow motion, the scoreboard story and the flash-forward cold open)."""
import numpy as np

import arena as A
from models import LINEUP, SPECS

FPS = 30
ZOMBIE, CREEPER, STEVE, WARDEN = 0, 1, 2, 3
POS = {k: np.array(off, float) for k, off in LINEUP}
NAMES = [k for k, _ in LINEUP]


def sec(t):
    return int(round(t * FPS))


def _ease(u):
    return u * u * (3 - 2 * u)


# ---------------------------------------------------------------------------------------------
# formations
# ---------------------------------------------------------------------------------------------
def body_targets(kind, n, rng, approach=None, shrink=0.4, parts=None):
    """Aim points inside a giant's body parts (weighted by front area); with `approach` (unit vector from the
    giant towards the archers) moved out to where they enter the part, so flight time = time of first contact."""
    spec = SPECS[kind]
    names = list(spec['parts']) if parts is None else parts
    boxes = np.array([spec['parts'][k] for k in names], float) * spec['pxu']
    boxes[:, :3] += POS[kind]
    area = boxes[:, 3] * boxes[:, 5]
    b = boxes[rng.choice(len(names), size=n, p=area / area.sum())]
    lo = b[:, :3] + shrink
    hi = b[:, :3] + b[:, 3:] - shrink
    pts = lo + (hi - lo) * rng.random((n, 3))
    if approach is not None:
        a = np.asarray(approach, float)[:2]
        a = a / np.linalg.norm(a)
        s = np.full(n, np.inf)
        for ax in range(2):
            if abs(a[ax]) > 1e-9:
                edge = b[:, ax] + (b[:, 3 + ax] if a[ax] > 0 else 0.0)
                s = np.minimum(s, (edge - pts[:, ax]) / a[ax])
        pts[:, :2] += a[None, :] * s[:, None]
    return pts


def arrow_volley(rng):
    """Round 1: a thousand arrows arcing in from the front, spread over all four giants (a few miss)."""
    n_each = [290, 150, 280, 280]
    targets = []
    for kind, n in zip(NAMES, n_each):
        if kind == 'creeper':
            # its thin legs are spared so it is still standing when it goes off, and most arrows go into its
            # body so its face stays readable while it flashes
            nh = n // 4
            t = np.concatenate([body_targets(kind, nh, rng, approach=(0.0, -1.0), parts=['head']),
                                body_targets(kind, n - nh, rng, approach=(0.0, -1.0), parts=['body'])])
        else:
            t = body_targets(kind, n, rng, approach=(0.0, -1.0))
        miss = rng.random(n) < 0.08
        m = int(miss.sum())
        if m:
            ang = rng.uniform(0, 2 * np.pi, m)
            r = rng.uniform(3.0, 10.0, m)
            t[miss] = np.stack([POS[kind][0] + r * np.cos(ang), r * np.sin(ang) - 4.0, np.zeros(m)], -1)
        targets.append(t)
    tg = np.concatenate(targets)
    n = len(tg)
    p0 = np.stack([tg[:, 0] * 0.6 + rng.normal(0, 10.0, n), np.full(n, -105.0) + rng.normal(0, 6.0, n),
                   18.0 + rng.normal(0, 3.0, n)], -1)
    T = 2.3 * (1.0 + rng.uniform(-0.04, 0.04, n))
    vel = (tg - p0) / T[:, None] - 0.5 * A.GRAV_A[None, :] * T[:, None]
    launch = 1.0 + rng.uniform(0.0, 0.25, n)
    return {'pos': p0, 'vel': vel, 'launch': launch, 'var': rng.integers(0, 3, n).astype(float), 'target': tg}


def grid_cloud(centres, counts, sigma, rng, cell, clip):
    """Grid positions scattered (normal, clipped) around several centres."""
    xy = []
    for c, n, s in zip(centres, counts, sigma):
        r = np.abs(rng.normal(0, s, n))
        r = np.where(r > clip, rng.uniform(0, clip, n), r)
        ang = rng.uniform(0, 2 * np.pi, n)
        xy.append(np.stack([c[0] + r * np.cos(ang), c[1] + r * np.sin(ang)], -1))
    xy = np.concatenate(xy)
    return np.round(xy / cell) * cell


ANVIL_BLOCKS = (('zombie', (0.0, -3.0), 5), ('steve', (0.0, 0.0), 3), ('warden', (0.0, -0.6), 2))


def anvil_rain(rng):
    """Round 2: a thousand anvils as three 10 x 10 blocks hanging over the three who are left (5, 3 and 2 layers:
    500 + 300 + 200), then dropping layer by layer from the bottom up."""
    xs, ys, zs, t0s = [], [], [], []
    ii, jj = np.meshgrid(np.arange(10) - 4.5, np.arange(10) - 4.5, indexing='ij')
    for kind, off, layers in ANVIL_BLOCKS:
        cx = np.round((POS[kind][0] + off[0]) / A.CELL) * A.CELL
        cy = np.round((POS[kind][1] + off[1]) / A.CELL) * A.CELL
        for L in range(layers):
            xs.append(cx + (ii.ravel() + 0.5) * A.CELL)
            ys.append(cy + (jj.ravel() + 0.5) * A.CELL)
            zs.append(np.full(100, 47.0 + L * A.CELL * 1.02))
            t0s.append(8.3 + L * 0.14 + rng.uniform(0, 0.1, 100))
    x, y, z, t0 = (np.concatenate(v) for v in (xs, ys, zs, t0s))
    n = len(x)
    return {'x': x, 'y': y, 'z': z, 't0': t0, 'vz': np.full(n, 12.0), 'yaw': rng.integers(0, 4, n),
            'var': rng.integers(0, 2, n).astype(float), 'hover': np.ones(n, bool), 'appear': np.full(n, 7.4)}


TNT_SLAB = (50, 10, 20)                   # 50 x 10 x 20 = exactly 10,000 blocks
TNT_Z0 = 50.0


def tnt_storm(rng):
    """Round 3: ten thousand primed TNT as one 50 x 10 x 20 wall hanging right over the last two (Steve under
    one half, the Warden under the other), then raining down layer by layer from the bottom up."""
    nx, ny, nz = TNT_SLAB
    x0 = 0.5 * (POS['steve'][0] + POS['warden'][0]) - nx / 2 + 0.5
    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    i, j, k = i.ravel(), j.ravel(), k.ravel()
    pos = np.stack([x0 + i, j - ny / 2, TNT_Z0 + k + 0.5], -1).astype(float)
    n = len(pos)
    t0 = 13.8 + k * 0.06 + rng.uniform(0, 0.05, n)
    return {'pos': pos, 'vel': np.zeros((n, 3)), 't0': t0, 'fuse': t0 + 5.0, 'hover': np.ones(n, bool),
            'appear': np.full(n, 12.8)}


WARDEN_HEAD = POS['warden'] + np.array([0.0, -1.2, 30.4])


def _zbox(x0, x1, y0, y1, z0, z1):
    o = POS['zombie']
    return (o[0] + x0, o[0] + x1, o[1] + y0, o[1] + y1, z0, z1)


def plan(seed=7):
    rng = np.random.default_rng(seed)
    return {
        'arrows': arrow_volley(rng),
        'creeper': dict(giant=CREEPER, prime=4.1, explode=5.7, radius=14.0, crater=7.0, fx_scale=3.4),
        'anvils': anvil_rain(rng),
        # the weight snaps the zombie's arms off, then its knees give way and it topples forward
        'severs': [(9.75, ZOMBIE, _zbox(-8.0, -4.0, -2.2, -1.3, 20.0, 24.0), (-2.0, -3.0, 0.0), (1.2, 0.0, 0.4)),
                   (10.0, ZOMBIE, _zbox(4.0, 8.0, -2.2, -1.3, 20.0, 24.0), (2.0, -3.0, 0.0), (1.2, 0.0, -0.4)),
                   (10.9, ZOMBIE, _zbox(-4.0, 4.0, -2.0, 2.0, 8.5, 9.6), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                    dict(pivot=POS['zombie'] + np.array([0.0, -2.0, 9.0]), axis=(1.0, 0.0, 0.0), omega=0.9))],
        'tnt': tnt_storm(rng),
        # his sonic booms blow a hole through the slab above him (the shock wave reaches well past the rings)
        'booms': [(14.5, WARDEN_HEAD, (0.0, 0.0, 1.0), 6.5), (15.3, WARDEN_HEAD, (0.0, 0.0, 1.0), 6.5)],
    }


# ---------------------------------------------------------------------------------------------
# the edit
# ---------------------------------------------------------------------------------------------
def cam_path(keys, t):
    """Keyframes (time_s, eye, target, fov), eased between keys."""
    ts = [k[0] for k in keys]
    if t <= ts[0] or len(keys) == 1:
        k = keys[0]
        return np.array(k[1], float), np.array(k[2], float), float(k[3])
    if t >= ts[-1]:
        k = keys[-1]
        return np.array(k[1], float), np.array(k[2], float), float(k[3])
    i = int(np.searchsorted(ts, t)) - 1
    a, b = keys[i], keys[i + 1]
    u = (t - a[0]) / (b[0] - a[0])
    e = 0.35 * u + 0.65 * _ease(u) if len(keys) == 2 else _ease(u)
    eye = np.array(a[1], float) * (1 - e) + np.array(b[1], float) * e
    tgt = np.array(a[2], float) * (1 - e) + np.array(b[2], float) * e
    return eye, tgt, a[3] * (1 - e) + b[3] * e


def scale_at(keys, t):
    """Slow motion: keyframes (shot_time_s, time_scale), eased between keys."""
    if not keys:
        return 1.0
    ts = [k[0] for k in keys]
    if t <= ts[0]:
        return float(keys[0][1])
    if t >= ts[-1]:
        return float(keys[-1][1])
    i = int(np.searchsorted(ts, t)) - 1
    u = _ease((t - ts[i]) / (ts[i + 1] - ts[i]))
    return float(keys[i][1] * (1 - u) + keys[i + 1][1] * u)


def _intro(name, eye0, eye1, tgt, fov0, fov1, dur=0.66):
    return dict(name=f'L_{name}', dur=dur, scale=[(0.0, 0.3)], intro=NAMES.index(name),
                keys=[(0.0, eye0, tgt, fov0), (0.14, eye0, tgt, fov1 + 2.0), (dur, eye1, tgt, fov1)])


def shots():
    """The main edit, in order. Each shot runs the battle on from where the last one stopped, or first skips
    ahead (unseen) to `sim` if given. dur: video seconds; scale: slow-motion keys; keys: camera keys."""
    return [
        # ---- the line-up: who is fighting (nothing moves yet, the archers have not loosed)
        _intro('zombie', (-40.0, -22.0, 5.5), (-39.0, -20.5, 6.0), (-33.0, -3.0, 20.5), 60.0, 52.0),
        _intro('creeper', (-7.0, -26.0, 4.0), (-8.2, -24.2, 4.4), (-16.0, 0.0, 15.5), 58.0, 50.0),
        _intro('steve', (-8.0, -28.0, 5.0), (-6.8, -26.2, 5.4), (1.0, 0.0, 19.5), 60.0, 52.0),
        _intro('warden', (32.0, -27.0, 4.5), (30.8, -25.2, 5.0), (23.0, -1.0, 18.0), 60.0, 52.0, dur=0.72),
        # ---- round 1: a thousand arrows loosed from the forest, over our heads, at all four
        dict(name='R1_volley', sim=0.95, dur=1.45, banner=1, scale=[(0.0, 1.0)],
             keys=[(0.0, (-3.0, -99.0, 48.0), (-4.0, 0.0, 12.0), 44.0),
                   (1.45, (-3.0, -95.0, 47.0), (-4.0, 0.0, 13.0), 42.0)]),
        dict(name='R1_impact', dur=2.0, scale=[(0.0, 1.0), (0.42, 1.0), (0.62, 0.28), (1.5, 0.28), (1.9, 0.8)],
             keys=[(0.0, (-11.0, -41.0, 6.0), (-8.0, 0.0, 18.0), 60.0),
                   (2.0, (-10.0, -38.0, 6.5), (-8.0, 0.0, 17.0), 58.0)]),
        # the creeper starts to hiss and flash
        dict(name='R1_hiss', dur=1.4, scale=[(0.0, 1.0)],
             keys=[(0.0, (-15.0, -36.0, 20.0), (-16.0, 0.0, 19.5), 44.0),
                   (1.4, (-15.5, -32.0, 20.5), (-16.0, 0.0, 20.0), 40.0)]),
        dict(name='R1_boom', dur=2.5, scale=[(0.0, 1.0), (0.72, 1.0), (0.84, 0.25), (1.75, 0.25), (2.1, 0.8)],
             keys=[(0.0, (-17.0, -42.0, 9.0), (-16.0, 0.0, 15.0), 66.0),
                   (2.5, (-17.0, -44.0, 9.8), (-16.0, 0.0, 14.0), 66.0)]),
        # ---- round 2: a thousand anvils, three blocks of them
        dict(name='R2_sky', sim=7.2, dur=1.4, banner=2, scale=[(0.0, 1.0)],
             keys=[(0.0, (-28.0, -46.0, 18.0), (-38.0, -2.0, 35.0), 64.0),
                   (1.4, (-28.5, -44.0, 18.5), (-38.0, -2.0, 35.0), 62.0)]),
        dict(name='R2_rain', dur=1.9, scale=[(0.0, 0.9)],
             keys=[(0.0, (-16.0, -40.0, 30.0), (-33.0, -3.0, 28.0), 58.0),
                   (1.9, (-16.5, -41.0, 29.0), (-33.0, -3.0, 24.0), 58.0)]),
        dict(name='R2_topple', sim=10.6, dur=2.4, scale=[(0.0, 1.0), (0.35, 1.0), (0.6, 0.7), (2.4, 0.7)],
             keys=[(0.0, (-10.0, -38.0, 7.0), (-33.0, -6.0, 16.0), 56.0),
                   (2.4, (-11.0, -39.0, 6.5), (-33.0, -12.0, 10.0), 58.0)]),
        # ---- round 3: ten thousand TNT in one slab over the last two
        dict(name='R3_sky', sim=12.6, dur=1.4, banner=3, scale=[(0.0, 1.0)],
             keys=[(0.0, (12.0, -46.0, 20.0), (12.0, 0.0, 44.0), 66.0),
                   (1.4, (12.0, -43.0, 21.0), (12.0, 0.0, 43.0), 64.0)]),
        dict(name='R3_boom1', dur=1.9, scale=[(0.0, 1.0), (0.35, 1.0), (0.55, 0.35), (1.5, 0.35), (1.9, 0.7)],
             keys=[(0.0, (45.0, -26.0, 26.0), (23.0, -2.0, 44.0), 62.0),
                   (1.9, (44.0, -25.0, 25.0), (23.0, -2.0, 45.0), 62.0)]),
        dict(name='R3_boom2', dur=1.1, scale=[(0.0, 0.7), (0.3, 0.55), (1.1, 0.7)],
             keys=[(0.0, (31.0, -27.0, 11.0), (23.0, -1.0, 33.0), 60.0),
                   (1.1, (30.5, -26.0, 11.0), (23.0, -1.0, 34.0), 58.0)]),
        dict(name='R3_storm', dur=2.0, scale=[(0.0, 0.8)],
             keys=[(0.0, (-10.0, -62.0, 30.0), (4.0, 0.0, 18.0), 50.0),
                   (2.0, (-10.5, -60.0, 30.5), (4.0, 0.0, 17.0), 48.0)]),
        dict(name='R3_steve', dur=1.4, scale=[(0.0, 0.8)],
             keys=[(0.0, (10.0, -60.0, 16.0), (12.0, 0.0, 18.0), 56.0),
                   (1.4, (10.5, -58.0, 16.5), (12.0, 0.0, 18.0), 55.0)]),
        # ---- the last giant standing
        dict(name='W_hero', dur=3.1, scale=[(0.0, 0.8)], winner=0.8,
             keys=[(0.0, (15.0, -36.0, 5.0), (23.0, -1.0, 19.0), 58.0),
                   (3.1, (18.5, -27.0, 6.5), (23.0, -1.0, 21.0), 50.0)]),
    ]


def cold_open():
    """Flash-forward to round 3, shown first: (main shot, from, duration, camera keys). The frames are the very
    same moments of the battle as in the main edit, filmed from their own angles."""
    return [
        dict(shot='R3_boom1', t0=0.4, dur=1.0,
             keys=[(0.0, (31.0, -34.0, 22.0), (21.0, -2.0, 45.0), 64.0), (1.0, (30.5, -33.0, 23.0), (21.0, -2.0, 46.0), 60.0)]),
        dict(shot='R3_storm', t0=0.5, dur=1.0,
             keys=[(0.0, (-12.0, -40.0, 14.0), (3.0, 0.0, 22.0), 58.0), (1.0, (-11.0, -38.5, 14.5), (3.0, 0.0, 21.0), 56.0)]),
    ]


def schedule():
    """Frame-by-frame plan of the main edit: [(shot index, frame in shot, time scale, skip_to or None)]."""
    out = []
    for si, sh in enumerate(shots()):
        n = sec(sh['dur'])
        for f in range(n):
            out.append((si, f, scale_at(sh.get('scale'), f / FPS), sh.get('sim') if f == 0 else None))
    return out
