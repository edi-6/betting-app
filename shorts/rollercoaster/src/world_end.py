"""The End, built around the track: the main island floating in the void.

The cart arrives on the obsidian platform east of the island and crosses the void to it; passes through the ring of
obsidian pillars (end crystals burning on top, two of them caged in iron bars); swings out over the void in a banked
spiral west of the island; comes back through the ring, and dives into the exit portal at the island's centre (a
bedrock fountain with the dragon egg on its column), where the dragon perches. A few small islands float further
out. The crystals, the dragon and the endermen are props (mobs.py).
"""
import numpy as np

import paths
import voxel as VX
from noise import fbm2d
from world_over import clear_track

ORIGIN = (-150, -130, -10)
SIZE = (280, 260, 130)
TOP = 40              # the island's surface at its centre
RING_R = 43.0


def build(tr=None, verbose=True):
    tr = tr or paths.the_end()
    rng = np.random.default_rng(99)
    w = VX.World(size=SIZE, origin=ORIGIN)
    X, Y, Z = SIZE
    ox, oy, oz = ORIGIN
    # the island's centre is the exit portal: where the track's final dive comes down through the surface
    s_p = tr.marks['portal']
    ss = np.arange(s_p, s_p + 16.0, 0.05)
    zz = np.array([tr.pos(s)[2] for s in ss])
    k = int(np.argmax(zz < TOP + 1.4))
    pc = tr.pos(ss[k])
    cx, cy = float(np.floor(pc[0]) + 0.5), float(np.floor(pc[1]) + 0.5)
    w.center = np.array([cx, cy, TOP + 1.0])

    xs = np.arange(X) + ox + 0.5
    ys = np.arange(Y) + oy + 0.5
    GX, GY = np.meshgrid(xs, ys, indexing='ij')
    dx, dy = GX - cx, GY - cy
    ang = np.arctan2(dy, dx)
    rad = np.hypot(dx, dy)
    edge = 66.0 + 9.0 * np.sin(ang * 3 + 0.7) + 6.0 * fbm2d(np.cos(ang) * 2 + 5, np.sin(ang) * 2 + 5, octaves=3,
                                                            seed=4)
    u = np.clip(rad / edge, 0, 1.2)
    top = TOP + 2.0 * fbm2d(GX / 20.0, GY / 20.0, octaves=3, seed=7) - 3.0 * u ** 3
    bottom = TOP - 3.0 - 44.0 * np.clip(1 - u ** 1.6, 0, 1) ** 0.6
    inside = u < 1.0
    top = np.round(top).astype(int)
    bottom = np.round(bottom).astype(int)
    zs = np.arange(Z) + oz
    es = VX.BL.B['end_stone']
    for kk, z in enumerate(zs):
        sel = inside & (z >= bottom) & (z < top)
        w.ids[:, :, kk][sel] = es
    w.H = np.where(inside, top, -999)
    # small islands further out
    for (ix, iy, iz, r) in ((120.0, 70.0, 30.0, 9.0), (-120.0, -80.0, 46.0, 12.0), (90.0, -90.0, 58.0, 7.0),
                            (-110.0, 95.0, 25.0, 10.0), (40.0, 110.0, 50.0, 8.0)):
        for ddx in range(-int(r), int(r) + 1):
            for ddy in range(-int(r), int(r) + 1):
                d = np.hypot(ddx, ddy) / r
                if d > 1:
                    continue
                t = int(iz + int(1.5 * (1 - d)))
                b = int(iz - int(r * 1.3 * (1 - d) ** 0.7) - 1)
                for z in range(b, t):
                    w.set(int(ix) + ddx, int(iy) + ddy, z, 'end_stone')
    if verbose:
        print('[end] island', flush=True)
    _platform(w, tr)
    w.pillars = _pillars(w, tr, rng)
    clear_track(w, tr, keep=())
    _fountain(w)
    return w


def _platform(w, tr):
    """The obsidian platform the cart arrives on (the game's spawn platform in the End)."""
    P = tr.pos(0.0)
    x0, y0, z0 = int(np.floor(P[0])), int(np.floor(P[1])), int(np.floor(P[2])) - 2
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            w.set(x0 + dx, y0 + dy, z0, 'obsidian')


def _pillars(w, tr, rng):
    """Ten obsidian pillars round the portal, spaced so the track threads between them (checked against the
    whole track). Returns [(x, y, top_z, radius, caged)]."""
    c = w.center
    ss = np.arange(0.0, tr.length, 1.0)
    pts = np.array([tr.pos(s) for s in ss])
    best = None
    for off in np.arange(0.0, 36.0, 3.0):
        placed = []
        for k in range(10):
            r = 3 + (k * 7) % 3
            got = None
            for j in (0, 6, -6, 12, -12, 18, -18):
                for dr in (0.0, 4.0, -4.0):
                    a = np.radians(off + k * 36.0 + j)
                    p = c[:2] + (RING_R + dr) * np.array([np.cos(a), np.sin(a)])
                    dmin = np.min(np.linalg.norm(pts[:, :2] - p, axis=1))
                    if dmin >= r + 3.5:
                        got = (p, r, dmin)
                        break
                if got:
                    break
            if got:
                placed.append(got)
        near = sum(1 for (_, r, d) in placed if d < r + 12)
        key = (len(placed), near)
        if best is None or key > best[0]:
            best = (key, placed)
    out = []
    for k, (p, r, d) in enumerate(best[1]):
        h = int(24 + (k * 13) % 5 * 7)                    # 24..52 blocks above the surface
        x, y = int(np.floor(p[0])), int(np.floor(p[1]))
        base = TOP - 6
        topz = TOP + h
        for ddx in range(-r, r + 1):
            for ddy in range(-r, r + 1):
                if ddx * ddx + ddy * ddy > r * r + r:
                    continue
                for z in range(base, topz):
                    w.set(x + ddx, y + ddy, z, 'obsidian')
        w.set(x, y, topz, 'bedrock')
        caged = k in (2, 7)
        if caged:
            for ddx in range(-2, 3):
                for ddy in range(-2, 3):
                    for z in range(topz + 1, topz + 6):
                        if max(abs(ddx), abs(ddy)) == 2 or z == topz + 5:
                            w.set(x + ddx, y + ddy, z, 'iron_bars')
        out.append((x + 0.5, y + 0.5, float(topz + 1), float(r), caged))
    return out


def _fountain(w):
    """The exit portal: a bedrock bowl (7 across) round the portal surface (a prop), a bedrock column in the middle
    with the dragon egg on top."""
    cx, cy = int(np.floor(w.center[0])), int(np.floor(w.center[1]))
    zt = TOP
    for dx in range(-4, 5):
        for dy in range(-4, 5):
            r = np.hypot(dx, dy)
            if r > 4.4:
                continue
            for z in range(zt - 5, zt):
                w.set(cx + dx, cy + dy, z, 'bedrock')
            if r > 3.4:
                w.set(cx + dx, cy + dy, zt, 'bedrock')
            elif r > 0.5:
                for z in range(zt - 1, zt + 2):
                    w.set(cx + dx, cy + dy, z, 'air')
    for z in range(zt - 1, zt + 4):
        w.set(cx, cy, z, 'bedrock')
    w.set(cx, cy, zt + 4, 'dragon_egg')
    w.exit_portal = dict(center=np.array([cx + 0.5, cy + 0.5, zt - 0.8]), radius=3.4)
