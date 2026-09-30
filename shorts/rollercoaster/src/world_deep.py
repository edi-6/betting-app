"""The deep dark, built around the track: an ancient city in a vast cavern, and its great portal.

The cart comes out of the dark onto a deepslate-brick bridge over a chasm, runs between rows of columns hung with soul
lanterns, round a tower, and straight at the ancient city's portal: the great frame of deepslate shaped like a
warden's head, its opening lined with reinforced deepslate. Sculk spreads over the floor and up the walls, with
sensors and shriekers in it. The portal's surface (it wakes as the cart comes: a prop) leads to the Sift.
"""
import numpy as np

import blocks as BL
import paths
import voxel as VX
from noise import fbm2d
from world_over import clear_track

ORIGIN = (-70, -90, 0)
SIZE = (140, 200, 96)
FLOOR = 14


def build(tr=None, verbose=True):
    tr = tr or paths.deep()
    rng = np.random.default_rng(41)
    w = VX.World(size=SIZE, origin=ORIGIN)
    X, Y, Z = SIZE
    ox, oy, oz = ORIGIN
    xs = np.arange(X) + ox + 0.5
    ys = np.arange(Y) + oy + 0.5
    GX, GY = np.meshgrid(xs, ys, indexing='ij')

    # -- the cavern: solid deepslate, hollowed out along the track ----------------------------------------------
    pts = np.array([tr.pos(s)[:2] for s in np.arange(0.0, tr.length, 2.0)])
    d = np.full(GX.shape, 1e9)
    for p in pts:
        d = np.minimum(d, np.hypot(GX - p[0], GY - p[1]))
    half = 40.0 + 8.0 * fbm2d(GX / 30.0, GY / 30.0, octaves=3, seed=5)
    open_ = d < half
    floor = FLOOR + np.round(3.0 * fbm2d(GX / 18.0, GY / 18.0, octaves=3, seed=6)).astype(int)
    # a chasm under the first stretch of the bridge
    chasm = (d < 14.0 + 4.0 * fbm2d(GX / 12.0, GY / 12.0, octaves=2, seed=8)) & (GY < -30.0)
    floor = np.where(chasm, 3, floor)
    # the city's heart is a raised terrace, level with the track, round the portal
    Pp = tr.pos(tr.marks['portal'])
    zr = int(np.floor(Pp[2] - 1.02))
    terrace = np.hypot(GX - Pp[0], GY - Pp[1]) < 30.0 + 5.0 * fbm2d(GX / 14.0, GY / 14.0, octaves=2, seed=9)
    floor = np.where(terrace, zr - 1, floor)
    ceil = 76 - np.round(10.0 * fbm2d(GX / 22.0, GY / 22.0, octaves=3, seed=7) + 14.0 * np.clip(d / half, 0, 1) ** 3)
    ceil = ceil.astype(int)
    zs = np.arange(Z) + oz
    ds_ = BL.B['deepslate']
    for k, z in enumerate(zs):
        solid = ~open_ | (z < floor) | (z >= ceil)
        w.ids[:, :, k] = np.where(solid, ds_, 0)
    w.ids[:, :, 0] = BL.B['bedrock']
    w.floor = floor
    if verbose:
        print('[deep] cavern', flush=True)

    _sculk(w, rng, floor, open_)
    _bridge(w, tr, rng)
    _columns(w, tr, rng)
    _tower(w, tr)
    _houses(w, tr, rng)
    clear_track(w, tr, keep=())
    _portal(w, tr)
    if verbose:
        print('[deep] city', flush=True)
    return w


def _sculk(w, rng, floor, open_):
    """Sculk spreading over the cavern floor, with sensors and shriekers growing out of it."""
    X, Y, _ = w.size
    ox, oy, oz = w.origin
    xs = np.arange(X) + ox + 0.5
    ys = np.arange(Y) + oy + 0.5
    GX, GY = np.meshgrid(xs, ys, indexing='ij')
    m = fbm2d(GX / 16.0, GY / 16.0, octaves=3, seed=12) > 0.05
    sk = BL.B['sculk']
    ii, jj = np.nonzero(m & open_)
    for i, j in zip(ii, jj):
        k = int(floor[i, j]) - 1 - oz
        if 0 <= k < w.size[2] and w.ids[i, j, k] == BL.B['deepslate']:
            w.ids[i, j, k] = sk
            r = rng.random()
            if r < 0.035:
                w.ids[i, j, k + 1] = BL.B['sculk_sensor']
            elif r < 0.045:
                w.ids[i, j, k + 1] = BL.B['sculk_shrieker']


def _bridge(w, tr, rng):
    """A deepslate-brick deck under the track with low polished walls, on tile piers down to the floor."""
    for s in np.arange(0.0, tr.marks['portal'] + 2.0, 0.5):
        P = tr.pos(s)
        T, R, U = tr.frame(s)
        rh = np.array([R[0], R[1], 0.0])
        rh /= np.linalg.norm(rh)
        zd = int(np.floor(P[2] - 1.02)) - 1
        for a in np.arange(-3.5, 3.51, 0.5):
            x, y = int(np.floor(P[0] + rh[0] * a)), int(np.floor(P[1] + rh[1] * a))
            w.set(x, y, zd, 'deepslate_bricks')
            if abs(a) >= 3.4:
                w.set(x, y, zd + 1, 'polished_deepslate')
                # soul lanterns along the parapets: a line of cold light leading the way to the portal
                if int(s * 2) % 14 == 0:
                    w.set(x, y, zd + 2, 'soul_lantern')
        if int(s * 2) % 18 == 0:
            for a in (-2.5, 2.5):
                x, y = int(np.floor(P[0] + rh[0] * a)), int(np.floor(P[1] + rh[1] * a))
                for z in range(2, zd):
                    if w.get(x, y, z) in ('air', 'sculk', 'deepslate'):
                        w.set(x, y, z, 'deepslate_tiles')


def _columns(w, tr, rng):
    """Rows of columns either side of the way, capped with chiseled deepslate, soul lanterns hanging from them."""
    w.lanterns = []
    for s in np.arange(6.0, tr.marks['portal'] - 8.0, 11.0):
        P = tr.pos(s)
        T, R, U = tr.frame(s)
        rh = np.array([R[0], R[1], 0.0])
        rh /= np.linalg.norm(rh)
        for side in (-1, 1):
            c = P[:2] + rh[:2] * side * 9.0
            x, y = int(np.floor(c[0])), int(np.floor(c[1]))
            top = int(P[2]) + 12
            for dx in (0, 1):
                for dy in (0, 1):
                    for z in range(2, top):
                        w.set(x + dx, y + dy, z, 'deepslate_tiles' if z % 6 else 'polished_deepslate')
            for dx in (-1, 0, 1, 2):
                for dy in (-1, 0, 1, 2):
                    w.set(x + dx, y + dy, top, 'chiseled_deepslate')
            # a lantern hanging on the inner side
            lx, ly = int(np.floor(c[0] - rh[0] * side * 2.0)), int(np.floor(c[1] - rh[1] * side * 2.0))
            w.set(lx, ly, top - 1, 'chain')
            w.set(lx, ly, top - 2, 'soul_lantern', BL.WALL)
            w.lanterns.append((lx + 0.5, ly + 0.5, top - 1.5))


def _tower(w, tr):
    """A tower of deepslate tiles inside the bend, banded with polished deepslate, soul torches round its crown."""
    P = tr.pos(tr.marks['tower'] - 12.0)
    T, R, U = tr.frame(tr.marks['tower'] - 12.0)
    rh = np.array([R[0], R[1], 0.0])
    rh /= np.linalg.norm(rh)
    c = P[:2] + rh[:2] * 12.0
    cx, cy = int(c[0]), int(c[1])
    for dx in range(-5, 6):
        for dy in range(-5, 6):
            r = np.hypot(dx, dy)
            if r > 5.2:
                continue
            for z in range(2, 70):
                if r > 4.2 or z % 9 == 0:
                    w.set(cx + dx, cy + dy, z, 'polished_deepslate' if z % 9 == 0 else 'deepslate_tiles')
    for (dx, dy) in ((6, 0), (-6, 0), (0, 6), (0, -6)):
        for z in (30, 48):
            w.set(cx + dx, cy + dy, z, 'soul_torch')
    w.tower = (cx + 0.5, cy + 0.5)


def _houses(w, tr, rng):
    """Low deepslate-brick buildings along the sides, dark doorways, gray wool floors."""
    for s in np.arange(10.0, tr.marks['approach'], 17.0):
        P = tr.pos(s)
        T, R, U = tr.frame(s)
        rh = np.array([R[0], R[1], 0.0])
        rh /= np.linalg.norm(rh)
        for side in (-1, 1):
            c = P[:2] + rh[:2] * side * rng.uniform(17.0, 24.0)
            x0, y0 = int(c[0]) - 3, int(c[1]) - 4
            fl = int(w.floor[x0 - w.origin[0], y0 - w.origin[1]]) if 0 <= x0 - w.origin[0] < w.size[0] and \
                0 <= y0 - w.origin[1] < w.size[1] else FLOOR
            h = int(rng.integers(5, 9))
            for x in range(x0, x0 + 7):
                for y in range(y0, y0 + 8):
                    edge = x in (x0, x0 + 6) or y in (y0, y0 + 7)
                    for z in range(fl, fl + h):
                        if edge:
                            w.set(x, y, z, 'deepslate_bricks')
                        else:
                            w.set(x, y, z, 'air')
                    w.set(x, y, fl - 1, 'gray_wool' if not edge else 'polished_deepslate')
                    w.set(x, y, fl + h, 'polished_deepslate')
            # a doorway facing the way
            dx = x0 + 6 if side < 0 else x0
            for z in range(fl, fl + 3):
                w.set(dx, y0 + 3, z, 'air')
                w.set(dx, y0 + 4, z, 'air')


def _portal(w, tr):
    """The ancient city's great portal, the track running through its opening: a frame of deepslate bricks and tiles
    shaped like a warden's head (the two tendrils rising from its top), the opening lined with reinforced deepslate.
    The surface is a prop (it wakes as the cart comes)."""
    s_p = tr.marks['portal']
    P = tr.pos(s_p)
    T, R, U = tr.frame(s_p)
    th = np.array([T[0], T[1], 0.0])
    th /= np.linalg.norm(th)
    rt = np.array([th[1], -th[0], 0.0])
    zr = int(np.floor(P[2] - 1.02))                     # the rail bed's level
    base = P.copy()

    def put(u, v, t, name):
        q = base + rt * u + th * t
        w.set(int(np.floor(q[0])), int(np.floor(q[1])), zr + v, name)

    for t in (-1, 0, 1):
        for u in range(-14, 15):
            for v in range(-6, 18):
                au = abs(u)
                in_head = (au <= 13 and v <= 13 - max(0, au - 8) * 1.2) or (au <= 9 and v <= 15)
                opening = au <= 6 and -1 <= v <= 9
                lining = (au <= 7 and -2 <= v <= 10) and not opening
                if opening:
                    continue
                if lining:
                    put(u, v, t, 'reinforced_deepslate')
                elif in_head:
                    put(u, v, t, 'deepslate_tiles' if (u + v) % 5 else 'deepslate_bricks')
            # the tendrils: two horns curving out and up from the crown
            for side in (-1, 1):
                for k in range(12):
                    u = side * int(round(8 + k * 0.45 + 0.03 * k * k))
                    for du in (0, side):
                        put(u + du, 15 + k, t, 'deepslate_bricks' if k % 3 else 'polished_deepslate')
    # a paved plaza in front of it
    for t in range(-14, -1):
        for u in range(-12, 13):
            put(u, -2, t, 'polished_deepslate' if (u + t) % 4 else 'chiseled_deepslate')
    for side in (-1, 1):
        for t in (-12, -6):
            put(side * 10, -1, t, 'soul_lantern')
    w.gate = dict(center=np.array([P[0], P[1], zr + 4.5]), width=13, height=11,
                  normal=(float(th[0]), float(th[1]), 0.0))
