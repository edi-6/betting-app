"""The Sift, built around the track: the new dimension, as Mojang showed it at Minecraft Live (September 2026).

A turquoise sky over pink and red land: the cart comes out of a portal on a siftslate cliff (red-orange stone in
strata) and drops into the Singer's Meadow (red sculk under pink grass, tall blue grass, green, yellow and pale-blue
flowers, white trees hung with pale-blue vines, coral-pink blocks stacked into bushes and towers, tiered siftslate
mesas wrapped in green healthy sculk). It runs low over a lake of ichor, into the edge of the Carapace (sand, walls
of blue stone, the ribs of a colossal fossil arching over the track), and into a rift that leads home.
"""
import numpy as np

import blocks as BL
import paths
import voxel as VX
from noise import fbm2d
from world_over import clear_track

ORIGIN = (-150, -180, 0)
SIZE = (260, 330, 128)
MEADOW = 44           # the meadow's ground (the track runs a few blocks over it)
ICHOR = 42            # the lake's surface


def build(tr=None, verbose=True):
    tr = tr or paths.sift()
    rng = np.random.default_rng(29)
    w = VX.World(size=SIZE, origin=ORIGIN)
    X, Y, Z = SIZE
    ox, oy, oz = ORIGIN
    xs = np.arange(X) + ox + 0.5
    ys = np.arange(Y) + oy + 0.5
    GX, GY = np.meshgrid(xs, ys, indexing='ij')

    # -- the land -------------------------------------------------------------------------------------------------
    pts = np.array([tr.pos(s)[:2] for s in np.arange(0.0, tr.length, 2.0)])
    ss = np.arange(0.0, tr.length, 2.0)
    d = np.full(GX.shape, 1e9)
    near_s = np.zeros(GX.shape)
    for p, s in zip(pts, ss):
        dd = np.hypot(GX - p[0], GY - p[1])
        closer = dd < d
        d = np.where(closer, dd, d)
        near_s = np.where(closer, s, near_s)
    H = MEADOW + 3.0 * fbm2d(GX / 22.0, GY / 22.0, octaves=3, seed=3) + 2.0 * np.clip((d - 20.0) / 30.0, 0, 1)
    # the cliff the cart starts on (the south end)
    lip = -126.0 + 5.0 * fbm2d(GX / 20.0, np.zeros_like(GX) + 1.7, octaves=2, seed=4)
    cliff = np.clip((lip - GY) / 2.5, 0, 1)
    H = np.maximum(H, cliff * 86.0 + (1 - cliff) * H)
    # tiered mesas away from the track: flat-topped, stepping down in tiers of 6
    mesa = fbm2d(GX / 40.0, GY / 40.0, octaves=3, seed=11)
    tall = np.clip((mesa - 0.12) * 3.2, 0, 1) * np.clip((d - 26.0) / 10.0, 0, 1)
    Hm = MEADOW + np.floor(tall * 58.0 / 6.0) * 6.0
    H = np.maximum(H, Hm)
    # the lake of ichor
    lake = (np.hypot(GX - tr.pos(tr.marks['lake'] + 12.0)[0], GY - tr.pos(tr.marks['lake'] + 12.0)[1]) <
            24.0 + 6.0 * fbm2d(GX / 12.0, GY / 12.0, octaves=2, seed=5))
    H = np.where(lake, ICHOR - 3.0, H)
    Hi = np.round(H).astype(int)
    w.H = Hi
    carapace = near_s > tr.marks['fossil'] - 14.0
    zs = np.arange(Z) + oz
    B = BL.B
    for k, z in enumerate(zs):
        below = z < Hi
        top = z == Hi - 1
        ids = np.where(below, B['siftslate'], 0)
        ids = np.where(top & ~carapace & (Hi <= MEADOW + 6), B['sift_grass_block'], ids)
        ids = np.where(top & carapace, B['sift_sand'], ids)
        ids = np.where(below & (z >= Hi - 4) & (z < Hi - 1) & ~carapace & (Hi <= MEADOW + 6), B['red_sculk'], ids)
        ids = np.where(lake & (z >= Hi) & (z <= ICHOR - 1), B['ichor'], ids)
        w.ids[:, :, k] = ids
    w.ids[:, :, 0] = B['bedrock']
    # green healthy sculk wrapping the mesas' tiers, and patches of it (and the orange and pink) on the meadow
    tops = (Hi > MEADOW + 6)
    ii, jj = np.nonzero(tops)
    for i, j in zip(ii, jj):
        w.ids[i, j, Hi[i, j] - 1 - oz] = B['healthy_sculk']
    patch = fbm2d(GX / 9.0, GY / 9.0, octaves=2, seed=13)
    for name, lo, hi_ in (('healthy_sculk', 0.32, 0.40), ('healthy_sculk_orange', 0.40, 0.46),
                          ('healthy_sculk_pink', 0.46, 1.0)):
        m = (patch > lo) & (patch <= hi_) & ~tops & ~lake & ~carapace
        ii, jj = np.nonzero(m)
        w.ids[ii, jj, Hi[ii, jj] - 1 - oz] = B[name]
    if verbose:
        print('[sift] land', flush=True)

    _plants(w, rng, Hi, lake, carapace, tops, d)
    _trees(w, tr, rng, Hi, lake, carapace, tops, d)
    _coral(w, rng, Hi, lake, carapace, tops, d)
    _walls(w, rng, Hi, carapace, d)
    clear_track(w, tr, keep=())
    _fossil(w, tr)
    _portals(w, tr)
    _supports(w, tr)
    if verbose:
        print('[sift] features', flush=True)
    return w


def _surface(w, i, j, Hi):
    return int(Hi[i, j]) - w.origin[2]


def _plants(w, rng, Hi, lake, carapace, tops, d):
    """Tall blue grass thick over the meadow, pink grass, three kinds of flowers; red and yellow tufts on the sand."""
    X, Y, _ = w.size
    r = rng.random((X, Y))
    kind = rng.random((X, Y))
    B = BL.B
    for i in range(X):
        for j in range(Y):
            if lake[i, j] or tops[i, j]:
                continue
            k = _surface(w, i, j, Hi)
            if k >= w.size[2] - 1 or w.ids[i, j, k] != 0:
                continue
            if carapace[i, j]:
                if r[i, j] < 0.05:
                    w.ids[i, j, k] = B['pink_grass'] if kind[i, j] < 0.5 else B['yellow_flowers']
                continue
            if r[i, j] < 0.42:
                v = kind[i, j]
                w.ids[i, j, k] = (B['blue_grass'] if v < 0.62 else B['pink_grass'] if v < 0.84 else
                                  B['green_flower'] if v < 0.90 else B['yellow_flowers'] if v < 0.95 else
                                  B['pale_blue_flower'])


def _trees(w, tr, rng, Hi, lake, carapace, tops, d):
    """White trees, some towering, their round pale-blue crowns hung with vines."""
    X, Y, _ = w.size
    n = 0
    w.trees = []
    for _ in range(900):
        i, j = int(rng.integers(3, X - 3)), int(rng.integers(3, Y - 3))
        if lake[i, j] or carapace[i, j] or tops[i, j] or d[i, j] < 7.0 or d[i, j] > 70.0:
            continue
        if any(abs(i - a) < 7 and abs(j - b) < 7 for (a, b) in w.trees):
            continue
        k0 = _surface(w, i, j, Hi)
        h = int(rng.integers(9, 17)) if rng.random() < 0.75 else int(rng.integers(18, 26))
        for k in range(k0, min(k0 + h, w.size[2] - 6)):
            w.ids[i, j, k] = BL.B['sift_log']
        rad = 3 if h < 17 else 4
        top = min(k0 + h, w.size[2] - 6)
        for di in range(-rad, rad + 1):
            for dj in range(-rad, rad + 1):
                for dk in range(-2, 3):
                    if di * di + dj * dj + (dk * 1.6) ** 2 > rad * rad + 1:
                        continue
                    a, b, c = i + di, j + dj, top + dk
                    if 0 <= a < X and 0 <= b < Y and 0 <= c < w.size[2] and w.ids[a, b, c] == 0:
                        w.ids[a, b, c] = BL.B['sift_leaves']
        # vines hanging from the crown's underside
        for di in range(-rad, rad + 1):
            for dj in range(-rad, rad + 1):
                if rng.random() > 0.35 or di * di + dj * dj > rad * rad:
                    continue
                a, b = i + di, j + dj
                if not (0 <= a < X and 0 <= b < Y):
                    continue
                c = top - 3
                for _ in range(int(rng.integers(2, 7))):
                    if 0 <= c < w.size[2] and w.ids[a, b, c] == 0:
                        w.ids[a, b, c] = BL.B['sift_vines']
                    c -= 1
        w.trees.append((i, j))
        n += 1
        if n >= 70:
            break


def _coral(w, rng, Hi, lake, carapace, tops, d):
    """Coral-pink blocks stacked into bushes and towers."""
    X, Y, _ = w.size
    n = 0
    for _ in range(600):
        i, j = int(rng.integers(3, X - 3)), int(rng.integers(3, Y - 3))
        if lake[i, j] or carapace[i, j] or tops[i, j] or d[i, j] < 6.0 or d[i, j] > 80.0:
            continue
        k0 = _surface(w, i, j, Hi)
        if rng.random() < 0.45:
            # a tower: a knobbly column, sometimes two wide, with lumps
            h = int(rng.integers(6, 20))
            wd = 1 if rng.random() < 0.5 else 2
            for k in range(k0, min(k0 + h, w.size[2] - 2)):
                for di in range(wd):
                    for dj in range(wd):
                        w.ids[i + di, j + dj, k] = BL.B['coral_block']
                if rng.random() < 0.25:
                    a, b = i + int(rng.integers(-1, 2 + wd - 1)), j + int(rng.integers(-1, 2 + wd - 1))
                    if w.ids[a, b, k] == 0:
                        w.ids[a, b, k] = BL.B['coral_block']
        else:
            # a bush: a lumpy mound
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    for dk in range(0, 2):
                        if rng.random() < 0.7 - 0.3 * dk:
                            a, b, c = i + di, j + dj, k0 + dk
                            if w.ids[a, b, c] == 0 or BL.SHAPE[w.ids[a, b, c]] == 'cross':
                                w.ids[a, b, c] = BL.B['coral_block']
        n += 1
        if n >= 120:
            break


def _walls(w, rng, Hi, carapace, d):
    """The Carapace: walls of blue stone jutting from the sand."""
    X, Y, _ = w.size
    n = 0
    for _ in range(400):
        i, j = int(rng.integers(3, X - 14)), int(rng.integers(3, Y - 14))
        if not carapace[i, j] or d[i, j] < 7.0 or d[i, j] > 60.0:
            continue
        k0 = _surface(w, i, j, Hi) - 1
        ln = int(rng.integers(5, 12))
        h = int(rng.integers(6, 15))
        along_i = rng.random() < 0.5
        for t in range(ln):
            a, b = (i + t, j) if along_i else (i, j + t)
            hh = h - abs(t - ln // 2) // 2
            for k in range(k0, min(k0 + hh, w.size[2] - 1)):
                w.ids[a, b, k] = BL.B['blue_stone']
        n += 1
        if n >= 22:
            break


def _fossil(w, tr):
    """The ribs of a colossal fossil arching over the track: a spine above it and ribs every four blocks curving down
    into the sand either side, then a skull half buried beyond."""
    s0, s1 = tr.marks['fossil'] + 2.0, tr.marks['fossil'] + 30.0
    bone = BL.B['bone_block']
    for s in np.arange(s0, s1, 0.5):
        P = tr.pos(s)
        T, R, U = tr.frame(s)
        rh = np.array([R[0], R[1], 0.0])
        rh /= np.linalg.norm(rh)
        spine = P + np.array([0.0, 0.0, 11.0])
        w.set(int(np.floor(spine[0])), int(np.floor(spine[1])), int(np.floor(spine[2])), bone)
        if abs((s - s0) % 4.0) < 0.25:
            for side in (-1, 1):
                for a in np.linspace(0.0, np.pi * 0.62, 40):
                    q = spine + rh * side * 9.5 * np.sin(a) - np.array([0.0, 0.0, 1.0]) * (11.0 - 11.0 * np.cos(a)) * 1.05
                    q = q + rh * side * 1.5 * (a / np.pi)
                    for dz in (0, 1):
                        w.set(int(np.floor(q[0])), int(np.floor(q[1])), int(np.floor(q[2])) + dz, bone)
                # down into the sand
                q = spine + rh * side * 10.5
                for z in range(int(P[2]) - 8, int(spine[2]) - 9):
                    w.set(int(np.floor(q[0])), int(np.floor(q[1])), z, bone)
    # the skull, half buried, beside the track beyond the ribs
    P = tr.pos(s1 + 6.0)
    T, R, U = tr.frame(s1 + 6.0)
    rh = np.array([R[0], R[1], 0.0])
    rh /= np.linalg.norm(rh)
    c = P + rh * 13.0
    cx, cy, cz = int(c[0]), int(c[1]), int(P[2]) - 4
    for dx in range(-4, 5):
        for dy in range(-5, 6):
            for dz in range(0, 8):
                inside = abs(dx) <= 3 and abs(dy) <= 4 and 1 <= dz <= 6
                shell = (abs(dx) <= 4 and abs(dy) <= 5 and dz <= 7) and not inside
                eye = dz in (4, 5) and abs(dy) in (2, 3) and dx == -4
                if shell and not eye:
                    w.set(cx + dx, cy + dy, cz + dz, bone)
    w.fossil = (s0, s1)


def _portals(w, tr):
    """The portal the cart comes out of on the cliff (reinforced deepslate lined, siftslate bricks round it), and the
    rift at the end of the ride (the same, rougher). Their surfaces are props."""
    w.portals = {}
    for name, s in (('arrival', 0.4), ('rift', tr.marks['rift'] + 0.5)):
        P = tr.pos(s)
        T, R, U = tr.frame(s)
        th = np.array([T[0], T[1], 0.0])
        th /= np.linalg.norm(th)
        rt = np.array([th[1], -th[0], 0.0])
        zr = int(np.floor(P[2] - 1.02))
        for u in range(-5, 6):
            for v in range(-1, 10):
                au = abs(u)
                opening = au <= 3 and 0 <= v <= 7
                if opening:
                    for t in (-1, 0):
                        q = P + rt * u + th * t
                        w.set(int(np.floor(q[0])), int(np.floor(q[1])), zr + v, 'air')
                    continue
                lining = au <= 4 and v <= 8
                q = P + rt * u
                w.set(int(np.floor(q[0])), int(np.floor(q[1])), zr + v,
                      'reinforced_deepslate' if lining else 'siftslate_bricks')
        w.portals[name] = dict(center=np.array([P[0], P[1], zr + 4.5]), width=7, height=8,
                               normal=(float(th[0]), float(th[1]), 0.0))


def _supports(w, tr):
    """Siftslate-brick piers under the track where it runs clear of the ground."""
    for s in np.arange(tr.marks['drop'] + 40.0, tr.length - 2.0, 7.0):
        P = tr.pos(s)
        if abs(tr.bank[int(s / 0.1)]) > 30:
            continue
        x, y = int(np.floor(P[0])), int(np.floor(P[1]))
        for z in range(1, int(np.floor(P[2] - 1.6)) + 1):
            if w.get(x, y, z) in ('air', 'ichor', 'blue_grass', 'pink_grass', 'green_flower', 'yellow_flowers',
                                  'pale_blue_flower'):
                w.set(x, y, z, 'siftslate_bricks')
