"""The valley in the Sift that the flight circles, built round the flight path.

From the spire at the valley's south-west corner the view runs north-east over all of it: the Singer's Meadow below
(pink grass on red sculk, tall blue and pink grass, flowers, patches of healthy sculk), a stream of ichor winding
through it from the lake to the spire's foot, the grove of white trees hung with pale-blue vines along the west side
(the flight threads it under the crowns), the Carapace at the far end (sand, walls of blue stone, a colossal fossil
whose ribs arch over the flight path), the lake of ichor filling the east side under a sunlit cliff that falls pour
from, and coral towers along the south side. Ridges and mesas of banded siftslate close the valley in: low in the
west where the sun comes from, tall in the east. Everything near the flight keeps its distance: its corridor is
carved clear.
"""
import numpy as np

import blocks as BL
import flight as FL
import voxel as VX
from noise import fbm2d

ORIGIN = (-80, -100, 0)
SIZE = (300, 320, 128)
MEADOW = 44           # the meadow's ground level (the top face)
ICHOR = 44            # the lake's surface
CENTER = (75.0, 55.0)  # the middle of the loop
LAKE = (116.0, 70.0, 40.0, 50.0)   # the lake's ellipse: centre x, y, radius x, radius y
EAST_CLIFF = 166.0    # the eastern mesa's cliff (the falls pour from it)
FALLS = ((164.0, 70.0), (164.0, 38.0))
# the stream, from the lake's south-west shore to the spire's foot
STREAM = [(86.0, 34.0), (70.0, 30.0), (58.0, 38.0), (44.0, 32.0), (32.0, 20.0), (24.0, 22.0), (16.0, 12.0),
          (8.0, 9.0)]
# strata of the cliffs, bottom up (repeating)
STRATA = (['siftslate'] * 3 + ['red_sculk'] + ['siftslate'] * 2 + ['healthy_sculk_orange'] + ['siftslate'] * 4 +
          ['coral_block'] + ['siftslate'] * 3 + ['red_sculk'] * 2 + ['siftslate'] * 2 + ['blue_stone'] +
          ['siftslate'] * 4)


def _inside(poly, X, Y):
    """Points (X, Y) inside the closed polygon (even-odd rule)."""
    inside = np.zeros(X.shape, bool)
    px, py = poly[:, 0], poly[:, 1]
    qx, qy = np.roll(px, 1), np.roll(py, 1)
    for x0, y0, x1, y1 in zip(px, py, qx, qy):
        c = ((y0 > Y) != (y1 > Y)) & (X < (x1 - x0) * (Y - y0) / (y1 - y0 + 1e-12) + x0)
        inside ^= c
    return inside


def _terrace(h):
    """Heights stepped onto uneven benches (cliffs of three to nine blocks)."""
    steps = np.array([0, 4, 9, 13, 19, 23, 30, 34, 41, 46, 52, 58, 64])
    k = np.searchsorted(steps, h, side='right') - 1
    return steps[np.clip(k, 0, len(steps) - 1)].astype(float)


def _polyline_dist(P, GX, GY):
    """Distance from each grid point to a polyline, and the position along it (0..1)."""
    P = np.asarray(P, float)
    best = np.full(GX.shape, 1e9)
    along = np.zeros(GX.shape)
    L = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    for k in range(len(P) - 1):
        a, b = P[k], P[k + 1]
        ab = b - a
        u = np.clip(((GX - a[0]) * ab[0] + (GY - a[1]) * ab[1]) / (ab @ ab), 0, 1)
        dd = np.hypot(GX - (a[0] + u * ab[0]), GY - (a[1] + u * ab[1]))
        m = dd < best
        best = np.where(m, dd, best)
        along = np.where(m, (L[k] + u * np.linalg.norm(ab)) / L[-1], along)
    return best, along


def _smooth_path(P, n=12):
    """A Catmull-Rom through the stream's points."""
    return FL._catmull_rom([(x, y, 0.0) for x, y in P], n)[:, :2]


def build(fl=None, verbose=True):
    fl = fl or FL.Flight()
    rng = np.random.default_rng(12)
    w = VX.World(size=SIZE, origin=ORIGIN)
    X, Y, Z = SIZE
    ox, oy, oz = ORIGIN
    xs = np.arange(X) + ox + 0.5
    ys = np.arange(Y) + oy + 0.5
    GX, GY = np.meshgrid(xs, ys, indexing='ij')
    B = BL.B

    # -- where each column is relative to the flight: its distance to the path and the arc length there -------------
    ss = np.arange(0.0, fl.length, 1.5)
    pts = np.array([fl.pos(s) for s in ss])
    d = np.full(GX.shape, 1e9)
    near = np.zeros(GX.shape)
    for p, s in zip(pts, ss):
        dd = np.hypot(GX - p[0], GY - p[1])
        m = dd < d
        d = np.where(m, dd, d)
        near = np.where(m, s, near)
    w.dist, w.near = d, near
    loop = _inside(fl.P[::5, :2], GX, GY)
    s_of = {k: fl.s_ctrl[k] for k in range(len(FL.CONTROL))}
    grove = (near > s_of[6] - 6) & (near < s_of[11] + 4) & (d < 30)
    carapace = (near >= s_of[11] + 4) & (near < s_of[15] - 6) & ~loop
    carapace |= (GY > 118) & (GX > 70) & (GX < EAST_CLIFF - 4)
    towers = (near >= s_of[19]) & (near < s_of[23] + 6)
    spire_d = np.hypot(GX - FL.SPIRE[0], GY - FL.SPIRE[1])

    # -- the land ---------------------------------------------------------------------------------------------------
    H = MEADOW + 1.6 * fbm2d(GX / 28.0, GY / 28.0, octaves=3, seed=3) + 0.8 * fbm2d(GX / 9.0, GY / 9.0, octaves=2,
                                                                                    seed=4)
    # the rim: ridges and mesas outside the loop, low in the west (towards the sun), tall in the east
    ang = np.arctan2(GY - CENTER[1], GX - CENTER[0])
    rim_h = (30.0 + 12.0 * np.cos(ang) + 2.0 * np.sin(ang)) - 12.0 * np.clip(-np.cos(ang) - 0.3, 0, 1)
    noise = fbm2d(GX / 40.0, GY / 40.0, octaves=4, seed=11)
    rise = np.clip((d - 24.0) / 16.0, 0, 1) * (~loop)
    rise = np.maximum(rise, np.clip((d - 40.0) / 16.0, 0, 1))
    rim = rise * rim_h * (0.75 + 0.6 * noise)
    rim = np.where(spire_d < 14.0, 0.0, rim)
    H = np.maximum(H, MEADOW + _terrace(rim))
    # the eastern mesa: a flat top, a cliff facing the sun
    edge = EAST_CLIFF + 3.0 * fbm2d(GX / 10.0, GY / 10.0, octaves=2, seed=21)
    east = (GX > edge) & (GY > -40) & (GY < 160)
    H = np.where(east, np.maximum(H, MEADOW + 44.0 + 3.0 * np.round(fbm2d(GX / 16.0, GY / 16.0, seed=22))), H)
    # the lake, and the stream feeding it (a sunken channel)
    lx, ly, rx, ry = LAKE
    wob = 5.0 * fbm2d(GX / 12.0, GY / 12.0, octaves=2, seed=5)
    lake = (((GX - lx) / rx) ** 2 + ((GY - ly) / ry) ** 2 < (1.0 + wob / 40.0) ** 2) & ~east
    lake |= (GX > 130) & (GX < edge) & (GY > 30) & (GY < 112 + wob) & (d < 22)
    shore = np.hypot((GX - lx) / rx, (GY - ly) / ry)
    H = np.where(lake, ICHOR - 5.0, H)
    near_lake = (~lake) & (shore < 1.25)
    H = np.where(near_lake, np.maximum(H, ICHOR + 0.4), H)
    sp = _smooth_path(STREAM)
    sd, sal = _polyline_dist(sp, GX, GY)
    width = 1.6 + 1.4 * sal
    stream = (sd < width) & ~lake
    H = np.where((sd < width + 4.0) & ~lake & (H < MEADOW + 0.6), MEADOW + 0.6, H)
    Hi = np.round(H).astype(int)
    Hi = np.where(stream, MEADOW - 1, Hi)
    w.H = Hi
    high = Hi > MEADOW + 3
    zs = np.arange(Z) + oz
    band_off = np.round(2.0 * fbm2d(GX / 30.0, GY / 30.0, octaves=2, seed=17)).astype(int)
    strata_ids = np.array([B[n] for n in STRATA])
    for k, z in enumerate(zs):
        below = z < Hi
        top = z == Hi - 1
        ids = np.where(below, B['siftslate'], 0)
        ids = np.where(below & high, strata_ids[(z + band_off) % len(STRATA)], ids)
        meadow = ~carapace & ~high
        ids = np.where(below & (z >= Hi - 4) & (z < Hi - 1) & meadow, B['red_sculk'], ids)
        ids = np.where(top & meadow, B['sift_grass_block'], ids)
        ids = np.where(top & carapace & ~high, B['sift_sand'], ids)
        ids = np.where(below & (z >= Hi - 3) & carapace & ~high, B['sift_sand'], ids)
        ids = np.where(top & high, B['healthy_sculk'], ids)
        ids = np.where(lake & (z >= Hi) & (z < ICHOR), B['ichor'], ids)
        ids = np.where(stream & (z >= Hi - 1) & (z < MEADOW), B['ichor'], ids)
        w.ids[:, :, k] = ids
    w.ids[:, :, 0] = B['bedrock']
    # patches of healthy sculk (green, orange, pink) on the meadow, and red sculk along the stream's banks
    patch = fbm2d(GX / 9.0, GY / 9.0, octaves=2, seed=13)
    patch2 = fbm2d(GX / 5.0, GY / 5.0, octaves=2, seed=14)
    free = ~high & ~lake & ~carapace & ~stream
    for name, m in (('healthy_sculk', free & (patch > 0.30) & (patch <= 0.38)),
                    ('healthy_sculk_orange', free & (patch > 0.38) & (patch <= 0.45)),
                    ('healthy_sculk_pink', free & (patch > 0.45)),
                    ('red_sculk', free & (sd < width + 1.6) & (patch2 > -0.1)),
                    ('healthy_sculk_pink', free & (patch2 > 0.52))):
        ii, jj = np.nonzero(m)
        w.ids[ii, jj, Hi[ii, jj] - 1 - oz] = B[name]
    if verbose:
        print('[valley] land', flush=True)

    masks = dict(lake=lake, carapace=carapace, high=high, grove=grove, towers=towers, stream=stream, loop=loop,
                 near_lake=near_lake, dist=d)
    _spire(w)
    _falls(w)
    _plants(w, rng, Hi, masks, patch)
    _trees(w, fl, rng, Hi, masks, d)
    _coral(w, fl, rng, Hi, masks, d)
    _walls(w, rng, Hi, masks, d)
    _boulders(w, rng, Hi, masks, d)
    _carve(w, fl)
    s0, s1 = fl.s_at(5.55), fl.s_at(6.50)
    _fossil(w, fl, s0, s1)
    _carve(w, fl, lateral=1.6, above=1.6, below=1.4, only=(s0 - 4.0, s1 + 12.0))
    if verbose:
        print('[valley] features', flush=True)
    w.zones = dict(grove=(s_of[6], s_of[11]), carapace=(s_of[11], s_of[15]), lake=(s_of[15], s_of[19]),
                   towers=(s_of[19], s_of[23]))
    w.stream = sp
    return w


def _spire(w):
    """The spire: a tapering, leaning column of banded siftslate, green creeping up it, vines hanging off its faces;
    the player stands at the north-east corner of its flat top."""
    cx, cy = FL.SPIRE[0], FL.SPIRE[1]
    top = int(FL.SPIRE[2])
    strata = np.array([BL.B[n] for n in STRATA])
    for z in range(1, top):
        u = (top - z) / (top - MEADOW)
        half = 2.9 + 6.5 * np.clip(u, 0, 1.3) ** 1.6
        wob = 0.7 * np.sin(z * 0.37) + 0.5 * np.sin(z * 0.91 + 1.0)
        lean = -2.0 * np.clip(u, 0, 1) ** 2               # the base sits a little to the south-west
        r = int(np.ceil(half + 3))
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                x, y = cx + dx, cy + dy
                qx, qy = x - (cx + lean), y - (cy + lean)
                if max(abs(qx), abs(qy)) + 0.3 * min(abs(qx), abs(qy)) <= half + wob * 0.35:
                    bid = BL.B['siftslate'] if z % 9 not in (0, 4) else strata[(z + 1) % len(strata)]
                    if z < top - 3 and (z * 7 + dx * 3 + dy * 5) % 13 == 0:
                        bid = BL.B['healthy_sculk']
                    w.set(int(np.floor(x)), int(np.floor(y)), z, int(bid))
    # the top: a flat 5x5 of grass, a tuft of pink grass and a flower away from the corner
    for x in range(-2, 3):
        for y in range(-2, 3):
            w.set(x, y, top - 1, 'sift_grass_block')
            w.set(x, y, top - 2, 'red_sculk')
            for z in range(top, top + 6):
                w.set(x, y, z, 'air')
    w.set(-2, -2, top, 'pink_grass')
    w.set(-1, -2, top, 'blue_grass')
    w.set(-2, 0, top, 'pale_blue_flower')
    # vines hanging down the north and east faces
    rng = np.random.default_rng(3)
    for _ in range(60):
        z = int(rng.integers(MEADOW + 8, top - 3))
        u = (top - z) / (top - MEADOW)
        half = 2.9 + 6.5 * u ** 1.6
        side = rng.integers(0, 2)
        off = rng.uniform(-half, half)
        x = cx + (half + 1.0 if side == 0 else off) - 2.0 * u ** 2
        y = cy + (off if side == 0 else half + 1.0) - 2.0 * u ** 2
        for k in range(int(rng.integers(2, 7))):
            if w.get(int(np.floor(x)), int(np.floor(y)), z - k) == 'air':
                w.set(int(np.floor(x)), int(np.floor(y)), z - k, 'sift_vines')


def _falls(w):
    """Ichor pouring off the eastern mesa's cliff into the lake: curtains two blocks thick in front of the cliff face,
    from notches in its lip down to the lake."""
    for fx, fy in FALLS:
        for y in range(int(fy) - 5, int(fy) + 5):
            xe = int(EAST_CLIFF) - 4
            while xe < EAST_CLIFF + 8 and w.get(xe, y, MEADOW + 30) == 'air':
                xe += 1
            for x in (xe - 2, xe - 1):
                for z in range(ICHOR - 1, MEADOW + 44):
                    if w.get(x, y, z) in ('air', 'ichor', 'healthy_sculk', 'sift_grass_block'):
                        w.set(x, y, z, 'ichor')
            for z in range(MEADOW + 42, MEADOW + 48):
                w.set(xe, y, z, 'air')
            w.set(xe, y, MEADOW + 42, 'ichor')
    w.falls = [(float(x), float(y)) for x, y in FALLS]


def _surface(w, i, j, Hi):
    return int(Hi[i, j]) - w.origin[2]


def _plants(w, rng, Hi, m, patch):
    """Tall blue and pink grass over most of the meadow, flowers in drifts, a few plants in the sand."""
    X, Y, _ = w.size
    r = rng.random((X, Y))
    kind = rng.random((X, Y))
    B = BL.B
    for i in range(X):
        for j in range(Y):
            if m['lake'][i, j] or m['stream'][i, j]:
                continue
            k = _surface(w, i, j, Hi)
            if k >= w.size[2] - 1 or w.ids[i, j, k] != 0:
                continue
            if m['high'][i, j]:
                if r[i, j] < 0.18:
                    w.ids[i, j, k] = B['blue_grass'] if kind[i, j] < 0.6 else B['pink_grass']
                continue
            if m['carapace'][i, j]:
                if r[i, j] < 0.07:
                    w.ids[i, j, k] = B['pink_grass'] if kind[i, j] < 0.5 else B['yellow_flowers']
                continue
            dens = 0.62 if m['dist'][i, j] > 14.0 else 0.34
            if r[i, j] < dens:
                v = kind[i, j]
                f = patch[i, j]
                if f > 0.2 and v < 0.5:
                    w.ids[i, j, k] = (B['yellow_flowers'] if f > 0.42 else B['pale_blue_flower'] if f > 0.3 else
                                      B['green_flower'])
                else:
                    w.ids[i, j, k] = B['blue_grass'] if v < (0.62 if m['dist'][i, j] > 14.0 else 0.4) else \
                        B['pink_grass']


def _tree(w, x, y, z0, h, rng, big=False, vines=(1, 5)):
    """A white tree: a trunk (two wide if big) with branches near the top, a crown of pale-blue leaves in lobes, and
    vines hanging from under it."""
    X, Y, Z = w.size
    ox, oy, oz = w.origin
    wd = 2 if big else 1
    top = min(z0 + h, oz + Z - 8)
    for z in range(z0, top):
        for dx in range(wd):
            for dy in range(wd):
                w.set(x + dx, y + dy, z, 'sift_log')
    lobes = [(0.0, 0.0, 0.0, 4.6 if big else 3.3)]
    for _ in range(4 if big else 2):
        a = rng.uniform(0, 2 * np.pi)
        rr = rng.uniform(3.0, 5.0) if big else rng.uniform(1.8, 2.8)
        lobes.append((np.cos(a) * rr, np.sin(a) * rr, rng.uniform(-2.5, 1.0), rng.uniform(2.6, 3.8)))
        for t in np.linspace(0, 1, 10):
            w.set(int(np.floor(x + wd / 2 + np.cos(a) * rr * t)), int(np.floor(y + wd / 2 + np.sin(a) * rr * t)),
                  int(top - 4 + 3 * t), 'sift_log')
    for (lx, ly, lz, rad) in lobes:
        cx, cy, cz = x + wd / 2 + lx, y + wd / 2 + ly, top + lz
        R = int(np.ceil(rad))
        for dx in range(-R - 1, R + 2):
            for dy in range(-R - 1, R + 2):
                for dz in range(-R, R + 1):
                    if dx * dx + dy * dy + (dz * 1.6) ** 2 <= rad * rad + rng.random() * 2.5:
                        p = (int(np.floor(cx + dx)), int(np.floor(cy + dy)), int(np.floor(cz + dz * 0.8)))
                        if w.get(*p) == 'air':
                            w.set(*p, 'sift_leaves')
        for dx in range(-R, R + 1):
            for dy in range(-R, R + 1):
                if dx * dx + dy * dy > rad * rad or rng.random() > 0.3:
                    continue
                vx, vy = int(np.floor(cx + dx)), int(np.floor(cy + dy))
                vz = int(np.floor(cz - rad * 0.6))
                while w.get(vx, vy, vz) == 'sift_leaves':
                    vz -= 1
                for _ in range(int(rng.integers(vines[0], vines[1] + 1))):
                    if w.get(vx, vy, vz) == 'air':
                        w.set(vx, vy, vz, 'sift_vines')
                    vz -= 1


def _trees(w, fl, rng, Hi, m, d):
    """The grove the flight threads (trunks well off the path, crowns meeting over it), and trees scattered over the
    meadow and on the ridges."""
    X, Y, _ = w.size
    ox, oy, oz = w.origin
    placed = []
    s_a, s_b = fl.s_ctrl[6] + 2.0, fl.s_ctrl[11] + 2.0
    side = 1
    for s in np.arange(s_a, s_b, 5.5):
        P = fl.pos(s)
        T = fl.tangent(s)
        rh = np.array([T[1], -T[0]])
        rh /= np.linalg.norm(rh)
        side = -side
        for sd in (side, -side) if rng.random() < 0.55 else (side,):
            lat = rng.uniform(6.0, 10.5)
            q = P[:2] + rh * sd * lat + T[:2] * rng.uniform(-1.5, 1.5)
            i, j = int(np.floor(q[0])) - ox, int(np.floor(q[1])) - oy
            if not (2 <= i < X - 3 and 2 <= j < Y - 3) or m['lake'][i, j] or m['stream'][i, j]:
                continue
            big = rng.random() < 0.55
            h = int(P[2] - Hi[i, j]) + int(rng.integers(8, 13))
            _tree(w, int(np.floor(q[0])), int(np.floor(q[1])), int(Hi[i, j]), h, rng, big, vines=(0, 3))
            placed.append((q[0], q[1]))
    n = 0
    for _ in range(4000):
        i, j = int(rng.integers(3, X - 4)), int(rng.integers(3, Y - 4))
        if m['lake'][i, j] or m['carapace'][i, j] or m['stream'][i, j] or m['near_lake'][i, j] or d[i, j] < 12.0:
            continue
        q = (i + ox + 0.5, j + oy + 0.5)
        if np.hypot(q[0] - FL.SPIRE[0], q[1] - FL.SPIRE[1]) < 16:
            continue
        gap = 81 if m['high'][i, j] else 49
        if any((q[0] - a) ** 2 + (q[1] - b) ** 2 < gap for (a, b) in placed):
            continue
        if m['high'][i, j] and rng.random() < 0.5:
            continue
        big = rng.random() < 0.3
        _tree(w, i + ox, j + oy, int(Hi[i, j]), int(rng.integers(9, 18 if big else 14)), rng, big)
        placed.append(q)
        n += 1
        if n >= 110:
            break
    w.trees = placed


def _coral(w, fl, rng, Hi, m, d):
    """Coral-pink towers the flight climbs between, and bushes and smaller towers dotted about the south."""
    X, Y, _ = w.size
    ox, oy, oz = w.origin
    B = BL.B

    def tower(x, y, z0, h, wd):
        for z in range(z0, z0 + h):
            ww = wd if z < z0 + h - 3 else max(1, wd - 1)
            for dx in range(ww):
                for dy in range(ww):
                    w.set(x + dx, y + dy, z, 'coral_block')
            if rng.random() < 0.3:
                a, b = x + int(rng.integers(-1, ww + 1)), y + int(rng.integers(-1, ww + 1))
                if w.get(a, b, z) == 'air':
                    w.set(a, b, z, 'coral_block')
        w.set(x, y, z0 + h, 'healthy_sculk_pink')          # a knob of pink sculk on top

    side = 1
    for s in np.arange(fl.s_ctrl[19] - 4.0, fl.s_ctrl[23] + 4.0, 7.5):
        P = fl.pos(s)
        T = fl.tangent(s)
        rh = np.array([T[1], -T[0]])
        rh /= max(np.linalg.norm(rh), 1e-6)
        side = -side
        for sd in (side, -side):
            q = P[:2] + rh * sd * rng.uniform(6.5, 11.0)
            i, j = int(np.floor(q[0])) - ox, int(np.floor(q[1])) - oy
            if 1 <= i < X - 4 and 1 <= j < Y - 4 and not m['lake'][i, j]:
                tower(int(np.floor(q[0])), int(np.floor(q[1])), int(Hi[i, j]), int(P[2] - Hi[i, j]) +
                      int(rng.integers(3, 16)), int(rng.integers(2, 4)))
    n = 0
    for _ in range(3000):
        i, j = int(rng.integers(3, X - 4)), int(rng.integers(3, Y - 4))
        if m['lake'][i, j] or m['high'][i, j] or m['stream'][i, j] or d[i, j] < 7.0 or d[i, j] > 70.0:
            continue
        south = (j + oy) < 20
        if not south and rng.random() < 0.75:
            continue
        if rng.random() < (0.45 if south else 0.15):
            tower(i + ox, j + oy, int(Hi[i, j]), int(rng.integers(5, 20)), int(rng.integers(1, 3)))
        else:
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(2):
                        if rng.random() < 0.7 - 0.3 * dz:
                            a, b, c = i + dx, j + dy, _surface(w, i, j, Hi) + dz
                            if w.ids[a, b, c] == 0 or BL.SHAPE[w.ids[a, b, c]] == 'cross':
                                w.ids[a, b, c] = B['coral_block']
        n += 1
        if n >= 150:
            break


def _walls(w, rng, Hi, m, d):
    """The Carapace's walls of blue stone jutting from the sand."""
    X, Y, _ = w.size
    n = 0
    for _ in range(2000):
        i, j = int(rng.integers(3, X - 16)), int(rng.integers(3, Y - 16))
        if not m['carapace'][i, j] or m['high'][i, j] or d[i, j] < 15.0 or d[i, j] > 60.0:
            continue
        k0 = _surface(w, i, j, Hi) - 1
        ln = int(rng.integers(5, 13))
        h = int(rng.integers(5, 15))
        along_i = rng.random() < 0.5
        for t in range(ln):
            a, b = (i + t, j) if along_i else (i, j + t)
            hh = h - abs(t - ln // 2) // 2
            for k in range(k0, min(k0 + hh, w.size[2] - 1)):
                w.ids[a, b, k] = BL.B['blue_stone']
        n += 1
        if n >= 16:
            break


def _boulders(w, rng, Hi, m, d):
    """Boulders of blue stone and siftslate scattered on the meadow, some mossy with healthy sculk."""
    X, Y, _ = w.size
    n = 0
    for _ in range(3000):
        i, j = int(rng.integers(4, X - 5)), int(rng.integers(4, Y - 5))
        if m['lake'][i, j] or m['high'][i, j] or m['stream'][i, j] or d[i, j] < 8.0:
            continue
        k0 = _surface(w, i, j, Hi) - 1
        rad = rng.uniform(1.2, 2.6)
        name = 'blue_stone' if rng.random() < 0.4 else 'siftslate'
        R = int(np.ceil(rad))
        for dx in range(-R, R + 1):
            for dy in range(-R, R + 1):
                for dz in range(0, R + 1):
                    if dx * dx + dy * dy + (dz * 1.3) ** 2 <= rad * rad + rng.random() * 0.8:
                        top = dz == R or (dx * dx + dy * dy + ((dz + 1) * 1.3) ** 2 > rad * rad)
                        w.ids[i + dx, j + dy, k0 + dz] = BL.B['healthy_sculk' if top and rng.random() < 0.5
                                                               else name]
        n += 1
        if n >= 45:
            break


def _fossil(w, fl, s0, s1):
    """A colossal fossil: its spine over the flight path, ribs arching down into the sand either side every five
    blocks (a few broken off), its skull half buried beyond."""
    bone = BL.B['bone_block']
    for s in np.arange(s0, s1, 0.5):
        P = fl.pos(s)
        T = fl.tangent(s)
        rh = np.array([T[1], -T[0], 0.0])
        rh /= max(np.linalg.norm(rh), 1e-6)
        spine = P + np.array([0.0, 0.0, 12.0])
        for dz in (0, 1):
            w.set(int(np.floor(spine[0])), int(np.floor(spine[1])), int(np.floor(spine[2])) + dz, bone)
        if abs((s - s0) % 5.0) < 0.25:
            k = int((s - s0) / 5.0)
            for side in (-1, 1):
                if (k * 7 + side) % 9 == 0:
                    continue                                      # a broken rib
                for a in np.linspace(0.0, np.pi * 0.66, 46):
                    q = spine + rh * side * 10.0 * np.sin(a) - np.array([0.0, 0.0, 1.0]) * (12.0 - 12.0 * np.cos(a))
                    q = q + rh * side * 1.8 * (a / np.pi)
                    for dz in (0, 1):
                        w.set(int(np.floor(q[0])), int(np.floor(q[1])), int(np.floor(q[2])) + dz, bone)
                q = spine + rh * side * 11.2
                for z in range(int(P[2]) - 10, int(spine[2]) - 10):
                    w.set(int(np.floor(q[0])), int(np.floor(q[1])), z, bone)
    # the skull, half buried off the path's outer side
    P = fl.pos(s1 + 6.0)
    T = fl.tangent(s1 + 6.0)
    rh = np.array([T[1], -T[0], 0.0])
    rh /= max(np.linalg.norm(rh), 1e-6)
    c = P - rh * 15.0
    cx, cy, cz = int(c[0]), int(c[1]), int(P[2]) - 5
    for dx in range(-5, 6):
        for dy in range(-6, 7):
            for dz in range(0, 9):
                inside = abs(dx) <= 4 and abs(dy) <= 5 and 1 <= dz <= 7
                shell = (abs(dx) <= 5 and abs(dy) <= 6 and dz <= 8) and not inside
                eye = dz in (5, 6) and abs(dy) in (2, 3) and dx == -5
                if shell and not eye:
                    w.set(cx + dx, cy + dy, cz + dz, bone)
    w.fossil = (s0, s1)


def _carve(w, fl, lateral=2.6, above=2.6, below=2.2, only=None):
    """Clear the flyer's corridor: nothing within a few blocks of the eye's path (the spire's top excepted, and the
    liquid left alone)."""
    s_a, s_b = only if only else (7.0, fl.length - 9.0)
    for s in np.arange(s_a, s_b, 0.5):
        P = fl.pos(s)
        if np.hypot(P[0] - FL.SPIRE[0], P[1] - FL.SPIRE[1]) < 7.0 and P[2] > FL.SPIRE[2] - 6:
            continue
        lo = np.floor(P - np.array([lateral, lateral, below])).astype(int)
        hi = np.floor(P + np.array([lateral, lateral, above])).astype(int)
        i0, j0, k0 = w.ix(*lo)
        i1, j1, k1 = w.ix(*hi)
        i0, j0, k0 = max(i0, 0), max(j0, 0), max(k0, 0)
        i1, j1, k1 = min(i1, w.size[0] - 1), min(j1, w.size[1] - 1), min(k1, w.size[2] - 1)
        if i1 < i0 or j1 < j0 or k1 < k0:
            continue
        gi, gj, gk = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), np.arange(k0, k1 + 1), indexing='ij')
        c = np.stack([gi + w.origin[0] + 0.5, gj + w.origin[1] + 0.5, gk + w.origin[2] + 0.5], -1) - P
        inside = (np.hypot(c[..., 0], c[..., 1]) <= lateral) & (c[..., 2] >= -below) & (c[..., 2] <= above)
        sub = w.ids[i0:i1 + 1, j0:j1 + 1, k0:k1 + 1]
        sub[inside & (sub != BL.B['ichor'])] = 0
