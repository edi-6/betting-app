"""The Nether, built around the track: a vast lava sea in a cavern under a netherrack ceiling.

The cart comes out of a portal on a cliff ledge 60 blocks over the sea and plunges down it; skims the lava between
netherrack islands and pillars that climb to the ceiling, past lava falls pouring from it; runs up a hump on a
span of nether-brick pillars (the span the ghast's fireball brings down: its track and pillars are drawn as props
so they can fall); lands on the far side and climbs onto a fortress bridge that ends at an End portal, which it dives
into. Glowstone hangs from the ceiling, a crimson forest grows on an island, basalt columns and magma dot the sea.
"""
import numpy as np

import blocks as BL
import paths
import voxel as VX
from noise import fbm2d
from world_over import clear_track

ORIGIN = (-170, -200, 0)
SIZE = (300, 460, 128)
LAVA = 31             # lava blocks up to this z
SEA_FLOOR = 18
CEIL = 112


def build(tr=None, verbose=True):
    tr = tr or paths.nether()
    rng = np.random.default_rng(66)
    w = VX.World(size=SIZE, origin=ORIGIN)
    X, Y, Z = SIZE
    ox, oy, oz = ORIGIN
    xs = np.arange(X) + ox + 0.5
    ys = np.arange(Y) + oy + 0.5
    GX, GY = np.meshgrid(xs, ys, indexing='ij')

    # -- floor heights: the sea bed, islands, the arrival cliff, the cavern's walls -----------------------------
    n = fbm2d(GX / 46.0, GY / 46.0, octaves=4, seed=3)
    ridge = 1.0 - np.abs(fbm2d(GX / 70.0, GY / 70.0, octaves=3, seed=9))
    isl = np.clip((n - 0.18) * 2.4, 0, 1) * (22.0 + 40.0 * ridge ** 3)
    Hf = SEA_FLOOR + isl
    # the arrival cliff: the whole south end, its lip at y = -138 (bitten into by noise)
    lip = -138.0 + 6.0 * fbm2d(GX / 25.0, np.zeros_like(GX) + 2.2, octaves=2, seed=4)
    cliff = np.clip((lip - GY) / 3.0, 0, 1)
    Hf = np.maximum(Hf, cliff * 95.0 + (1 - cliff) * Hf)
    # the cavern's walls east and west (and north), climbing to the ceiling
    wall_d = np.minimum(GX - (ox + 18), (ox + X - 22) - GX)
    wall_d = np.minimum(wall_d, (oy + Y - 16) - GY)
    wall_n = 10.0 * fbm2d(GX / 30.0, GY / 30.0, octaves=3, seed=12)
    wall = np.clip(1.0 - (wall_d + wall_n) / 24.0, 0, 1)
    Hf = np.maximum(Hf, SEA_FLOOR + wall * (CEIL + 10 - SEA_FLOOR))
    # open lava under the jump: no island rises within ~24 blocks of the flight (the gap must read as a gap)
    dg = gap_distance(tr, GX, GY)
    sink = np.clip((dg - 24.0) / 12.0, 0, 1)
    Hf = np.where(Hf > SEA_FLOOR + 3, sink * Hf + (1 - sink) * np.minimum(Hf, SEA_FLOOR + 3), Hf)
    # ceiling (bottom surface), bumpy, lower where the walls are
    Hc = CEIL - 8.0 * fbm2d(GX / 24.0, GY / 24.0, octaves=3, seed=21) - 30.0 * wall
    Hf = np.round(Hf).astype(int)
    Hc = np.round(Hc).astype(int)
    w.H = Hf
    zs = np.arange(Z) + oz
    for k, z in enumerate(zs):
        solid = (z < Hf) | (z >= Hc)
        w.ids[:, :, k] = np.where(solid, BL.B['netherrack'], np.where(z <= LAVA, BL.B['lava'], 0))
    w.ids[:, :, :2] = BL.B['bedrock']
    w.ids[:, :, -2:] = BL.B['bedrock']
    # ores and variety in the rock: gold and quartz specks, blackstone patches, magma at the waterline
    rock = w.ids == BL.B['netherrack']
    r = rng.random(w.ids.shape, dtype=np.float32)
    w.ids[rock & (r < 0.004)] = BL.B['nether_gold_ore']
    w.ids[rock & (r > 0.995)] = BL.B['nether_quartz_ore']
    blk = fbm2d(GX / 18.0, GY / 18.0, octaves=2, seed=31) > 0.35
    for k in range(Z):
        z = k + oz
        sel = blk & (w.ids[:, :, k] == BL.B['netherrack']) & (z < Hf) & (z > Hf - 4)
        w.ids[:, :, k][sel] = BL.B['blackstone']
    shore = (Hf >= LAVA - 1) & (Hf <= LAVA + 2)
    si, sj = np.nonzero(shore)
    keep = rng.random(len(si)) < 0.35
    w.ids[si[keep], sj[keep], (Hf[si[keep], sj[keep]] - 1 - oz)] = BL.B['magma_block']
    if verbose:
        print('[nether] rock and lava', flush=True)

    _pillars(w, tr, rng)
    _glowstone(w, rng, Hc)
    _crimson_island(w, rng)
    _lava_falls(w, tr, rng)
    _basalt(w, tr, rng)
    clear_track(w, tr, keep=())
    _arrival_portal(w, tr)
    _bridge(w, tr, rng)
    _end_portal(w, tr)
    _supports(w, tr)
    w.gap_supports = _gap_supports(tr)
    w.ghasts = ghast_spots(tr)
    for g in w.ghasts:
        x0, y0, z0 = (int(np.floor(v)) for v in g)
        for dx in range(-7, 8):
            for dy in range(-7, 8):
                for dz in range(-7, 8):
                    if dx * dx + dy * dy + dz * dz <= 49 and w.get(x0 + dx, y0 + dy, z0 + dz) not in ('air', 'lava'):
                        w.set(x0 + dx, y0 + dy, z0 + dz, 'air')
    if verbose:
        print('[nether] features', flush=True)
    return w


def gap_distance(tr, GX, GY):
    """Horizontal distance from each (x, y) to the track from a little before the gap to a little past it."""
    g0, g1 = tr.gaps[0]
    pts = np.array([tr.pos(s)[:2] for s in np.arange(g0 - 36.0, g1 + 14.0, 2.0)])
    d = np.full(np.shape(GX), 1e9)
    for p in pts:
        d = np.minimum(d, np.hypot(GX - p[0], GY - p[1]))
    return d


def ghast_spots(tr):
    """Where the two ghasts float: one right of the track and above it ahead of the hump (it fires at the span as
    the cart comes up to it), one beyond the gap (it fires at the cart in the air)."""
    g0, g1 = tr.gaps[0]
    out = []
    for (s, lat, up) in ((g0 - 35.0, 11.0, 9.0), (g1 + 20.0, 15.0, 10.0)):
        P = tr.pos(s)
        T, R, U = tr.frame(s)
        rh = np.array([R[0], R[1], 0.0])
        rh /= np.linalg.norm(rh)
        out.append(P + rh * lat + np.array([0.0, 0.0, up]))
    return out


def _column(w, x, y, z0, z1, name):
    for z in range(int(z0), int(z1)):
        w.set(x, y, z, name)


def _pillars(w, tr, rng):
    """Great netherrack pillars from the sea to the ceiling, some right beside the track (never on it)."""
    ss = np.arange(20.0, tr.length, 2.0)
    pts = np.array([tr.pos(s)[:2] for s in ss])
    placed = 0
    for _ in range(400):
        c = np.array([rng.uniform(-150, 110), rng.uniform(-130, 240)])
        d = np.min(np.linalg.norm(pts - c, axis=1))
        r = rng.uniform(3.0, 7.5)
        if d < r + 5.0 or d > 90:
            continue
        top = CEIL + 5 if rng.random() < 0.55 else int(rng.uniform(55, 95))
        if gap_distance(tr, np.array([c[0]]), np.array([c[1]]))[0] < r + 22.0:
            continue
        for dx in range(-int(r) - 2, int(r) + 3):
            for dy in range(-int(r) - 2, int(r) + 3):
                for z in range(SEA_FLOOR, top):
                    rr = r * (1.0 + 0.25 * np.sin(z * 0.21 + dx) + (0.35 if z > top - 12 and top < CEIL else 0))
                    if dx * dx + dy * dy <= rr * rr:
                        w.set(int(c[0]) + dx, int(c[1]) + dy, z, 'netherrack' if (z + dx) % 7 else 'blackstone')
        placed += 1
        if placed >= 26:
            break


def _glowstone(w, rng, Hc):
    """Clusters of glowstone hanging from the ceiling."""
    X, Y, _ = w.size
    ox, oy, oz = w.origin
    for _ in range(170):
        i, j = int(rng.integers(4, X - 4)), int(rng.integers(4, Y - 4))
        top = int(Hc[i, j])
        if top >= w.size[2] + oz - 3:
            continue
        x, y = i + ox, j + oy
        for k in range(int(rng.integers(6, 16))):
            dx, dy = int(rng.integers(-2, 3)), int(rng.integers(-2, 3))
            dz = int(rng.integers(1, 6))
            w.set(x + dx, y + dy, top - dz, 'glowstone')


def _crimson_island(w, rng):
    """An island of crimson forest east of the drop: red nylium, huge fungi with wart-block caps and shroomlights."""
    cx, cy = 42, -92
    for dx in range(-22, 23):
        for dy in range(-18, 19):
            d = (dx / 22.0) ** 2 + (dy / 18.0) ** 2
            if d > 1:
                continue
            top = LAVA + 3 + int(6 * (1 - d))
            _column(w, cx + dx, cy + dy, SEA_FLOOR, top - 1, 'netherrack')
            w.set(cx + dx, cy + dy, top - 1, 'crimson_nylium')
    for _ in range(9):
        x = cx + int(rng.integers(-16, 17))
        y = cy + int(rng.integers(-12, 13))
        z = LAVA + 3
        while w.get(x, y, z) != 'air' and z < 60:
            z += 1
        h = int(rng.integers(6, 11))
        _column(w, x, y, z, z + h, 'crimson_stem')
        for dz in range(-2, 2):
            rr = 3 if dz < 0 else 2
            for dx in range(-rr, rr + 1):
                for dy in range(-rr, rr + 1):
                    if abs(dx) == rr and abs(dy) == rr:
                        continue
                    p = (x + dx, y + dy, z + h + dz)
                    if w.get(*p) == 'air':
                        w.set(*p, 'shroomlight' if rng.random() < 0.08 else 'nether_wart_block')


def _lava_falls(w, tr, rng):
    """Lava falls from the ceiling: along the 'falls' stretch on both sides of the track, and a few further off."""
    spots = []
    for (s_off, side, lat) in ((-26, 1, 13.0), (-6, -1, 11.0), (14, 1, 16.0), (30, -1, 14.0)):
        s = tr.marks['falls'] + s_off
        P = tr.pos(s)
        T, R, U = tr.frame(s)
        rh = np.array([R[0], R[1], 0.0])
        rh /= np.linalg.norm(rh)
        c = P[:2] + rh[:2] * side * lat
        spots.append((c, 2))
    for c in ((60.0, -30.0), (-90.0, 60.0), (30.0, 120.0), (-70.0, -60.0)):
        spots.append((np.array(c), 3))
    falls = []
    for c, wd in spots:
        x0, y0 = int(c[0]), int(c[1])
        for dx in range(wd):
            for dy in range(wd):
                for z in range(LAVA, CEIL + 12):
                    if w.get(x0 + dx, y0 + dy, z) == 'air':
                        w.set(x0 + dx, y0 + dy, z, 'lava')
        falls.append((x0 + wd / 2, y0 + wd / 2, wd))
    w.lava_falls = falls


def _basalt(w, tr, rng):
    ss = np.arange(20.0, tr.length, 2.0)
    pts = np.array([tr.pos(s)[:2] for s in ss])
    n = 0
    for _ in range(300):
        c = np.array([rng.uniform(-140, 100), rng.uniform(-120, 230)])
        d = np.min(np.linalg.norm(pts - c, axis=1))
        if d < 6 or d > 60:
            continue
        h = int(rng.uniform(4, 16))
        x, y = int(c[0]), int(c[1])
        for dx in range(2):
            for dy in range(2):
                _column(w, x + dx, y + dy, SEA_FLOOR, LAVA + 1 + h - (dx + dy), 'basalt')
        n += 1
        if n > 40:
            break


def _arrival_portal(w, tr):
    """The portal the cart comes out of, on the cliff's ledge, facing north (x-z plane at the track's start)."""
    P = tr.pos(0.4)
    cx = int(np.floor(P[0]))
    y0 = int(np.floor(P[1]))
    zb = int(np.floor(P[2])) - 3
    for dx in range(-3, 4):
        for dz in range(0, 9):
            if abs(dx) == 3 or dz in (0, 8):
                w.set(cx + dx, y0, zb + dz, 'obsidian')
            else:
                w.set(cx + dx, y0, zb + dz, 'air')
    w.portal = dict(center=np.array([cx + 0.5, y0 + 0.5, zb + 4.5]), width=5, height=7, normal=(0, 1, 0))


def _bridge(w, tr, rng):
    """The fortress bridge from past the landing to the End portal: a nether-brick deck under the track, fences
    along its edges, arches down to the lava."""
    s0 = tr.marks['land'] + 12.0
    s1 = tr.marks['portal'] - 3.0
    for s in np.arange(s0, s1, 0.5):
        P = tr.pos(s)
        T, R, U = tr.frame(s)
        rh = np.array([R[0], R[1], 0.0])
        rh /= np.linalg.norm(rh)
        zd = int(np.floor(P[2] - 1.02)) - 1
        for a in np.arange(-3.0, 3.01, 0.5):
            x, y = int(np.floor(P[0] + rh[0] * a)), int(np.floor(P[1] + rh[1] * a))
            w.set(x, y, zd, 'nether_bricks')
            if abs(a) >= 2.9:
                w.set(x, y, zd + 1, 'nether_brick_fence')
                w.set(x, y, zd + 2, 'nether_brick_fence' if int(s) % 6 else 'air')
        if int(s * 2) % 16 == 0:
            for a in (-2.5, 2.5):
                x, y = int(np.floor(P[0] + rh[0] * a)), int(np.floor(P[1] + rh[1] * a))
                for z in range(SEA_FLOOR, zd):
                    w.set(x, y, z, 'nether_bricks')
    # netherrack posts along the bridge with the game's everlasting fire on top (the flames are props)
    fires = []
    for s in np.arange(s0 + 4.0, s1 - 2.0, 9.0):
        P = tr.pos(s)
        T, R, U = tr.frame(s)
        rh = np.array([R[0], R[1], 0.0])
        rh /= np.linalg.norm(rh)
        zd = int(np.floor(P[2] - 1.02)) - 1
        for side in (-1, 1):
            x, y = int(np.floor(P[0] + rh[0] * 3.6 * side)), int(np.floor(P[1] + rh[1] * 3.6 * side))
            for z in range(zd, zd + 3):
                w.set(x, y, z, 'netherrack')
            fires.append((x + 0.5, y + 0.5, float(zd + 3)))
    w.bridge = (s0, s1)
    w.fires = fires


def _end_portal(w, tr):
    """The End portal at the bridge's end: a 5x5 ring of frame blocks (every eye in place) round a 3x3 opening
    whose surface is drawn as a prop, on a nether-brick platform. The track dives into it."""
    s_p = tr.marks['portal']
    zt = 44                                               # the frame's top face
    ss = np.arange(s_p, s_p + 14.0, 0.05)
    zz = np.array([tr.pos(s)[2] for s in ss])
    i = int(np.argmax(zz < zt - 0.4))
    c = tr.pos(ss[i])
    cx, cy = int(np.floor(c[0])), int(np.floor(c[1]))
    for dx in range(-6, 7):
        for dy in range(-6, 7):
            for z in range(SEA_FLOOR, zt - 1):
                if abs(dx) <= 2 and abs(dy) <= 2 and z >= zt - 4:
                    continue
                w.set(cx + dx, cy + dy, z, 'nether_bricks')
            if max(abs(dx), abs(dy)) == 6 and (dx + dy) % 2 == 0:
                w.set(cx + dx, cy + dy, zt - 1, 'nether_brick_fence')
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            ring = max(abs(dx), abs(dy)) == 2 and not (abs(dx) == 2 and abs(dy) == 2)
            if ring:
                w.set(cx + dx, cy + dy, zt - 1, 'end_portal_frame_eye')
            elif max(abs(dx), abs(dy)) <= 1:
                for z in range(zt - 6, zt):
                    w.set(cx + dx, cy + dy, z, 'air')
    w.end_portal = dict(center=np.array([cx + 0.5, cy + 0.5, zt - 0.2]), size=3)


def _supports(w, tr):
    """Nether-brick pillars under the track where it runs over the sea (not under the span the fireball breaks:
    those are props)."""
    g0, g1 = tr.gaps[0]
    for s in np.arange(tr.marks['lava'], tr.marks['portal'] - 4.0, 7.0):
        if g0 - 14 < s < g1 + 6:
            continue
        P = tr.pos(s)
        if abs(tr.bank[int(s / 0.1)]) > 30:
            continue
        x, y = int(np.floor(P[0])), int(np.floor(P[1]))
        for z in range(SEA_FLOOR, int(np.floor(P[2] - 1.6)) + 1):
            if w.get(x, y, z) in ('air', 'lava'):
                w.set(x, y, z, 'nether_bricks')


def _gap_supports(tr):
    """The pillars under the span the fireball breaks: [(x, y, z_top)] every 7 blocks from before the take-off to
    past the landing (drawn as props so the explosion can bring them down)."""
    g0, g1 = tr.gaps[0]
    out = []
    for s in np.arange(g0 - 10.0, g1 + 4.0, 7.0):
        P = tr.pos(s)
        out.append((float(np.floor(P[0])), float(np.floor(P[1])), float(np.floor(P[2] - 1.6)), float(s)))
    return out
