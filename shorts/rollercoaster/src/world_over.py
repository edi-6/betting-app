"""The Overworld: a badlands canyon at golden hour, built around the track.

A mesa plateau (grass and oak trees on top, at z 100) cut by a winding canyon that follows the ride from the drop to
the portal: terraced walls in the game's stripes of terracotta, a river at the bottom (water top at 19.9), red sand
on the floor and the ledges, dead bushes and cacti. Set pieces: the amphitheatre the first drop plunges into, a
natural arch with a stream pouring off it (the waterfall the cart goes through), and the ruined portal in the cliff
at the canyon's end. Taller mesas round the edges close off the view. The lift hill stands on wooden supports with
redstone torches.
"""
import numpy as np
from scipy import ndimage

import blocks as BL
import paths
import voxel as VX
from noise import fbm2d

ORIGIN = (-160, -240, 0)
SIZE = (320, 480, 150)
RIM = 100                 # plateau surface (top face of the ground)
FLOOR = 21                # canyon floor surface
WATER = 19                # water blocks up to this z (top face at 19.9)
BED = 16                  # river bed surface

# the badlands' horizontal stripes (bottom to top, repeating): name, thickness
BANDS = [('terracotta', 3), ('orange_terracotta', 2), ('yellow_terracotta', 1), ('terracotta', 2),
         ('white_terracotta', 1), ('orange_terracotta', 3), ('red_terracotta', 2), ('terracotta', 1),
         ('light_gray_terracotta', 2), ('brown_terracotta', 1), ('orange_terracotta', 2), ('yellow_terracotta', 2),
         ('terracotta', 3), ('red_terracotta', 1), ('white_terracotta', 2), ('orange_terracotta', 2),
         ('brown_terracotta', 2), ('terracotta', 2), ('light_gray_terracotta', 1), ('orange_terracotta', 3),
         ('red_terracotta', 2), ('yellow_terracotta', 1), ('terracotta', 4), ('white_terracotta', 1),
         ('orange_terracotta', 2), ('brown_terracotta', 1), ('red_terracotta', 3), ('terracotta', 2)]


def band_table():
    ids = []
    for name, n in BANDS:
        ids += [BL.B[name]] * n
    return np.array(ids, np.uint16)


def build(tr=None, verbose=True):
    tr = tr or paths.overworld()
    rng = np.random.default_rng(21)
    w = VX.World(size=SIZE, origin=ORIGIN)
    X, Y, Z = SIZE
    ox, oy, oz = ORIGIN
    xs = np.arange(X) + ox + 0.5
    ys = np.arange(Y) + oy + 0.5
    GX, GY = np.meshgrid(xs, ys, indexing='ij')

    # -- the canyon's centre line: the drop's footprint back to the head, then the ride's path to the portal ------
    s_bot = tr.marks['bottom']
    s_end = tr.marks['portal']
    ss = np.arange(s_bot - 40.0, s_end + 0.1, 1.0)
    cl = np.array([tr.pos(s)[:2] for s in ss])
    head = np.array([[0.0, y] for y in np.arange(-104.0, cl[0, 1], 1.0)])
    cl = np.concatenate([head, cl])
    cs = np.arange(len(cl), dtype=float)                   # a parameter along the line (1 per block)
    mask = np.ones((X, Y), bool)
    ci = np.clip(np.round(cl[:, 0] - ox - 0.5).astype(int), 0, X - 1)
    cj = np.clip(np.round(cl[:, 1] - oy - 0.5).astype(int), 0, Y - 1)
    mask[ci, cj] = False
    idx_map = -np.ones((X, Y), int)
    idx_map[ci, cj] = np.arange(len(cl))
    d, inds = ndimage.distance_transform_edt(mask, return_indices=True)
    near = idx_map[inds[0], inds[1]]                         # index of the nearest centre-line point
    u = cs[near]
    # the canyon's floor half-width and wall run vary along it
    n1 = fbm2d(u / 60.0, np.zeros_like(u) + 3.7, octaves=3, seed=5)
    half = 19.0 + 6.0 * n1
    run = 30.0 + 10.0 * fbm2d(u / 45.0, np.zeros_like(u) + 9.1, octaves=2, seed=6)
    # the portal cliff: the canyon ends at the portal
    beyond = GY > 151.5
    d = np.where(beyond, np.maximum(d, (GY - 151.5) * 3.0 + half), d)

    # -- heights ------------------------------------------------------------------------------------------------
    mesa = fbm2d(GX / 110.0, GY / 110.0, octaves=4, seed=11)
    plat = RIM + np.floor(np.clip(mesa * 46.0 - 6.0, 0, 40) / 8.0) * 8.0
    # keep the lift hill's end of the plateau flat, and wall the world in with tall mesas
    lift_zone = (np.abs(GX) < 40) & (GY < -120)
    plat = np.where(lift_zone, RIM, plat)
    edge = np.minimum.reduce([GX - ox, (ox + X) - GX, GY - oy, (oy + Y) - GY])
    rim_ring = np.clip(1.0 - edge / 40.0, 0, 1)
    plat = plat + np.floor(rim_ring * 34.0 / 6.0) * 6.0
    wall_n = 0.18 * fbm2d(GX / 14.0, GY / 14.0, octaves=3, seed=13)
    t = np.clip((d - half) / run + wall_n, 0.0, 1.0)
    lin = FLOOR + (plat - FLOOR) * t ** 0.85
    # terraces: steep risers with narrow ledges every 9 blocks
    step = 9.0
    base = np.floor(lin / step) * step
    frac = lin - base
    terr = base + np.clip((frac - 3.0) * 1.5, 0.0, step)
    H = np.where(d <= half, FLOOR, np.minimum(terr, plat))
    H = np.where(t >= 1.0, plat, H)
    # the river down the middle of the floor
    rw = 4.5 + 1.5 * fbm2d(u / 30.0, np.zeros_like(u) + 1.3, octaves=2, seed=7)
    river = (d < rw) & ~beyond & (GY > -110)
    H = np.where(river, BED, H)
    H = np.round(H).astype(int)
    w.H = H

    # -- fill ----------------------------------------------------------------------------------------------------
    bands = band_table()
    boff = (np.round(4.0 * fbm2d(GX / 70.0, GY / 70.0, octaves=2, seed=17))).astype(int)
    zs = np.arange(Z) + oz
    for k, z in enumerate(zs):
        solid = z < H
        if not solid.any():
            continue
        col = bands[(z + boff) % len(bands)]
        w.ids[:, :, k] = np.where(solid, col, w.ids[:, :, k])
    # surfaces: red sand on the floor and ledges, grass on the plateau, stone deep down
    top = H - 1 - oz
    ii, jj = np.nonzero(H > oz)
    kk = top[ii, jj]
    flat = np.zeros((X, Y), bool)
    gy, gx = np.gradient(H.astype(float))
    slope = np.hypot(gx, gy)
    flat = slope < 0.9
    surf = w.ids[ii, jj, kk]
    is_floor = (H[ii, jj] <= FLOOR + 1)
    is_plat = H[ii, jj] >= RIM - 1
    ledge = flat[ii, jj] & ~is_plat & ~is_floor
    surf = np.where(is_floor, BL.B['red_sand'], surf)
    surf = np.where(ledge, BL.B['red_sand'], surf)
    surf = np.where(is_plat & flat[ii, jj], BL.B['grass_block'], surf)
    w.ids[ii, jj, kk] = surf
    below = kk - 1 >= 0
    w.ids[ii[below & is_plat & flat[ii, jj]], jj[below & is_plat & flat[ii, jj]],
          kk[below & is_plat & flat[ii, jj]] - 1] = BL.B['dirt']
    w.ids[:, :, :4] = BL.B['stone']
    # the river's water and its sandy bed
    rv = river & (H <= BED)
    ri, rj = np.nonzero(rv)
    for z in range(BED, WATER + 1):
        w.ids[ri, rj, z - oz] = BL.B['water']
    w.ids[ri, rj, BED - 1 - oz] = BL.B['sand']
    # banks: water fills the floor's low spots next to the river so it doesn't show cliffs of sand
    if verbose:
        print('[over] terrain', flush=True)

    _arch_and_waterfall(w, tr, rng)
    _portal(w, tr)
    _plateau_life(w, tr, rng)
    _floor_life(w, tr, rng)
    clear_track(w, tr)
    _lift(w, tr)
    _supports(w, tr)
    if verbose:
        print('[over] features', flush=True)
    return w


def _arch_and_waterfall(w, tr, rng):
    """A natural arch across the canyon over the track, with a stream on top that pours off its upstream (south)
    face in a curtain the cart rides through."""
    s_a = tr.marks['arch'] - 10.0
    P = tr.pos(s_a)
    T, R, U = tr.frame(s_a)
    th = np.array([T[0], T[1], 0.0])
    th /= np.linalg.norm(th)
    rt = np.array([th[1], -th[0], 0.0])
    zb, zt = 37, 47                                        # underside (at the ends) and top of the arch
    for a in np.arange(-60.0, 60.1, 0.5):                  # across the canyon
        for b in np.arange(-5.0, 5.1, 0.5):                # along the track
            c = P[:2] + rt[:2] * a + th[:2] * b
            x, y = int(np.floor(c[0])), int(np.floor(c[1]))
            if not w.inside(x, y, 0):
                continue
            i, j, _ = w.ix(x, y, 0)
            ground = w.H[i, j]
            # the underside arcs up in the middle; the ends merge into the walls
            span = 1.0 - (a / 34.0) ** 2
            lo = zb + int(round(6.0 * max(span, 0.0))) + (1 if abs(b) > 4 else 0)
            if ground >= zt:
                continue
            for z in range(max(lo, ground), zt + (1 if abs(a) > 22 else 0)):
                band = band_table()[z % len(BANDS)]
                w.set(x, y, z, int(band))
            if abs(a) < 3.5:
                w.set(x, y, zt - 1, 'water')               # the stream crossing the arch
                w.set(x, y, zt, 'air')
    # the curtain: water pouring off the arch's upstream face down to the river, square across the track where the
    # cart meets it (the track curves here, so it is placed from the track itself, not the arch's axis)
    s_c = s_a - 6.0
    Pc = tr.pos(s_c)
    Tc, _, _ = tr.frame(s_c)
    thc = np.array([Tc[0], Tc[1], 0.0])
    thc /= np.linalg.norm(thc)
    rtc = np.array([thc[1], -thc[0], 0.0])
    for a in np.arange(-5.0, 5.01, 0.4):
        for bb in (0.0, 0.5):
            c = Pc[:2] + rtc[:2] * a + thc[:2] * bb
            x, y = int(np.floor(c[0])), int(np.floor(c[1]))
            for z in range(WATER, zt):
                if w.get(x, y, z) == 'air':
                    w.set(x, y, z, 'water')
    # and the lip it pours from: a ledge of the arch reaching back over the curtain
    for a in np.arange(-5.5, 5.51, 0.4):
        for bb in np.arange(0.0, 6.0, 0.4):
            c = Pc[:2] + rtc[:2] * a + thc[:2] * bb
            x, y = int(np.floor(c[0])), int(np.floor(c[1]))
            for z in range(zt - 3, zt):
                if w.get(x, y, z) == 'air':
                    w.set(x, y, z, 'orange_terracotta' if z != zt - 1 else 'water')
    w.waterfall = (Pc, rtc, thc)


def _portal(w, tr):
    """The ruined portal in the cliff at the canyon's end: an obsidian frame (a few blocks crying) 7 wide and 9
    tall in the x-z plane, corrupted netherrack and magma round it, a gold block or two, the tunnel behind."""
    s_p = tr.marks['portal']
    P = tr.pos(s_p)
    cx = int(np.floor(P[0]))
    y0 = int(np.floor(P[1])) + 1
    zb = int(np.floor(P[2])) - 3                            # the frame's bottom row
    rng = np.random.default_rng(4)
    for dx in range(-3, 4):
        for dz in range(0, 9):
            edge = abs(dx) == 3 or dz in (0, 8)
            x, z = cx + dx, zb + dz
            if edge:
                w.set(x, y0, z, 'crying_obsidian' if rng.random() < 0.22 else 'obsidian')
            else:
                w.set(x, y0, z, 'air')
    # clear in front of it, and a tunnel behind (never seen: the cut to the Nether comes first)
    for dx in range(-2, 3):
        for dz in range(1, 8):
            for dy in range(1, 14):
                w.set(cx + dx, y0 + dy, zb + dz, 'air')
    for dx in range(-9, 10):
        for dy in range(-6, 3):
            for dz in range(-2, 12):
                x, y, z = cx + dx, y0 + dy, zb + dz
                if abs(dx) <= 3 and 0 <= dz <= 8 and dy == 0:
                    continue
                nm = w.get(x, y, z)
                if nm == 'air' or nm == 'water':
                    continue
                r = np.hypot(dx / 9.0, dz / 11.0) + 0.25 * rng.random()
                if r < 0.8:
                    w.set(x, y, z, 'magma_block' if rng.random() < 0.18 else 'netherrack')
    for (dx, dz) in ((-5, 0), (5, 1), (-4, 9)):
        if w.get(cx + dx, y0 - 1, zb + dz) not in ('air', 'water'):
            w.set(cx + dx, y0 - 1, zb + dz, 'gold_block')
    w.portal = dict(center=np.array([cx + 0.5, y0 + 0.5, zb + 4.5]), width=5, height=7, normal=(0, 1, 0))


def _tree(w, x, y, z, rng):
    h = int(rng.integers(4, 7))
    for k in range(h):
        w.set(x, y, z + k, 'oak_log')
    top = z + h
    for dz in range(-2, 2):
        r = 2 if dz < 0 else 1
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if abs(dx) == r and abs(dy) == r and rng.random() < 0.6:
                    continue
                if w.get(x + dx, y + dy, top + dz) == 'air':
                    w.set(x + dx, y + dy, top + dz, 'oak_leaves')


def _plateau_life(w, tr, rng):
    X, Y, _ = w.size
    ox, oy, oz = w.origin
    cnt = 0
    for _ in range(2600):
        i, j = int(rng.integers(2, X - 2)), int(rng.integers(2, Y - 2))
        h = int(w.H[i, j])
        x, y = i + ox, j + oy
        if w.get(x, y, h - 1) != 'grass_block':
            continue
        if abs(x) < 6 and -200 < y < -110:
            continue                                          # keep the lift hill's line clear
        r = rng.random()
        if r < 0.06:
            _tree(w, x, y, h, rng)
            cnt += 1
        elif r < 0.5:
            w.set(x, y, h, 'tall_grass')
        elif r < 0.53:
            w.set(x, y, h, 'dead_bush')


def _floor_life(w, tr, rng):
    X, Y, _ = w.size
    ox, oy, oz = w.origin
    for _ in range(9000):
        i, j = int(rng.integers(2, X - 2)), int(rng.integers(2, Y - 2))
        h = int(w.H[i, j])
        x, y = i + ox, j + oy
        if w.get(x, y, h - 1) != 'red_sand' or w.get(x, y, h) != 'air':
            continue
        r = rng.random()
        if r < 0.10:
            for k in range(int(rng.integers(1, 4))):
                w.set(x, y, h + k, 'cactus')
        elif r < 0.45:
            w.set(x, y, h, 'dead_bush')


def _lift(w, tr):
    """Supports under the lift hill and the crest: spruce log posts every 4 blocks, with redstone torches."""
    s_top = tr.marks['drop'] + 2.0
    for s in np.arange(2.0, s_top, 4.0):
        P = tr.pos(s)
        T, R, U = tr.frame(s)
        for side in (-1, 1):
            q = P + R * (0.55 * side) - U * 1.0
            x, y = int(np.floor(q[0])), int(np.floor(q[1]))
            ztop = int(np.floor(q[2]))
            if not w.inside(x, y, 0):
                continue
            g = w.H[w.ix(x, y, 0)[0], w.ix(x, y, 0)[1]]
            for z in range(g, ztop + 1):
                w.set(x, y, z, 'spruce_log')
            if int(s) % 8 == 2 and ztop - g > 3:
                tx = x + (1 if side > 0 else -1)
                w.set(tx, y, ztop - 1, 'redstone_torch', BL.WALL | (1 if side > 0 else 3))


def _supports(w, tr):
    """Posts under the low river section: a spruce post every 6 blocks down to the ground or the river bed."""
    for s in np.arange(tr.marks['bottom'] + 6.0, tr.marks['portal'] - 4.0, 6.0):
        P = tr.pos(s)
        T, R, U = tr.frame(s)
        if abs(tr.bank[int(s / 0.1)]) > 30:
            continue                                      # hard-banked stretches hang from the rails' own strength
        x, y = int(np.floor(P[0])), int(np.floor(P[1]))
        ztop = int(np.floor(P[2] - 1.6))
        if not w.inside(x, y, 0):
            continue
        i, j, _ = w.ix(x, y, 0)
        g = int(w.H[i, j])
        for z in range(g, ztop + 1):
            if w.get(x, y, z) in ('air', 'water', 'dead_bush', 'tall_grass'):
                w.set(x, y, z, 'spruce_log')


def clear_track(w, tr, lateral=2.2, above=3.4, below=1.3, step=0.5, keep=('water',), s0=0.0, s1=None):
    """Carve the rider's envelope out of the world along the track (a box round the rails, in the track's frame)
    so nothing clips the cart. Blocks named in keep (the waterfall) stay."""
    s1 = tr.length if s1 is None else s1
    keep_ids = [BL.B[k] for k in keep]
    for s in np.arange(s0, s1, step):
        P = tr.pos(s)
        T, R, U = tr.frame(s)
        corners = []
        for a in (-lateral, lateral):
            for b in (-below, above):
                for c in (-0.6, 0.6):
                    corners.append(P + R * a + U * b + T * c)
        corners = np.array(corners)
        lo = np.floor(corners.min(0)).astype(int)
        hi = np.floor(corners.max(0)).astype(int)
        i0, j0, k0 = w.ix(*lo)
        i1, j1, k1 = w.ix(*hi)
        i0, j0, k0 = max(i0, 0), max(j0, 0), max(k0, 0)
        i1, j1, k1 = min(i1, w.size[0] - 1), min(j1, w.size[1] - 1), min(k1, w.size[2] - 1)
        if i1 < i0 or j1 < j0 or k1 < k0:
            continue
        gi, gj, gk = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), np.arange(k0, k1 + 1), indexing='ij')
        c = np.stack([gi + w.origin[0] + 0.5, gj + w.origin[1] + 0.5, gk + w.origin[2] + 0.5], -1) - P
        la, up, fw = c @ R, c @ U, c @ T
        inside = (np.abs(la) <= lateral) & (up >= -below) & (up <= above) & (np.abs(fw) <= 0.6)
        # the bed itself: the blocks right under the rails are the track's own (drawn as props)
        sub = w.ids[i0:i1 + 1, j0:j1 + 1, k0:k1 + 1]
        keepm = np.isin(sub, keep_ids)
        sub[inside & ~keepm] = 0
