"""The coast: a plains village on a bay. The sea is to the north (+y), sea level at z = 0. A sandy beach, the
village's flat meadow at z = 2, grassy hills rising behind it to the south, rocky headlands with cliffs on both sides
of the bay, and snowy mountains on the horizon (for the widest shots, a coarse mesh out to kilometres).

Heightmap H[iy, ix]: the top of the column at x = ix + X0, y = iy + Y0 (blocks; the top face is at z = H).
Meshes are chunked, indexed triangle lists (vertex = pos3, normal3, uv2, layer1, tint3), built with numpy.
mesh_blocks() meshes any set of loose blocks (the village, which the water knocks apart frame by frame).
"""
import numpy as np

import blocks as BL
from noise import fbm2d, perlin2d

L = BL.L
X0, X1 = -320, 320
Y0, Y1 = -420, 300
NX, NY = X1 - X0, Y1 - Y0
CHUNK = 32
SEA = 0.0
PLAIN = 2                          # the village's meadow: the top of the grass at z = 2


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def coast_y(x):
    """Where the land meets the sea: a straight beach in front of the village, the bay's headlands further out."""
    ax = np.abs(x)
    return 4.0 + 46.0 * smoothstep(90.0, 175.0, ax) + 3.0 * np.sin(x / 23.0) * smoothstep(40.0, 90.0, ax)


def heights(X, Y, seed=5):
    """Float terrain height at points (X, Y)."""
    yc = coast_y(X)
    d = Y - yc                                       # > 0: out in the sea
    ax = np.abs(X)
    # the sea floor shelves down, deeper further out
    sea = -1.0 - 0.10 * np.maximum(d, 0) - 0.00012 * np.maximum(d, 0) ** 2
    sea += 1.6 * fbm2d(X / 37.0, Y / 37.0, octaves=3, seed=seed + 9)
    sea = np.maximum(sea, -48.0)
    # far out the sea floor rolls in long dunes (only ever seen once the sea is gone)
    sea += smoothstep(300.0, 700.0, d) * (7.0 * fbm2d(X / 350.0, Y / 350.0, octaves=3, seed=seed + 11) +
                                         2.0 * fbm2d(X / 60.0, Y / 60.0, octaves=2, seed=seed + 12))
    # the beach: sand at 1 above the water, a wet strip at 0 right at the edge
    beach = np.where(d > -3.0, 0.0, 1.0)
    # the meadow and the hills behind it
    back = np.maximum(0.0, -112.0 - Y)
    hills = PLAIN + 0.16 * back + 6.0 * smoothstep(0, 60, back) * (0.6 + 0.4 * fbm2d(X / 55.0, Y / 55.0, octaves=3,
                                                                                        seed=seed))
    hills += 1.5 * smoothstep(0, 40, back) * fbm2d(X / 15.0, Y / 15.0, octaves=2, seed=seed + 1)
    # the meadow outside the village gets a gentle swell
    swell = smoothstep(70.0, 110.0, ax) * 1.8 * np.maximum(fbm2d(X / 30.0, Y / 30.0, octaves=2, seed=seed + 2), 0)
    land = np.where(d < -9.0, hills + swell, np.where(d < -4.0, 1.0 + (hills - 1.0) * smoothstep(-4.0, -9.0, d), beach))
    land = np.maximum(land, np.where(d < -9.0, PLAIN, 0.0))
    # the headlands: rocky cliffs above the sea on both sides of the bay
    head = smoothstep(105.0, 150.0, ax)
    cliff = 14.0 + 10.0 * fbm2d(X / 40.0, Y / 40.0, octaves=3, seed=seed + 3) + 0.04 * (ax - 150.0)
    inland = smoothstep(4.0, -8.0, d)                # how far behind the cliff edge
    land = np.maximum(land, head * (cliff * inland))
    # mountains far behind
    far = smoothstep(220.0, 420.0, -Y) + smoothstep(260.0, 520.0, ax) * smoothstep(-200.0, 0.0, -Y)
    ridge = 1.0 - np.abs(perlin2d(X / 140.0 + 3.1, Y / 140.0 + 1.7, seed=seed + 4))
    mount = far * (30.0 + 70.0 * ridge ** 2 + 18.0 * fbm2d(X / 45.0, Y / 45.0, octaves=3, seed=seed + 5))
    land = np.maximum(land, mount)
    return np.where(d > 0.0, sea, land)


def make_heightmap(seed=5):
    xs = np.arange(X0, X1) + 0.5
    ys = np.arange(Y0, Y1) + 0.5
    X, Y = np.meshgrid(xs, ys)
    h = heights(X, Y, seed)
    H = np.floor(h + 0.5).astype(np.int32)
    # the village area: perfectly flat meadow, the beach flat sand
    yc = coast_y(X)
    vil = (np.abs(X) < 92) & (Y < yc - 9) & (Y > -112)
    H[vil] = PLAIN
    return H


# ----------------------------------------------------------------------------------------------------------------
# meshing

def _quads(pos, nrm, uv, layer, tint):
    """pos (N,4,3), nrm (3,), uv (N,4,2), layer (N,), tint (N,3) -> vertices (4N,12), indices (6N,)."""
    n = len(pos)
    v = np.zeros((n, 4, 12), np.float32)
    v[:, :, 0:3] = pos
    v[:, :, 3:6] = nrm
    v[:, :, 6:8] = uv
    v[:, :, 8] = layer[:, None]
    v[:, :, 9:12] = tint[:, None, :]
    b = np.arange(n, dtype=np.uint32)[:, None] * 4
    idx = (b + np.array([0, 1, 2, 0, 2, 3], np.uint32)).reshape(-1)
    return v.reshape(-1, 12), idx


def face_quads(face, x0, y0, z0, x1, y1, z1, layer, tint=None):
    """Vectorised box_face: arrays of boxes -> quads for one face direction (same corners and uv as world.box_face
    in the earlier videos, so textures line up with the terrain)."""
    x0, y0, z0, x1, y1, z1 = (np.asarray(a, np.float32) for a in (x0, y0, z0, x1, y1, z1))
    n = x0.shape[0]
    layer = np.broadcast_to(np.asarray(layer, np.float32), (n,))
    tint = np.ones((n, 3), np.float32) if tint is None else np.broadcast_to(np.asarray(tint, np.float32), (n, 3))
    S = lambda *c: np.stack(c, -1)
    if face == 'pz':
        p = np.stack([S(x0, y0, z1), S(x1, y0, z1), S(x1, y1, z1), S(x0, y1, z1)], 1)
        uv = np.stack([S(x0, -y0), S(x1, -y0), S(x1, -y1), S(x0, -y1)], 1)
        nr = (0, 0, 1)
    elif face == 'nz':
        p = np.stack([S(x0, y1, z0), S(x1, y1, z0), S(x1, y0, z0), S(x0, y0, z0)], 1)
        uv = np.stack([S(x0, y1), S(x1, y1), S(x1, y0), S(x0, y0)], 1)
        nr = (0, 0, -1)
    elif face == 'px':
        p = np.stack([S(x1, y0, z0), S(x1, y1, z0), S(x1, y1, z1), S(x1, y0, z1)], 1)
        uv = np.stack([S(y0, -z0), S(y1, -z0), S(y1, -z1), S(y0, -z1)], 1)
        nr = (1, 0, 0)
    elif face == 'nx':
        p = np.stack([S(x0, y1, z0), S(x0, y0, z0), S(x0, y0, z1), S(x0, y1, z1)], 1)
        uv = np.stack([S(-y1, -z0), S(-y0, -z0), S(-y0, -z1), S(-y1, -z1)], 1)
        nr = (-1, 0, 0)
    elif face == 'py':
        p = np.stack([S(x1, y1, z0), S(x0, y1, z0), S(x0, y1, z1), S(x1, y1, z1)], 1)
        uv = np.stack([S(-x1, -z0), S(-x0, -z0), S(-x0, -z1), S(-x1, -z1)], 1)
        nr = (0, 1, 0)
    else:
        p = np.stack([S(x0, y0, z0), S(x1, y0, z0), S(x1, y0, z1), S(x0, y0, z1)], 1)
        uv = np.stack([S(x0, -z0), S(x1, -z0), S(x1, -z1), S(x0, -z1)], 1)
        nr = (0, -1, 0)
    return _quads(p, np.array(nr, np.float32), uv, layer, tint)


DIRS = {'px': (1, 0, 0), 'nx': (-1, 0, 0), 'py': (0, 1, 0), 'ny': (0, -1, 0), 'pz': (0, 0, 1), 'nz': (0, 0, -1)}


def mesh_blocks(K, origin, ground=None, tint=None):
    """Mesh a dense block grid K[z, y, x] (kind ids, 0 = air) whose cell (0, 0, 0) has its lower corner at origin.
    ground: optional bool grid of the same shape counted as solid (the terrain) so faces against it are hidden.
    tint: optional (nz, ny, nx, 3) per block tint (leaves)."""
    solid = K > 0
    occ = solid if ground is None else (solid | ground)
    ox, oy, oz = origin
    verts, inds = [], []
    nv = 0
    for face, (dx, dy, dz) in DIRS.items():
        nb = np.zeros_like(occ)
        # neighbour in direction (dx, dy, dz)
        src = [slice(None)] * 3
        dst = [slice(None)] * 3
        for ax, d in ((2, dx), (1, dy), (0, dz)):
            if d == 1:
                dst[ax], src[ax] = slice(0, -1), slice(1, None)
            elif d == -1:
                dst[ax], src[ax] = slice(1, None), slice(0, -1)
        nb[tuple(dst)] = occ[tuple(src)]
        show = solid & ~nb
        zz, yy, xx = np.nonzero(show)
        if len(zz) == 0:
            continue
        kinds = K[zz, yy, xx]
        col = 0 if face == 'pz' else (2 if face == 'nz' else 1)
        layer = BL.FACE_LAYERS[kinds, col]
        t = None if tint is None else tint[zz, yy, xx]
        x0, y0, z0 = xx + ox, yy + oy, zz + oz
        v, i = face_quads(face, x0, y0, z0, x0 + 1, y0 + 1, z0 + 1, layer, t)
        verts.append(v)
        inds.append(i + nv)
        nv += len(v)
    if not verts:
        return np.zeros((0, 12), np.float32), np.zeros(0, np.uint32)
    return np.concatenate(verts), np.concatenate(inds)


def terrain_layers(H, paint=None):
    """Top layer per column, and a code for what its sides are made of (0 grass/dirt/stone, 1 sand, 2 rock).
    paint: {(x, y): layer} overrides for the top (the village's streets, floors and fields)."""
    gy, gx = np.gradient(H.astype(float))
    steep = np.hypot(gx, gy) > 1.6
    xs = np.arange(X0, X1) + 0.5
    ys = np.arange(Y0, Y1) + 0.5
    X, Y = np.meshgrid(xs, ys)
    d = Y - coast_y(X)
    sandy = (H <= 1) & (d > -12)
    top = np.full(H.shape, L['grass_top'], np.int32)
    top[sandy] = L['sand']
    under = H < 0
    deep = H < -7
    n = fbm2d(X / 9.0, Y / 9.0, octaves=2, seed=41)
    top[under] = L['sand']
    top[deep & (n > 0.05)] = L['gravel']
    top[deep & (n < -0.25)] = L['clay']
    rock = steep & ~sandy & ~under
    top[rock] = np.where(n[rock] > 0.1, L['andesite'], L['stone'])
    top[H >= 78] = L['snow'] if 'snow' in L else L['stone']
    kind = np.zeros(H.shape, np.int32)
    kind[sandy | under] = 1
    kind[rock] = 2
    for (x, y), lay in (paint or {}).items():
        top[y - Y0, x - X0] = lay
    return top, kind


def mesh_terrain(H, paint=None):
    """Chunked terrain mesh: one quad per column top (merged along x where equal), and the exposed sides split into
    grass/dirt/stone (or sand/sandstone) bands."""
    top, kind = terrain_layers(H, paint)
    chunks = {}

    def add(key, v, i):
        chunks.setdefault(key, []).append((v, i))

    ny, nx = H.shape
    # tops: merge runs along x within a chunk row
    for iy in range(ny):
        h, lay = H[iy], top[iy]
        brk = np.ones(nx, bool)
        brk[1:] = (h[1:] != h[:-1]) | (lay[1:] != lay[:-1]) | ((np.arange(1, nx) % CHUNK) == 0)
        starts = np.nonzero(brk)[0]
        ends = np.append(starts[1:], nx)
        y = iy + Y0
        v, i = face_quads('pz', starts + X0, np.full(len(starts), y), h[starts] - 1, ends + X0,
                          np.full(len(starts), y + 1), h[starts], lay[starts])
        # split by chunk
        cx = (starts + X0) // CHUNK
        cy = y // CHUNK
        for c in np.unique(cx):
            sel = np.nonzero(cx == c)[0]
            vv = v.reshape(-1, 4, 12)[sel].reshape(-1, 12)
            ii = (np.arange(len(sel), dtype=np.uint32)[:, None] * 4 + np.array([0, 1, 2, 0, 2, 3], np.uint32)).ravel()
            add((int(c), int(cy)), vv, ii)
    # sides
    Hp = np.pad(H, 1, mode='edge')
    for face, (dx, dy, _) in DIRS.items():
        if face in ('pz', 'nz'):
            continue
        hn = Hp[1 + dy:1 + dy + ny, 1 + dx:1 + dx + nx]
        diff = H - hn
        iy, ix = np.nonzero(diff > 0)
        if len(iy) == 0:
            continue
        h, hlow, k = H[iy, ix], hn[iy, ix], kind[iy, ix]
        x0, y0 = ix + X0, iy + Y0
        bands = []
        # band 1: the top block
        z1 = h
        z0 = np.maximum(h - 1, hlow)
        lay = np.where(k == 1, L['sand'], np.where(k == 2, L['stone'], L['grass_side']))
        lay = np.where(H[iy, ix] >= 78, L['stone'], lay)
        bands.append((z0, z1, lay))
        # band 2: dirt / sandstone, three blocks
        z1b = z0
        z0b = np.maximum(h - 4, hlow)
        lay2 = np.where(k == 1, L['sandstone'], np.where(k == 2, L['stone'], L['dirt']))
        bands.append((z0b, z1b, lay2))
        # band 3: stone below
        z1c = z0b
        z0c = hlow
        n = (x0 * 7 + y0 * 13) % 5
        lay3 = np.where(n < 1, L['andesite'], L['stone'])
        bands.append((z0c, z1c, lay3))
        for (za, zb, lay) in bands:
            sel = zb > za
            if not np.any(sel):
                continue
            xa, ya = x0[sel], y0[sel]
            v, i = face_quads(face, xa, ya, za[sel], xa + 1, ya + 1, zb[sel], lay[sel])
            keys_x = xa // CHUNK
            keys_y = ya // CHUNK
            kk = keys_x.astype(np.int64) * 100000 + keys_y.astype(np.int64)
            order = np.argsort(kk, kind='stable')
            kk_s = kk[order]
            cuts = np.nonzero(np.diff(kk_s))[0] + 1
            for grp in np.split(order, cuts):
                vv = v.reshape(-1, 4, 12)[grp].reshape(-1, 12)
                ii = (np.arange(len(grp), dtype=np.uint32)[:, None] * 4 + np.array([0, 1, 2, 0, 2, 3],
                                                                                   np.uint32)).ravel()
                add((int(xa[grp[0]] // CHUNK), int(ya[grp[0]] // CHUNK)), vv, ii)
    return chunks


def assemble(chunks):
    verts, idx, ranges = [], [], []
    base_v = base_i = 0
    for key in sorted(chunks):
        parts = chunks[key]
        v = np.concatenate([p[0] for p in parts])
        offs = np.cumsum([0] + [len(p[0]) for p in parts[:-1]])
        i = np.concatenate([p[1] + o for p, o in zip(parts, offs)])
        if len(i) == 0:
            continue
        verts.append(v)
        idx.append(i + base_v)
        ranges.append((base_i, len(i), v[:, :3].min(0), v[:, :3].max(0)))
        base_v += len(v)
        base_i += len(i)
    return np.concatenate(verts), np.concatenate(idx).astype(np.uint32), ranges


# ----------------------------------------------------------------------------------------------------------------
# trees (the ones outside the village; the village's own are loose blocks the water can take)

LEAF_TINTS = [(1.00, 1.00, 1.00), (0.88, 1.02, 0.82), (1.08, 1.06, 0.88), (0.84, 0.96, 0.80)]


def tree_blocks(rng, birch=False):
    trunk = int(rng.integers(5, 8) if birch else rng.integers(4, 7))
    blocks = [(0, 0, z, 'log') for z in range(trunk)]
    cr = rng.uniform(1.8, 2.6) if birch else rng.uniform(2.2, 3.3)
    cz = trunk - 0.3 + rng.uniform(-0.3, 0.5)
    rz = cr * rng.uniform(0.8, 1.0)
    R = int(np.ceil(cr)) + 1
    for x in range(-R, R + 1):
        for y in range(-R, R + 1):
            for z in range(int(cz - rz) - 1, int(cz + rz) + 2):
                if x == 0 and y == 0 and z < trunk:
                    continue
                d = (x / cr) ** 2 + (y / cr) ** 2 + ((z - cz) / rz) ** 2
                if d <= 1.0 and not (d > 0.72 and rng.random() < 0.28):
                    blocks.append((x, y, z, 'leaf'))
    return blocks


def tree_spots(H, rng, avoid, spacing=7.0):
    """Places for trees: grass, level ground, not in the village (avoid(x, y) -> True to keep clear)."""
    top, kind = terrain_layers(H)
    pts = []
    cand = rng.uniform((X0 + 4, Y0 + 4), (X1 - 4, Y1 - 4), size=(60000, 2))
    grid = {}
    for x, y in cand:
        ix, iy = int(np.floor(x)) - X0, int(np.floor(y)) - Y0
        h = H[iy, ix]
        if h < 2 or h > 70 or top[iy, ix] != L['grass_top'] or avoid(x, y):
            continue
        if np.ptp(H[max(0, iy - 2):iy + 3, max(0, ix - 2):ix + 3]) > 1:
            continue
        gx, gy = int(x // spacing), int(y // spacing)
        ok = True
        for a in range(gx - 1, gx + 2):
            for b in range(gy - 1, gy + 2):
                for (px, py) in grid.get((a, b), ()):
                    if (px - x) ** 2 + (py - y) ** 2 < spacing * spacing:
                        ok = False
        if not ok:
            continue
        # woods on the hills, a few on the meadow
        dens = 0.25 + 0.75 * smoothstep(-110.0, -160.0, y)
        if rng.random() > dens:
            continue
        grid.setdefault((gx, gy), []).append((x, y))
        pts.append((int(np.floor(x)), int(np.floor(y)), int(h)))
    return pts


def mesh_trees(spots, rng):
    chunks = {}
    for (bx, by, h) in spots:
        birch = rng.random() < 0.25
        blocks = tree_blocks(rng, birch)
        tint = LEAF_TINTS[rng.integers(len(LEAF_TINTS))]
        xs = np.array([b[0] for b in blocks])
        ys = np.array([b[1] for b in blocks])
        zs = np.array([b[2] for b in blocks])
        lo = np.array([xs.min(), ys.min(), zs.min()])
        K = np.zeros((zs.max() - lo[2] + 1, ys.max() - lo[1] + 1, xs.max() - lo[0] + 1), np.uint8)
        T = np.ones(K.shape + (3,), np.float32)
        for (x, y, z, kind) in blocks:
            K[z - lo[2], y - lo[1], x - lo[0]] = BL.KIND_ID['log' if kind == 'log' else 'leaves']
            if kind != 'log':
                T[z - lo[2], y - lo[1], x - lo[0]] = tint
        v, i = mesh_blocks(K, (bx + lo[0], by + lo[1], h + lo[2]), tint=T)
        if birch:
            v[v[:, 8] == L['log'], 8] = L['birch_log']
            v[v[:, 8] == L['leaves'], 8] = L['birch_leaves']
        key = (bx // CHUNK, by // CHUNK)
        chunks.setdefault(key, []).append((v, i))
    return chunks


# ----------------------------------------------------------------------------------------------------------------
# the far land: kilometres of hills and mountains around the detailed area, coarse and smooth

FAR = 6000.0
FAR_STEP = 24.0
SEA_FAR = 7200.0                 # how far out to sea the far mesh's sea floor reaches


def far_heights(X, Y):
    h = heights(X, Y)
    ax = np.abs(X)
    # out beyond the detailed map the land rolls on into big mountains; the sea stays sea
    big = smoothstep(500.0, 1600.0, np.maximum(-Y, ax * 0.8))
    ridge = 1.0 - np.abs(perlin2d(X / 700.0 + 7.3, Y / 700.0 + 2.9, seed=88))
    mount = big * (60.0 + 260.0 * ridge ** 2 + 60.0 * fbm2d(X / 260.0, Y / 260.0, octaves=4, seed=89))
    land = h > -0.5
    return np.where(land, np.maximum(h, mount), h)


def mesh_far():
    """A coarse smooth-shaded mesh from -FAR to FAR, with a hole where the detailed map is."""
    n = int(2 * FAR / FAR_STEP) + 1
    xs = np.linspace(-FAR, FAR, n)
    ys = np.linspace(-FAR, SEA_FAR, int((FAR + SEA_FAR) / FAR_STEP) + 1)
    X, Y = np.meshgrid(xs, ys)
    Z = far_heights(X, Y)
    inside = (X > X0 + 8) & (X < X1 - 8) & (Y > Y0 + 8) & (Y < Y1 - 8)
    Z = np.where(inside, Z - 3.0, Z)                   # tucked under the detailed map
    gy, gx = np.gradient(Z, ys[1] - ys[0], xs[1] - xs[0])
    nrm = np.stack([-gx, -gy, np.ones_like(Z)], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    slope = 1 - nrm[..., 2]
    layer = np.full(Z.shape, L['grass_top'], np.float32)
    layer[slope > 0.25] = L['stone']
    layer[Z > 170] = L['snow'] if 'snow' in L else L['stone']
    layer[Z < 0.5] = L['sand']
    pn = fbm2d(X / 180.0, Y / 180.0, octaves=3, seed=91)
    deep = Z < -12.0
    layer[deep & (pn > 0.28)] = L['gravel']
    layer[deep & (pn < -0.32)] = L['clay']
    v = np.zeros(Z.shape + (12,), np.float32)
    v[..., 0], v[..., 1], v[..., 2] = X, Y, Z
    v[..., 3:6] = nrm
    v[..., 6], v[..., 7] = X / 16.0, -Y / 16.0             # big texels: reads as colour at this distance
    v[..., 8] = layer
    v[..., 9:12] = 1.0
    ny, nx = Z.shape
    ii = np.arange(ny * nx).reshape(ny, nx)
    a, b, c, d = ii[:-1, :-1], ii[:-1, 1:], ii[1:, 1:], ii[1:, :-1]
    keep = ~(inside[:-1, :-1] & inside[1:, 1:] & inside[:-1, 1:] & inside[1:, :-1])
    tri = np.stack([a, b, c, a, c, d], -1)[keep].reshape(-1).astype(np.uint32)
    return v.reshape(-1, 12), tri


def build(seed=5):
    import village as VL
    rng = np.random.default_rng(seed)
    H = make_heightmap(seed)
    vil = VL.build(H)
    chunks = mesh_terrain(H, vil.paint)
    spots = tree_spots(H, rng, VL.keep_clear)
    for k, parts in mesh_trees(spots, rng).items():
        chunks.setdefault(k, []).extend(parts)
    v, i, ranges = assemble(chunks)
    fv, fi = mesh_far()
    ranges.append((len(i), len(fi), fv[:, :3].min(0), fv[:, :3].max(0)))
    i = np.concatenate([i, fi + len(v)]).astype(np.uint32)
    v = np.concatenate([v, fv])
    return {'H': H, 'vertices': v, 'indices': i, 'chunks': ranges, 'village': vil}


if __name__ == '__main__':
    import time
    t = time.time()
    H = make_heightmap()
    print('heightmap', H.shape, H.min(), H.max(), f'{time.time() - t:.1f}s')
    t = time.time()
    ch = mesh_terrain(H)
    v, i, r = assemble(ch)
    print('terrain verts', len(v), 'tris', len(i) // 3, 'chunks', len(r), f'{time.time() - t:.1f}s')
    t = time.time()
    fv, fi = mesh_far()
    print('far verts', len(fv), 'tris', len(fi) // 3, f'{time.time() - t:.1f}s')
