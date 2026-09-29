"""The world: a sunny plains biome. A wide flat meadow where the run and the field are built, a river across it with
sandy banks and an oak bridge, oak and birch trees around (kept clear of the dominoes), rolling hills further out and
mountains on the horizon.

Heightmap H[ix, iy]: the top of the column at x = ix - HALF, y = iy - Y_OFF (the grass is at z = H). The meadow is
at 0, the riverbed at -2 (its water surface is drawn separately). Produces chunked, indexed triangle meshes
(vertex = pos3, normal3, uv2, layer1, tint3). 1 unit = 1 block.
"""
import numpy as np

import layout as LY
from noise import fbm2d, perlin2d

LAYERS = ['grass_top', 'dirt', 'grass_side', 'stone', 'stone2', 'log', 'log_top', 'leaves', 'sand', 'water',
          'planks', 'birch_log', 'birch_leaves', 'gravel', 'snow', 'snow_side', 'crater']
L = {n: i for i, n in enumerate(LAYERS)}
HALF = 288                        # x in [-HALF, HALF)
Y_OFF = 298                       # y in [-Y_OFF, 2 * HALF - Y_OFF)
CHUNK = 32
WATER_Z = -0.12                   # a full water block's surface, just under the grass
BED = -2
PATCH = (-3, 12, 6, 21)           # blocks around the creeper's spot, drawn by effects.ground_patch (it craters)


class MeshBuilder:
    def __init__(self):
        self.v = []
        self.i = []
        self.n = 0

    def quad(self, p, nrm, uv, layer, tint):
        for k in range(4):
            self.v.append((*p[k], *nrm, *uv[k], layer, *tint))
        b = self.n
        self.i.extend((b, b + 1, b + 2, b, b + 2, b + 3))
        self.n += 4

    def arrays(self):
        return np.array(self.v, np.float32).reshape(-1, 12), np.array(self.i, np.uint32)


def box_face(mb, x0, y0, z0, x1, y1, z1, face, layer, tint=(1.0, 1.0, 1.0)):
    if face == 'pz':
        p = [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        uv = [(x0, -y0), (x1, -y0), (x1, -y1), (x0, -y1)]
        n = (0, 0, 1)
    elif face == 'nz':
        p = [(x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (x0, y0, z0)]
        uv = [(x0, y1), (x1, y1), (x1, y0), (x0, y0)]
        n = (0, 0, -1)
    elif face == 'px':
        p = [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)]
        uv = [(y0, -z0), (y1, -z0), (y1, -z1), (y0, -z1)]
        n = (1, 0, 0)
    elif face == 'nx':
        p = [(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)]
        uv = [(-y1, -z0), (-y0, -z0), (-y0, -z1), (-y1, -z1)]
        n = (-1, 0, 0)
    elif face == 'py':
        p = [(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)]
        uv = [(-x1, -z0), (-x0, -z0), (-x0, -z1), (-x1, -z1)]
        n = (0, 1, 0)
    else:
        p = [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)]
        uv = [(x0, -z0), (x1, -z0), (x1, -z1), (x0, -z1)]
        n = (0, -1, 0)
    mb.quad(p, n, uv, layer, tint)


FACES = {'px': (1, 0, 0), 'nx': (-1, 0, 0), 'py': (0, 1, 0), 'ny': (0, -1, 0), 'pz': (0, 0, 1), 'nz': (0, 0, -1)}


def box(mb, x0, y0, z0, x1, y1, z1, layer, tint=(1.0, 1.0, 1.0), skip=()):
    for f in FACES:
        if f not in skip:
            box_face(mb, x0, y0, z0, x1, y1, z1, f, layer if not isinstance(layer, dict) else layer[f], tint)


# ---------------------------------------------------------------------------------------------
# terrain
# ---------------------------------------------------------------------------------------------
def build_area_distance(X, Y):
    """Distance (blocks) from the part of the meadow that has to stay flat."""
    dx = np.maximum(np.abs(X) - 34.0, 0.0)
    dy = np.maximum(np.maximum(-128.0 - Y, Y - 112.0), 0.0)
    return np.hypot(dx, dy)


def make_heightmap(seed=5):
    xs = np.arange(-HALF, HALF) + 0.5
    ys = np.arange(-Y_OFF, 2 * HALF - Y_OFF) + 0.5
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    d = build_area_distance(X, Y)
    rise = np.clip((d - 6.0) / 60.0, 0.0, 1.0) ** 1.4
    hills = 2.5 + 7.0 * fbm2d(X / 60.0, Y / 60.0, octaves=3, seed=seed) + 2.0 * fbm2d(X / 17.0, Y / 17.0, octaves=2,
                                                                                    seed=seed + 1)
    far = np.clip((d - 90.0) / 110.0, 0.0, 1.0) ** 1.3
    ridge = 1.0 - np.abs(perlin2d(X / 90.0 + 3.1, Y / 90.0 + 1.7, seed=seed + 2))
    mountains = far * (22.0 + 34.0 * ridge ** 2 + 10.0 * fbm2d(X / 30.0, Y / 30.0, octaves=3, seed=seed + 3))
    h = rise * np.maximum(hills, 0.0) + mountains
    # low swells in the meadow's outskirts so it isn't a billiard table (never inside the build area)
    h += np.clip((d - 2.0) / 12.0, 0, 1) * np.maximum(1.2 * fbm2d(X / 23.0, Y / 23.0, octaves=2, seed=seed + 4), 0.0)
    H = np.floor(h + 0.5).astype(np.int32)
    H[H < 0] = 0
    # the river: across the map, straight through the meadow, meandering outside it
    wob = np.where(np.abs(X) > 40.0, 6.0 * np.sin(X / 37.0) * np.clip((np.abs(X) - 40.0) / 30.0, 0, 1), 0.0)
    yc = (LY.RIVER_Y[0] + LY.RIVER_Y[1]) / 2.0 + wob
    half_w = (LY.RIVER_Y[1] - LY.RIVER_Y[0]) / 2.0
    river = np.abs(Y - yc) < half_w
    H[river] = BED
    # the river cuts through the hills: carve its valley
    valley = np.abs(Y - yc) < half_w + 6.0
    H[valley & (H > 0)] = np.minimum(H[valley & (H > 0)], np.clip((np.abs(Y - yc) - half_w) * 0.7, 0, None).astype(
        np.int32)[valley & (H > 0)])
    return H, river


def side_layer(z, h_top, rock):
    depth = h_top - 1 - z
    if depth == 0:
        return L['snow_side'] if h_top >= 44 else L['grass_side']
    if depth <= 2 and not rock:
        return L['dirt']
    return L['stone'] if (z % 5) not in (2,) else L['stone2']


def top_layer(h, rock, sand):
    if sand:
        return L['sand']
    if h >= 44:
        return L['snow']
    if rock:
        return L['stone']
    return L['grass_top']


def build_terrain(H, river, mb_for_chunk, seed=7):
    nx, ny = H.shape
    rng = np.random.default_rng(seed)
    # steep columns show stone; banks next to water are sand
    gy, gx = np.gradient(H.astype(float))
    rock = (np.hypot(gx, gy) > 2.2) | (H >= 38)
    near_water = np.zeros_like(river)
    for dx in (-1, 0, 1):
        for dy in (-2, -1, 0, 1, 2):
            near_water |= np.roll(np.roll(river, dx, 0), dy, 1)
    sand = near_water & ~river & (H <= 0)
    del rng
    px0, py0, px1, py1 = PATCH
    patch = np.zeros_like(river)
    patch[px0 + HALF:px1 + HALF, py0 + Y_OFF:py1 + Y_OFF] = True
    # top faces, greedily merged along y
    for ix in range(nx):
        iy = 0
        while iy < ny:
            if patch[ix, iy]:
                iy += 1
                continue
            h = H[ix, iy]
            lay = top_layer(h, rock[ix, iy], sand[ix, iy]) if h != BED else L['gravel']
            j = iy
            while (j + 1 < ny and H[ix, j + 1] == h and (j + 1 - iy) < 16 and (j + 1) // CHUNK == iy // CHUNK and
                   not patch[ix, j + 1] and
                   (top_layer(H[ix, j + 1], rock[ix, j + 1], sand[ix, j + 1]) if h != BED else L['gravel']) == lay):
                j += 1
            x0 = ix - HALF
            y0 = iy - Y_OFF
            box_face(mb_for_chunk(x0, y0), x0, y0, h - 1, x0 + 1, j + 1 - Y_OFF, h, 'pz', lay)
            iy = j + 1
    # side faces where the neighbour is lower
    dirs = {'px': (1, 0), 'nx': (-1, 0), 'py': (0, 1), 'ny': (0, -1)}
    for ix in range(nx):
        for iy in range(ny):
            h = H[ix, iy]
            x0 = ix - HALF
            y0 = iy - Y_OFF
            for face, (dx, dy) in dirs.items():
                jx, jy = ix + dx, iy + dy
                if not (0 <= jx < nx and 0 <= jy < ny):
                    continue
                hn = H[jx, jy]
                if hn >= h:
                    continue
                mb = mb_for_chunk(x0, y0)
                z = hn
                while z < h:
                    lay = side_layer(z, h, rock[ix, iy])
                    if sand[ix, iy] and z == h - 1:
                        lay = L['sand']
                    z1 = z + 1
                    while z1 < h and side_layer(z1, h, rock[ix, iy]) == lay and lay not in (L['grass_side'],
                                                                                          L['snow_side']):
                        z1 += 1
                    box_face(mb, x0, y0, z, x0 + 1, y0 + 1, z1, face, lay)
                    z = z1


# ---------------------------------------------------------------------------------------------
# trees
# ---------------------------------------------------------------------------------------------
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


def poisson(rng, xmin, xmax, ymin, ymax, mind, accept, max_pts=100000):
    cell = mind / np.sqrt(2)
    gw = int((xmax - xmin) / cell) + 1
    gh = int((ymax - ymin) / cell) + 1
    grid = -np.ones((gw, gh), np.int64)
    pts = []
    tries = 0
    while tries < 60 * max(1, len(pts) + 1) and len(pts) < max_pts:
        tries += 1
        p = (rng.uniform(xmin, xmax), rng.uniform(ymin, ymax))
        if not accept(*p):
            continue
        gx, gy = int((p[0] - xmin) / cell), int((p[1] - ymin) / cell)
        ok = True
        for i in range(max(0, gx - 2), min(gw, gx + 3)):
            for j in range(max(0, gy - 2), min(gh, gy + 3)):
                q = grid[i, j]
                if q >= 0 and (pts[q][0] - p[0]) ** 2 + (pts[q][1] - p[1]) ** 2 < mind * mind:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            grid[gx, gy] = len(pts)
            pts.append(p)
            tries = 0
    return np.array(pts)


def keep_clear(layout):
    """A function (x, y) -> distance to the nearest domino (or the field / the creeper's walk)."""
    from scipy.spatial import cKDTree
    run = layout.run
    tree = cKDTree(np.stack([run['x'], run['y']], -1))
    x0, x1 = -LY.NCOL * LY.PITCH / 2 - 1.0, LY.NCOL * LY.PITCH / 2 + 1.0
    y0, y1 = LY.FEEDER_Y - 1.0, LY.FIELD_Y0 + layout.nrow * LY.SPACING + 1.0

    def dist(x, y):
        d = tree.query((x, y))[0]
        dx = max(x0 - x, 0.0, x - x1)
        dy = max(y0 - y, 0.0, y - y1)
        return min(d, np.hypot(dx, dy))
    return dist


def build_trees(H, river, mb_for_chunk, layout, seed=21):
    rng = np.random.default_rng(seed)
    nx, ny = H.shape
    clear = keep_clear(layout)

    def height_at(x, y):
        ix, iy = int(np.floor(x)) + HALF, int(np.floor(y)) + Y_OFF
        if 0 <= ix < nx and 0 <= iy < ny:
            return H[ix, iy]
        return -99

    def accept(x, y):
        h = height_at(x, y)
        if h < 0 or h >= 36:
            return False
        for dx in (-2, 0, 2):
            for dy in (-2, 0, 2):
                if height_at(x + dx, y + dy) != h:
                    return False
        if abs(y - (LY.RIVER_Y[0] + LY.RIVER_Y[1]) / 2) < 9.0 and abs(x) < 60:
            return False
        c = clear(x, y)
        if c < 9.0:
            return False
        # a few scattered trees near the run for scale, woods further out
        dens = np.clip((c - 9.0) / 30.0, 0.18, 1.0)
        return rng.random() < dens

    pts = poisson(rng, -HALF + 4, HALF - 4, -Y_OFF + 4, 2 * HALF - Y_OFF - 4, 7.0, accept)
    placed = []
    for (x, y) in pts:
        bx, by = int(np.floor(x)), int(np.floor(y))
        h = height_at(bx + 0.5, by + 0.5)
        birch = rng.random() < 0.3
        blocks = tree_blocks(rng, birch)
        tint = LEAF_TINTS[rng.integers(len(LEAF_TINTS))]
        occ = {(b[0], b[1], b[2]) for b in blocks}
        mb = mb_for_chunk(bx, by)
        for (x0, y0, z0, kind) in blocks:
            for face, (dx, dy, dz) in FACES.items():
                if (x0 + dx, y0 + dy, z0 + dz) in occ:
                    continue
                if kind == 'log' and face == 'nz' and z0 == 0:
                    continue
                X0, Y0, Z0 = bx + x0, by + y0, h + z0
                if kind == 'log':
                    lay = L['log_top'] if face in ('pz', 'nz') else L['birch_log' if birch else 'log']
                    t = (1.0, 1.0, 1.0)
                else:
                    lay = L['birch_leaves' if birch else 'leaves']
                    t = tint
                box_face(mb, X0, Y0, Z0, X0 + 1, Y0 + 1, Z0 + 1, face, lay, t)
        placed.append((bx + 0.5, by + 0.5, h, birch))
    return np.array(placed)


# ---------------------------------------------------------------------------------------------
# the bridge
# ---------------------------------------------------------------------------------------------
def build_bridge(mb):
    y0, y1 = LY.RIVER_Y[0] - 1.0, LY.RIVER_Y[1] + 1.0
    # deck: a slab-thick plank floor flush with the meadow, a block wide either side of the dominoes
    box(mb, -1.5, y0, -0.5, 1.5, y1, 0.0, L['planks'])
    # beams under it and log piers in the water
    for x in (-1.5, 1.0):
        box(mb, x, y0, -1.0, x + 0.5, y1, -0.5, L['log'])
    for y in np.arange(y0 + 2.5, y1 - 1.0, 3.5):
        for x in (-1.5, 1.0):
            box(mb, x, y, BED, x + 0.5, y + 0.5, -1.0, L['log'])
    # a low railing along both sides: posts every two blocks and one rail, so the dominoes stay in view
    for side in (-1, 1):
        xc = side * 1.35
        for y in np.arange(y0 + 0.5, y1, 2.0):
            box(mb, xc - 0.125, y - 0.125, 0.0, xc + 0.125, y + 0.125, 0.55, L['log'])
        box(mb, xc - 0.0625, y0 + 0.5, 0.3, xc + 0.0625, y1 - 0.5, 0.45, L['planks'])


def build_world(layout, seed=5):
    H, river = make_heightmap(seed)
    chunks = {}

    def mb_for_chunk(x, y):
        key = (int(np.floor(x / CHUNK)), int(np.floor(y / CHUNK)))
        if key not in chunks:
            chunks[key] = MeshBuilder()
        return chunks[key]

    build_terrain(H, river, mb_for_chunk)
    trees = build_trees(H, river, mb_for_chunk, layout)
    build_bridge(mb_for_chunk(0.0, (LY.RIVER_Y[0] + LY.RIVER_Y[1]) / 2))
    verts, idx, ranges = [], [], []
    base_v = base_i = 0
    for key in sorted(chunks):
        v, i = chunks[key].arrays()
        if len(i) == 0:
            continue
        verts.append(v)
        idx.append(i + base_v)
        ranges.append((base_i, len(i), v[:, :3].min(0), v[:, :3].max(0)))
        base_v += len(v)
        base_i += len(i)
    return {'heightmap': H, 'river': river, 'vertices': np.concatenate(verts),
            'indices': np.concatenate(idx).astype(np.uint32), 'chunks': ranges, 'trees': trees}


def water_mesh(H, river):
    """The river's surface as merged quads (drawn with the animated water)."""
    mb = MeshBuilder()
    nx, ny = H.shape
    for ix in range(nx):
        iy = 0
        while iy < ny:
            if not river[ix, iy]:
                iy += 1
                continue
            j = iy
            while j + 1 < ny and river[ix, j + 1] and j + 1 - iy < 32:
                j += 1
            x0, y0 = ix - HALF, iy - Y_OFF
            box_face(mb, x0, y0, WATER_Z - 1, x0 + 1, j + 1 - Y_OFF, WATER_Z, 'pz', L['water'])
            iy = j + 1
    return mb.arrays()


if __name__ == '__main__':
    import time
    t = time.time()
    w = build_world(LY.Layout())
    print('verts', len(w['vertices']), 'tris', len(w['indices']) // 3, 'chunks', len(w['chunks']),
          'trees', len(w['trees']), 'time %.1fs' % (time.time() - t))
    wv, wi = water_mesh(w['heightmap'], w['river'])
    print('water quads', len(wi) // 6)
