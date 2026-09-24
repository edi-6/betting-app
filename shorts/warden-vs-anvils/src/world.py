"""Static world: a grassy basin surrounded by terraced dirt/stone cliffs, with blocky oak trees.

Produces chunked, indexed triangle meshes (vertex = pos3, normal3, uv2, layer1, tint3).
1 world unit = 1 block = 1 skin pixel of the giant.
"""
import numpy as np
from noise import fbm2d, perlin2d

LAYERS = ['grass_top', 'dirt', 'grass_side', 'stone', 'stone2', 'log', 'log_top', 'leaves', 'sculk']
L = {n: i for i, n in enumerate(LAYERS)}

HALF = 224          # heightmap covers [-HALF, HALF)
CHUNK = 32


class MeshBuilder:
    def __init__(self):
        self.v = []
        self.i = []
        self.n = 0

    def quad(self, p, nrm, uv, layer, tint):
        """p: 4 corners (counter-clockwise seen from the normal side)."""
        for k in range(4):
            self.v.append((*p[k], *nrm, *uv[k], layer, *tint))
        b = self.n
        self.i.extend((b, b + 1, b + 2, b, b + 2, b + 3))
        self.n += 4

    def arrays(self):
        return np.array(self.v, np.float32).reshape(-1, 12), np.array(self.i, np.uint32)


def box_face(mb, x0, y0, z0, x1, y1, z1, face, layer, tint, uv_scale=1.0):
    """Emit one face of an axis aligned box. face in {'px','nx','py','ny','pz','nz'}."""
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
    else:  # 'ny'
        p = [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)]
        uv = [(x0, -z0), (x1, -z0), (x1, -z1), (x0, -z1)]
        n = (0, -1, 0)
    uv = [(u * uv_scale, v * uv_scale) for (u, v) in uv]
    mb.quad(p, n, uv, layer, tint)


# ---------------------------------------------------------------------------------------------
# Terrain
# ---------------------------------------------------------------------------------------------
def cliff_radius(theta, seed):
    return 112 + 16 * perlin2d(np.cos(theta) * 1.7 + 3.1, np.sin(theta) * 1.7 + 7.7, seed=seed) \
        + 6 * perlin2d(np.cos(theta) * 5.0, np.sin(theta) * 5.0, seed=seed + 5)


def make_heightmap(seed=5):
    n = 2 * HALF
    c = np.arange(n) - HALF + 0.5
    X, Y = np.meshgrid(c, c, indexing='ij')     # H[ix, iy] -> block at x = ix - HALF, y = iy - HALF
    r = np.hypot(X, Y)
    th = np.arctan2(Y, X)
    rc = cliff_radius(th, seed)
    plateau = 22 + 5 * fbm2d(X / 70.0, Y / 70.0, octaves=3, seed=seed + 11) \
        + 1.4 * fbm2d(X / 14.0, Y / 14.0, octaves=2, seed=seed + 12)
    # terraced rise: a few big steps with noisy ledge positions
    t = (r - rc) / 13.0 + 0.35 * fbm2d(X / 9.0, Y / 9.0, octaves=2, seed=seed + 13)
    t = np.clip(t, 0, 1)
    steps = 5
    ts = np.floor(t * steps + 0.25 * perlin2d(X / 6.0, Y / 6.0, seed=seed + 14)) / steps
    ts = np.clip(ts, 0, 1)
    ts = np.maximum(ts, (t >= 0.999).astype(float))
    h = plateau * ts
    h = np.where(r < rc - 1, 0, h)
    H = np.round(h).astype(np.int32)
    H[H < 0] = 0
    return H


def side_layer(z, h_top, x, y, seed_noise):
    """Texture for the block at height z (block spans [z, z+1]) in a column whose top is h_top."""
    depth = h_top - 1 - z
    if depth == 0:
        return L['grass_side']
    band = seed_noise
    if depth <= 2 + band:
        return L['dirt']
    # strata: alternating stone / darker stone with occasional dirt seams
    s = (z + band) % 7
    if s in (0,):
        return L['dirt']
    if s in (3, 4):
        return L['stone2']
    return L['stone']


def build_terrain(H, mb_for_chunk):
    n = H.shape[0]
    rng = np.random.default_rng(3)
    seam = rng.integers(0, 2, size=H.shape)
    white = (1.0, 1.0, 1.0)
    grass_tint = (1.0, 1.0, 1.0)
    # top faces: greedy merge per row of equal heights (skip h == 0; the ground plane covers it)
    for ix in range(n):
        iy = 0
        while iy < n:
            h = H[ix, iy]
            if h == 0:
                iy += 1
                continue
            j = iy
            while j + 1 < n and H[ix, j + 1] == h and (j + 1 - iy) < 16 and ((j + 1) // CHUNK == iy // CHUNK):
                j += 1
            x0 = ix - HALF
            y0 = iy - HALF
            mb = mb_for_chunk(x0, y0)
            box_face(mb, x0, y0, h - 1, x0 + 1, j + 1 - HALF, h, 'pz', L['grass_top'], grass_tint)
            iy = j + 1
    # side faces
    dirs = {'px': (1, 0), 'nx': (-1, 0), 'py': (0, 1), 'ny': (0, -1)}
    for ix in range(n):
        for iy in range(n):
            h = H[ix, iy]
            if h == 0:
                continue
            x0 = ix - HALF
            y0 = iy - HALF
            mb = mb_for_chunk(x0, y0)
            for face, (dx, dy) in dirs.items():
                jx, jy = ix + dx, iy + dy
                hn = H[jx, jy] if (0 <= jx < n and 0 <= jy < n) else 0
                if hn >= h:
                    continue
                # vertical runs of identical texture
                z = hn
                while z < h:
                    lay = side_layer(z, h, x0, y0, seam[ix, iy])
                    z1 = z + 1
                    while z1 < h and side_layer(z1, h, x0, y0, seam[ix, iy]) == lay and lay != L['grass_side']:
                        z1 += 1
                    box_face(mb, x0, y0, z, x0 + 1, y0 + 1, z1, face, lay, white)
                    z = z1


# ---------------------------------------------------------------------------------------------
# Trees
# ---------------------------------------------------------------------------------------------
LEAF_TINTS = [(1.00, 1.00, 1.00), (0.86, 1.02, 0.80), (1.10, 1.08, 0.86), (0.78, 0.92, 0.78),
              (1.06, 1.12, 0.92), (0.92, 0.98, 0.84)]


def tree_blocks(rng):
    """Return list of (x, y, z, kind) integer blocks relative to the trunk base. kind: 'log' or 'leaf'."""
    trunk = int(rng.integers(4, 7))
    blocks = [(0, 0, z, 'log') for z in range(trunk)]
    cr = rng.uniform(2.2, 3.4)
    cz = trunk - 0.5 + rng.uniform(-0.3, 0.6)
    rz = cr * rng.uniform(0.75, 0.95)
    R = int(np.ceil(cr)) + 1
    for x in range(-R, R + 1):
        for y in range(-R, R + 1):
            for z in range(int(cz - rz) - 1, int(cz + rz) + 2):
                if x == 0 and y == 0 and z < trunk:
                    continue
                d = (x / cr) ** 2 + (y / cr) ** 2 + ((z - cz) / rz) ** 2
                if d <= 1.0:
                    # thin out the surface a little so canopies aren't perfect ellipsoids
                    if d > 0.72 and rng.random() < 0.28:
                        continue
                    blocks.append((x, y, z, 'leaf'))
    return blocks


def poisson(rng, xmin, xmax, ymin, ymax, mind, accept, k=24, max_pts=100000):
    """Bridson-ish dart throwing on a grid."""
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


def build_trees(H, mb_for_chunk, seed=21, clearing=31.0):
    rng = np.random.default_rng(seed)
    n = H.shape[0]

    def height_at(x, y):
        ix, iy = int(np.floor(x)) + HALF, int(np.floor(y)) + HALF
        if 0 <= ix < n and 0 <= iy < n:
            return H[ix, iy]
        return -1

    def flat_enough(x, y):
        h = height_at(x, y)
        if h < 0:
            return False
        for dx in (-2, 0, 2):
            for dy in (-2, 0, 2):
                if height_at(x + dx, y + dy) != h:
                    return False
        return True

    def accept(x, y):
        r = np.hypot(x, y)
        # slightly irregular clearing edge
        edge = clearing + 3.0 * np.sin(3 * np.arctan2(y, x) + 1.3) + 2.0 * np.sin(7 * np.arctan2(y, x))
        if r < edge:
            return False
        if r > HALF - 4:
            return False
        if not flat_enough(x, y):
            return False
        h = height_at(x, y)
        if h == 0:
            # valley floor: sparse near the clearing edge, dense further out
            dens = np.clip((r - edge) / 14.0, 0.3, 1.0)
            return rng.random() < dens
        return rng.random() < 0.85

    pts = poisson(rng, -HALF + 4, HALF - 4, -HALF + 4, HALF - 4, 6.2, accept)
    placed = []
    for (x, y) in pts:
        bx, by = int(np.floor(x)), int(np.floor(y))
        h = height_at(bx + 0.5, by + 0.5)
        blocks = tree_blocks(rng)
        tint = LEAF_TINTS[rng.integers(len(LEAF_TINTS))]
        occ = {(b[0], b[1], b[2]) for b in blocks}
        mb = mb_for_chunk(bx, by)
        for (x0, y0, z0, kind) in blocks:
            for face, (dx, dy, dz) in (('px', (1, 0, 0)), ('nx', (-1, 0, 0)), ('py', (0, 1, 0)),
                                        ('ny', (0, -1, 0)), ('pz', (0, 0, 1)), ('nz', (0, 0, -1))):
                if (x0 + dx, y0 + dy, z0 + dz) in occ:
                    continue
                if kind == 'log' and face == 'nz' and z0 == 0:
                    continue
                X0, Y0, Z0 = bx + x0, by + y0, h + z0
                if kind == 'log':
                    lay = L['log_top'] if face in ('pz', 'nz') else L['log']
                    t = (1.0, 1.0, 1.0)
                else:
                    lay = L['leaves']
                    t = tint
                box_face(mb, X0, Y0, Z0, X0 + 1, Y0 + 1, Z0 + 1, face, lay, t)
        placed.append((bx + 0.5, by + 0.5, h))
    return np.array(placed)


def build_world(seed=5):
    H = make_heightmap(seed)
    chunks = {}

    def mb_for_chunk(x, y):
        key = (int(np.floor(x / CHUNK)), int(np.floor(y / CHUNK)))
        if key not in chunks:
            chunks[key] = MeshBuilder()
        return chunks[key]

    build_terrain(H, mb_for_chunk)
    trees = build_trees(H, mb_for_chunk)
    verts, idx, ranges = [], [], []
    base_v = 0
    base_i = 0
    for key in sorted(chunks):
        v, i = chunks[key].arrays()
        if len(i) == 0:
            continue
        lo = v[:, :3].min(0)
        hi = v[:, :3].max(0)
        verts.append(v)
        idx.append(i + base_v)
        ranges.append((base_i, len(i), lo, hi))
        base_v += len(v)
        base_i += len(i)
    return {
        'heightmap': H,
        'vertices': np.concatenate(verts),
        'indices': np.concatenate(idx).astype(np.uint32),
        'chunks': ranges,
        'trees': trees,
    }


if __name__ == '__main__':
    import time
    t = time.time()
    w = build_world()
    print('verts', len(w['vertices']), 'tris', len(w['indices']) // 3, 'chunks', len(w['chunks']),
          'trees', len(w['trees']), 'time', time.time() - t)
