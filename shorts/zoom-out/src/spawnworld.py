"""The spawn area in 3D: 640 x 640 blocks round the origin, built from the map shader's own fields (zoommap.py in
'fields' mode gives height, river, biome and trees at one pixel a block), so the 3D world and the map are the same
world and the zoom can hand over from one to the other without a seam.

On top of the generated land: Steve's camp (the campfire he's lying by, his little house with a chimney, a wheat
farm, a crafting table, a furnace and a chest, a sheep pen), a dirt path south to a village (a well, ten houses, a
church with a tower, a blacksmith with a lava pool, farms, pens, lamp posts, villagers, an iron golem) and the
village's road east over the river on a bridge.

Blocks go into a dense voxel grid G[ix, iy, iz] (x = ix - HALF, y = iy - HALF, z = iz + ZMIN) of block ids, meshed
with hidden faces removed into chunked triangle meshes (vertex = pos3, normal3, uv2, layer1, tint3; 1 unit = 1
block). The water's surface and the things that aren't whole blocks (the campfire, torches, fences, lamp posts, mobs)
are meshed separately. The grass top of the ground at Steve's spot is z = 0.
"""
import numpy as np
from scipy import ndimage

import zblocks as ZB

TL = ZB.L                         # texture layers
HALF = 320                        # x, y in [-HALF, HALF)
ZMIN, NZ = -6, 34                 # z in [ZMIN, ZMIN + NZ)
CHUNK = 32
WATER_Z = -1.12                   # the river's surface: water under the banks' grass (at -1)
VILLAGE = (-12, -88)              # the village's well (the map's shader keeps this spot open plains)

# block id -> textures on its faces (+x, -x, +y, -y, +z, -z)
_DEFS = [('air', None)]


def _b(name, top, side=None, bottom=None, faces=None):
    side = side or top
    bottom = bottom or top
    _DEFS.append((name, faces or (side, side, side, side, top, bottom)))


_b('grass', 'grass_top', 'grass_side', 'dirt')
_b('dirt', 'dirt')
_b('stone', 'stone')
_b('sand', 'sand')
_b('gravel', 'gravel')
_b('path', 'path', 'dirt')
_b('farmland', 'farmland', 'dirt')
_b('wheat', 'wheat', 'dirt')
_b('carrots', 'carrots', 'dirt')
_b('log', 'log_top', 'log')
_b('log_x', None, faces=('log_top', 'log_top', 'log', 'log', 'log', 'log'))
_b('log_y', None, faces=('log', 'log', 'log_top', 'log_top', 'log', 'log'))
_b('leaves', 'leaves')
_b('birch_log', 'log_top', 'birch_log')
_b('birch_leaves', 'birch_leaves')
_b('planks', 'planks')
_b('cobblestone', 'cobblestone')
_b('mossy_cobble', 'mossy_cobble')
_b('stone_bricks', 'stone_bricks')
_b('glass', 'glass', 'glass', 'planks')
_b('spruce_planks', 'spruce_planks')
_b('dark_planks', 'dark_planks')
_b('hay', 'hay_top', 'hay_side')
_b('crafting', 'crafting_top', 'planks')
_b('chest', 'chest_top')
_b('furnace', 'furnace_top', 'cobblestone')
_b('wool', 'white_wool')
_b('door', 'planks', 'door')
_b('poppy', 'poppy_grass', 'grass_side', 'dirt')
_b('dandelion', 'dandelion_grass', 'grass_side', 'dirt')
_b('cornflower', 'cornflower_grass', 'grass_side', 'dirt')
_b('bed', 'bed_red', 'bed_red', 'planks')
_b('lava', 'lava')
_b('bricks_dark', 'dark_planks', 'stone_bricks')
B = {name: i for i, (name, _) in enumerate(_DEFS)}
FACE_LAYER = np.zeros((len(_DEFS), 6), np.float32)
for _i, (_n, _f) in enumerate(_DEFS):
    if _f is not None:
        FACE_LAYER[_i] = [TL[t] for t in _f]

# faces: +x, -x, +y, -y, +z, -z: corners of the unit cube (in the order the renderer's box faces use), the normal,
# and which coordinate (and sign) gives u and v
CORNERS = np.array([
    [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],
    [(0, 1, 0), (0, 0, 0), (0, 0, 1), (0, 1, 1)],
    [(1, 1, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1)],
    [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
    [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
    [(0, 1, 0), (1, 1, 0), (1, 0, 0), (0, 0, 0)]], np.float32)
NORMALS = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], np.float32)
UVMAP = [((1, 1), (2, -1)), ((1, -1), (2, -1)), ((0, -1), (2, -1)), ((0, 1), (2, -1)), ((0, 1), (1, -1)),
         ((0, 1), (1, 1))]


def _hash(*a):
    """A deterministic hash of integer coordinates to [0, 1)."""
    with np.errstate(over='ignore'):
        shape = np.broadcast(*[np.asarray(v) for v in a]).shape
        h = np.full(shape, 0x9E3779B97F4A7C15, np.uint64)
        for v in a:
            vi = np.floor(np.asarray(v, np.float64)).astype(np.int64).astype(np.uint64)
            h ^= vi + np.uint64(0x9E3779B97F4A7C15) + (h << np.uint64(6)) + (h >> np.uint64(2))
            h *= np.uint64(0xBF58476D1CE4E5B9)
            h ^= h >> np.uint64(31)
            h *= np.uint64(0x94D049BB133111EB)
            h ^= h >> np.uint64(29)
        return (h >> np.uint64(40)).astype(np.float64) / float(1 << 24)


# ---------------------------------------------------------------------------------------------
# quads
# ---------------------------------------------------------------------------------------------
def face_quads(lo, hi, d, layer, tint):
    """Quads of face d of boxes lo..hi (n, 3): (n, 4, 12) vertices."""
    n = len(lo)
    c = CORNERS[d]
    pos = lo[:, None, :] + c[None] * (hi - lo)[:, None, :]
    (au, su), (av, sv) = UVMAP[d]
    v = np.zeros((n, 4, 12), np.float32)
    v[..., 0:3] = pos
    v[..., 3:6] = NORMALS[d]
    v[..., 6] = su * pos[..., au]
    v[..., 7] = sv * pos[..., av]
    v[..., 8] = np.broadcast_to(np.asarray(layer, np.float32).reshape(-1, 1), (n, 4))
    v[..., 9:12] = np.broadcast_to(np.asarray(tint, np.float32).reshape(-1, 1, 3), (n, 4, 3))
    return v


class Details:
    """Things that aren't whole blocks: boxes of any size, in a local frame turned by a yaw and moved."""

    def __init__(self):
        self.parts = []

    def box(self, lo, hi, layers, tint=(1.0, 1.0, 1.0), yaw=0.0, at=(0.0, 0.0, 0.0), skip=()):
        lo = np.asarray(lo, np.float32)[None]
        hi = np.asarray(hi, np.float32)[None]
        if isinstance(layers, str):
            layers = (layers,) * 6
        elif len(layers) == 3:
            layers = (layers[1],) * 4 + (layers[0], layers[2])        # (top, side, bottom)
        cy, sy = np.cos(yaw), np.sin(yaw)
        R = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], np.float32)
        for d in range(6):
            if d in skip:
                continue
            q = face_quads(lo, hi, d, TL[layers[d]], tint)
            if yaw:
                q[..., 0:3] = q[..., 0:3] @ R.T
                q[..., 3:6] = q[..., 3:6] @ R.T
            q[..., 0:3] += np.asarray(at, np.float32)
            self.parts.append(q)

    def quads(self):
        if not self.parts:
            return np.zeros((0, 4, 12), np.float32)
        return np.concatenate(self.parts)


# ---------------------------------------------------------------------------------------------
# the grid
# ---------------------------------------------------------------------------------------------
class World:
    def __init__(self, fields):
        """fields: (2 HALF, 2 HALF, 4) from MapRenderer.fields(HALF), row 0 = north."""
        F = fields[::-1].transpose(1, 0, 2)              # F[ix, iy]
        self.F = F
        n = 2 * HALF
        self.n = n
        self.h0 = float(F[HALF, HALF, 0])
        xs = np.arange(n) - HALF + 0.5
        self.X, self.Y = np.meshgrid(xs, xs, indexing='ij')
        self.water = F[..., 1] > 0.5
        self.G = None
        self.Z = None
        self.top = None                                  # block id of each column's top block
        self.waters = []                                 # (ix, iy, surface z)
        self.details = Details()
        self.smoke = []                                  # (x, y, z, strength): chimneys, the campfire
        self.lights = []                                 # (x, y, z, radius): flames
        self.clear = np.zeros((n, n), bool)              # no trees here (buildings, roads, farms)
        self.ops = []
        self.rng = np.random.default_rng(23)

    # -- helpers ------------------------------------------------------------------------------
    def ij(self, x, y):
        return int(x) + HALF, int(y) + HALF

    def zg(self, x, y):
        i, j = self.ij(x, y)
        return int(self.Z[i, j])

    def fill(self, x0, x1, y0, y1, z0, z1, name):
        """Blocks in [x0, x1) x [y0, y1) x [z0, z1) (placed once the ground is filled, in order)."""
        i0, j0 = self.ij(x0, y0)
        i1, j1 = self.ij(x1, y1)
        k0, k1 = int(z0) - ZMIN, int(z1) - ZMIN
        self.ops.append((max(i0, 0), max(i1, 0), max(j0, 0), max(j1, 0), max(k0, 0), max(k1, 0),
                         B[name] if isinstance(name, str) else name))

    def apply_ops(self):
        for (i0, i1, j0, j1, k0, k1, b) in self.ops:
            self.G[i0:i1, j0:j1, k0:k1] = b
        self.ops = []

    def level(self, x0, x1, y0, y1, z):
        i0, j0 = self.ij(x0, y0)
        i1, j1 = self.ij(x1, y1)
        self.Z[i0:i1, j0:j1] = z

    def set_top(self, x0, x1, y0, y1, name, z=None):
        """Replace the top block of the ground (optionally levelling it to z)."""
        i0, j0 = self.ij(x0, y0)
        i1, j1 = self.ij(x1, y1)
        if z is not None:
            self.Z[i0:i1, j0:j1] = z
        self.top[i0:i1, j0:j1] = B[name]

    def mark_clear(self, x0, x1, y0, y1, pad=2):
        i0, j0 = self.ij(x0 - pad, y0 - pad)
        i1, j1 = self.ij(x1 + pad, y1 + pad)
        self.clear[max(i0, 0):i1, max(j0, 0):j1] = True

    @staticmethod
    def box_dist(X, Y, x0, x1, y0, y1):
        dx = np.maximum(np.maximum(x0 - X, X - x1), 0.0)
        dy = np.maximum(np.maximum(y0 - Y, Y - y1), 0.0)
        return np.hypot(dx, dy)

    # -- the land -------------------------------------------------------------------------------
    def terrain(self):
        F, X, Y = self.F, self.X, self.Y
        zr = (F[..., 0] - self.h0) * 0.6                 # (gentler than the map's: fewer one-block steps)
        # Steve's camp and the village are levelled
        w = np.maximum(1.0 - np.clip((self.box_dist(X, Y, -16, 24, -14, 8) - 4.0) / 16.0, 0, 1),
                       1.0 - np.clip((self.box_dist(X, Y, -74, 30, -132, -62) - 3.0) / 18.0, 0, 1))
        w = w * w * (3 - 2 * w)
        Z = np.floor(zr * (1.0 - w) + 0.5)
        # the river: banks slope down to it, its bed is deeper in the middle
        d_out = ndimage.distance_transform_edt(~self.water)
        d_in = ndimage.distance_transform_edt(self.water)
        bank = -1.0 + np.floor(np.maximum(d_out - 1.0, 0.0) * 0.75)
        Z = np.where(d_out < 9, np.minimum(Z, bank), Z)
        bed = -2.0 - np.minimum(2.0, np.floor(d_in / 2.6))
        Z = np.where(self.water, bed, Z)
        self.Z = Z.astype(np.int32)
        self.d_water = d_out
        top = np.full(Z.shape, B['grass'], np.uint8)
        top[(d_out <= 2.5) & ~self.water] = B['sand']
        g = _hash(np.arange(self.n)[:, None] + np.zeros((1, self.n)), np.arange(self.n)[None, :] + np.zeros((self.n, 1)), 7)
        top[self.water] = np.where(g[self.water] < 0.35, B['gravel'], B['sand'])
        self.top = top
        for (ix, iy) in zip(*np.nonzero(self.water)):
            self.waters.append((ix, iy, WATER_Z))

    def fill_ground(self):
        k = (np.arange(NZ) + ZMIN)[None, None, :]
        Zc = self.Z[..., None]
        G = np.zeros((self.n, self.n, NZ), np.uint8)
        G[k < Zc] = B['stone']
        G[(k >= Zc - 4) & (k < Zc - 1)] = B['dirt']
        sandy = (self.top == B['sand']) | (self.top == B['gravel'])
        sub = np.where(sandy, B['sand'], B['dirt'])[..., None]
        G = np.where((k >= Zc - 4) & (k < Zc - 1), sub, G).astype(np.uint8)
        top = np.broadcast_to(self.top[..., None], G.shape)
        G = np.where(k == Zc - 1, top, G).astype(np.uint8)
        self.G = G

    # -- trees ------------------------------------------------------------------------------------
    def trees(self):
        """Oak (and now and then birch) trees where the map has them: the canopy's footprint is the map's."""
        F, G = self.F, self.G
        ti, tj = np.nonzero(F[..., 3] > 0)
        clear = ndimage.binary_dilation(self.clear | (self.d_water < 3.0), iterations=4)
        o = np.arange(-4, 5)
        DX, DY = np.meshgrid(o, o, indexing='ij')
        DM = np.maximum(np.abs(DX), np.abs(DY)) * 0.55 + np.hypot(DX, DY) * 0.45
        rag = self.rng.random((4,) + DX.shape + (64,)) < 0.3
        count = 0
        for t, (i, j) in enumerate(zip(ti, tj)):
            if clear[i, j]:
                continue
            x, y = i - HALF, j - HALF
            r = float(F[i, j, 3])
            th = 4 + int(_hash(x, y, 41) * 3)
            birch = _hash(x, y, 43) > 0.9
            z0 = int(self.Z[i, j])
            log, leaf = (B['birch_log'], B['birch_leaves']) if birch else (B['log'], B['leaves'])
            G[i, j, z0 - ZMIN:z0 + th - ZMIN] = log
            for li, (dz, R) in enumerate(((th - 2, r), (th - 1, r), (th, r - 1.0), (th + 1, max(r - 2.0, 1.0)))):
                sel = (DM < R) & ~((DM > R - 1.0) & rag[li, ..., t % 64])
                ii, jj = i + DX[sel], j + DY[sel]
                ok = (ii >= 0) & (ii < self.n) & (jj >= 0) & (jj < self.n)
                ii, jj = ii[ok], jj[ok]
                k = z0 + dz - ZMIN
                if not 0 <= k < NZ:
                    continue
                free = G[ii, jj, k] == 0
                G[ii[free], jj[free], k] = leaf
            count += 1
        return count

    # -- builds ----------------------------------------------------------------------------------
    def flowers(self, cx, cy, rad, frac, seed):
        X, Y = self.X, self.Y
        near = np.hypot(X - cx, Y - cy) < rad
        g = _hash(X + 1000, Y + 1000, seed)
        g2 = _hash(X + 1000, Y + 1000, seed + 1)
        sel = near & (g < frac) & (self.top == B['grass'])
        kinds = np.array([B['poppy'], B['dandelion'], B['cornflower']], np.uint8)
        self.top[sel] = kinds[np.minimum((g2[sel] * 3).astype(int), 2)]

    def path_line(self, pts, width=1.15, name='path'):
        pts = np.asarray(pts, float)
        X, Y = self.X, self.Y
        d = np.full(X.shape, 1e9)
        for a, b in zip(pts[:-1], pts[1:]):
            ab = b - a
            t = np.clip(((X - a[0]) * ab[0] + (Y - a[1]) * ab[1]) / max(ab @ ab, 1e-9), 0, 1)
            d = np.minimum(d, np.hypot(X - a[0] - t * ab[0], Y - a[1] - t * ab[1]))
        sel = (d < width) & ~self.water & (self.top != B['sand'] if name == 'path' else True)
        self.top[sel] = B[name]
        self.clear |= d < width + 1.5
        return sel

    def house(self, x0, y0, w, d, wall='planks', frame='log', roof='spruce_planks', ridge='y', wall_h=4,
              door=('ny', 0.5), chimney=None, base='cobblestone', windows='glass', flat=False):
        z0 = self.zg(x0 + w // 2, y0 + d // 2)
        self.level(x0 - 2, x0 + w + 2, y0 - 2, y0 + d + 2, z0)
        self.set_top(x0, x0 + w, y0, y0 + d, base, z0)
        # walls
        for z in range(z0, z0 + wall_h):
            m = 'cobblestone' if (z == z0 and base == 'cobblestone' and wall != 'cobblestone') else wall
            self.fill(x0, x0 + w, y0, y0 + 1, z, z + 1, m)
            self.fill(x0, x0 + w, y0 + d - 1, y0 + d, z, z + 1, m)
            self.fill(x0, x0 + 1, y0, y0 + d, z, z + 1, m)
            self.fill(x0 + w - 1, x0 + w, y0, y0 + d, z, z + 1, m)
        if frame:
            for (cx, cy) in ((x0, y0), (x0 + w - 1, y0), (x0, y0 + d - 1), (x0 + w - 1, y0 + d - 1)):
                self.fill(cx, cx + 1, cy, cy + 1, z0, z0 + wall_h, frame)
        if windows:
            for x in range(x0 + 2, x0 + w - 2, 2):
                for yy in (y0, y0 + d - 1):
                    self.fill(x, x + 1, yy, yy + 1, z0 + 1, z0 + 3, windows)
            for y in range(y0 + 2, y0 + d - 2, 2):
                for xx in (x0, x0 + w - 1):
                    self.fill(xx, xx + 1, y, y + 1, z0 + 1, z0 + 3, windows)
        side, f = door
        if side in ('ny', 'py'):
            dx = x0 + int(round(f * (w - 1)))
            dy = y0 if side == 'ny' else y0 + d - 1
        else:
            dy = y0 + int(round(f * (d - 1)))
            dx = x0 if side == 'nx' else x0 + w - 1
        self.fill(dx, dx + 1, dy, dy + 1, z0, z0 + 2, 'door')
        ox, oy = {'ny': (0, -1), 'py': (0, 1), 'nx': (-1, 0), 'px': (1, 0)}[side]
        self.set_top(dx + ox, dx + ox + 1, dy + oy, dy + oy + 1, 'path')
        self.set_top(dx + 2 * ox, dx + 2 * ox + 1, dy + 2 * oy, dy + 2 * oy + 1, 'path')
        # the roof
        zt = z0 + wall_h
        if flat:
            self.fill(x0 - 1, x0 + w + 1, y0 - 1, y0 + d + 1, zt, zt + 1, roof)
            self.fill(x0, x0 + w, y0, y0 + d, zt + 1, zt + 2, frame or roof)
            self.fill(x0 + 1, x0 + w - 1, y0 + 1, y0 + d - 1, zt + 1, zt + 2, roof)
        elif ridge == 'y':
            n = w + 2
            for c in range(n):
                i = min(c, n - 1 - c)
                x = x0 - 1 + c
                self.fill(x, x + 1, y0 - 1, y0 + d + 1, zt - 1 + i, zt + i, roof)
                for yy in (y0, y0 + d - 1):
                    if x0 <= x < x0 + w and i >= 1:
                        self.fill(x, x + 1, yy, yy + 1, zt, zt - 1 + i, wall)
        else:
            n = d + 2
            for c in range(n):
                i = min(c, n - 1 - c)
                y = y0 - 1 + c
                self.fill(x0 - 1, x0 + w + 1, y, y + 1, zt - 1 + i, zt + i, roof)
                for xx in (x0, x0 + w - 1):
                    if y0 <= y < y0 + d and i >= 1:
                        self.fill(xx, xx + 1, y, y + 1, zt, zt - 1 + i, wall)
        if chimney is not None:
            cx, cy = chimney
            top = zt + (min(w, d) + 2) // 2 + 1
            self.fill(cx, cx + 1, cy, cy + 1, z0, top, 'cobblestone')
            self.smoke.append((cx + 0.5, cy + 0.5, top + 0.1, 0.6))
        self.mark_clear(x0, x0 + w, y0, y0 + d, pad=3)
        return z0

    def farm(self, x0, y0, w, d, crops=('wheat', 'carrots')):
        z0 = self.zg(x0 + w // 2, y0 + d // 2)
        # a log frame, crops in rows, a water channel along the middle
        self.set_top(x0, x0 + w, y0, y0 + d, 'log_x', z0)
        self.set_top(x0, x0 + 1, y0, y0 + d, 'log_y')
        self.set_top(x0 + w - 1, x0 + w, y0, y0 + d, 'log_y')
        mid = y0 + d // 2
        for y in range(y0 + 1, y0 + d - 1):
            crop = crops[0] if y < mid else crops[-1]
            self.set_top(x0 + 1, x0 + w - 1, y, y + 1, crop)
        i0, j0 = self.ij(x0 + 1, mid)
        i1, _ = self.ij(x0 + w - 1, mid)
        self.Z[i0:i1, j0] = z0 - 1
        self.top[i0:i1, j0] = B['dirt']
        for ii in range(i0, i1):
            self.waters.append((ii, j0, z0 - 0.12))
        self.mark_clear(x0, x0 + w, y0, y0 + d, pad=2)
        return z0

    def fence(self, x0, y0, w, d, gap=None):
        """Fence posts round [x0, x0 + w) x [y0, y0 + d) (block cells), with rails between them."""
        D = self.details
        cells = []
        for x in range(x0, x0 + w):
            cells += [(x, y0), (x, y0 + d - 1)]
        for y in range(y0 + 1, y0 + d - 1):
            cells += [(x0, y), (x0 + w - 1, y)]
        cells = sorted(set(cells))
        if gap is not None:
            cells = [c for c in cells if c != gap]
        cs = set(cells)
        for (x, y) in cells:
            z = self.zg(x, y)
            D.box((x + 0.375, y + 0.375, z), (x + 0.625, y + 0.625, z + 1.5), 'fence')
            for (nx, ny) in ((x + 1, y), (x, y + 1)):
                if (nx, ny) in cs:
                    for zz in (0.375, 0.875):
                        if nx > x:
                            D.box((x + 0.625, y + 0.44, z + zz), (x + 1.375, y + 0.56, z + zz + 0.19), 'fence')
                        else:
                            D.box((x + 0.44, y + 0.625, z + zz), (x + 0.56, y + 1.375, z + zz + 0.19), 'fence')
        self.mark_clear(x0, x0 + w, y0, y0 + d, pad=1)

    def torch(self, x, y, z=None, post=0.0):
        """A torch standing at the centre of block (x, y); post: a fence post under it (lamp posts)."""
        D = self.details
        if z is None:
            z = self.zg(x, y)
        if post:
            D.box((x + 0.375, y + 0.375, z), (x + 0.625, y + 0.625, z + post), 'fence')
            z = z + post
        D.box((x + 0.4375, y + 0.4375, z), (x + 0.5625, y + 0.5625, z + 0.5), ('flame', 'planks', 'planks'))
        D.box((x + 0.42, y + 0.42, z + 0.5), (x + 0.58, y + 0.58, z + 0.66), 'flame')
        self.lights.append((x + 0.5, y + 0.5, z + 0.62, 6.0))

    def mob(self, kind, x, y, yaw=0.0, z=None):
        """Simple box mobs, facing +y before the yaw."""
        D = self.details
        if z is None:
            z = self.zg(x, y)
        at = (x, y, z)
        W = 'white_wool'
        if kind == 'sheep':
            body = (0.95, 0.95, 0.93)
            for lx, ly in ((-0.3, -0.45), (0.3, -0.45), (-0.3, 0.4), (0.3, 0.4)):
                D.box((lx - 0.12, ly - 0.12, 0), (lx + 0.12, ly + 0.12, 0.5), W, (0.82, 0.74, 0.66), yaw, at)
            D.box((-0.45, -0.65, 0.45), (0.45, 0.65, 1.25), W, body, yaw, at)
            D.box((-0.25, 0.6, 0.85), (0.25, 1.05, 1.45), W, (0.84, 0.76, 0.66), yaw, at)
            D.box((-0.28, 0.58, 1.2), (0.28, 0.9, 1.5), W, body, yaw, at)
        elif kind == 'cow':
            for lx, ly in ((-0.3, -0.5), (0.3, -0.5), (-0.3, 0.5), (0.3, 0.5)):
                D.box((lx - 0.13, ly - 0.13, 0), (lx + 0.13, ly + 0.13, 0.75), 'cow', (1, 1, 1), yaw, at)
            D.box((-0.45, -0.75, 0.7), (0.45, 0.75, 1.45), 'cow', (1, 1, 1), yaw, at)
            D.box((-0.3, 0.72, 0.95), (0.3, 1.12, 1.55), W, (0.42, 0.3, 0.22), yaw, at)
            D.box((-0.4, 0.82, 1.48), (0.4, 0.92, 1.6), W, (0.85, 0.82, 0.72), yaw, at)
        elif kind == 'pig':
            for lx, ly in ((-0.25, -0.35), (0.25, -0.35), (-0.25, 0.35), (0.25, 0.35)):
                D.box((lx - 0.12, ly - 0.12, 0), (lx + 0.12, ly + 0.12, 0.38), 'pink', (1, 1, 1), yaw, at)
            D.box((-0.32, -0.55, 0.35), (0.32, 0.55, 0.9), 'pink', (1, 1, 1), yaw, at)
            D.box((-0.27, 0.5, 0.45), (0.27, 0.95, 0.98), 'pink', (1, 1, 1), yaw, at)
            D.box((-0.13, 0.94, 0.55), (0.13, 1.02, 0.72), 'pink', (0.85, 0.62, 0.62), yaw, at)
        elif kind == 'villager':
            robes = [(0.48, 0.34, 0.22), (0.62, 0.62, 0.6), (0.3, 0.45, 0.26), (0.52, 0.3, 0.46), (0.28, 0.3, 0.5)]
            robe = robes[int(_hash(np.array(int(x * 7)), np.array(int(y * 7)), 9) * len(robes)) % len(robes)]
            skin = (0.78, 0.6, 0.46)
            D.box((-0.25, -0.17, 0), (0.25, 0.17, 1.35), W, robe, yaw, at)
            D.box((-0.27, -0.12, 0.8), (0.27, 0.26, 1.1), W, tuple(c * 0.85 for c in robe), yaw, at)
            D.box((-0.25, -0.25, 1.35), (0.25, 0.25, 1.95), W, skin, yaw, at)
            D.box((-0.06, 0.25, 1.45), (0.06, 0.37, 1.7), W, (0.72, 0.52, 0.4), yaw, at)
        elif kind == 'golem':
            iron = (0.88, 0.85, 0.78)
            D.box((-0.42, -0.2, 0), (-0.08, 0.15, 1.05), W, iron, yaw, at)
            D.box((0.08, -0.2, 0), (0.42, 0.15, 1.05), W, iron, yaw, at)
            D.box((-0.6, -0.3, 1.0), (0.6, 0.3, 2.2), W, iron, yaw, at)
            D.box((-0.82, -0.18, 0.55), (-0.6, 0.18, 2.15), W, iron, yaw, at)
            D.box((0.6, -0.18, 0.55), (0.82, 0.18, 2.15), W, iron, yaw, at)
            D.box((-0.25, -0.1, 2.2), (0.25, 0.4, 2.8), W, iron, yaw, at)
            D.box((-0.62, -0.31, 1.7), (0.0, 0.31, 2.21), 'leaves', (0.8, 0.9, 0.8), yaw, at)
        elif kind == 'cat':
            fur = (0.86, 0.58, 0.3)
            D.box((-0.15, -0.35, 0.2), (0.15, 0.35, 0.45), W, fur, yaw, at)
            D.box((-0.15, 0.33, 0.3), (0.15, 0.58, 0.55), W, fur, yaw, at)
            D.box((-0.04, -0.75, 0.3), (0.04, -0.35, 0.38), W, fur, yaw, at)
            for lx, ly in ((-0.1, -0.28), (0.1, -0.28), (-0.1, 0.28), (0.1, 0.28)):
                D.box((lx - 0.05, ly - 0.05, 0), (lx + 0.05, ly + 0.05, 0.2), W, fur, yaw, at)
        elif kind == 'wolf':
            # Steve's dog, lying beside him: white-grey, a red collar
            fur = (0.9, 0.88, 0.85)
            D.box((-0.27, -0.55, 0.0), (0.27, 0.45, 0.42), W, fur, yaw, at)
            D.box((-0.31, -0.62, 0.0), (0.31, -0.05, 0.47), W, (0.82, 0.8, 0.77), yaw, at)
            D.box((-0.24, 0.42, 0.06), (0.24, 0.5, 0.52), W, (0.8, 0.14, 0.12), yaw, at)
            D.box((-0.23, 0.5, 0.05), (0.23, 0.88, 0.5), W, fur, yaw, at)
            D.box((-0.11, 0.88, 0.07), (0.11, 1.1, 0.26), W, (0.75, 0.72, 0.68), yaw, at)
            D.box((-0.04, 1.06, 0.2), (0.04, 1.11, 0.27), W, (0.12, 0.1, 0.1), yaw, at)
            for ex in (-0.17, 0.11):
                D.box((ex, 0.6, 0.5), (ex + 0.07, 0.72, 0.64), W, (0.8, 0.78, 0.74), yaw, at)
            D.box((-0.06, -1.05, 0.08), (0.06, -0.55, 0.2), W, fur, yaw, at)
            for lx in (-0.2, 0.12):
                D.box((lx, 0.45, 0.0), (lx + 0.09, 0.85, 0.12), W, fur, yaw, at)

    # -- Steve's camp ----------------------------------------------------------------------------
    def camp(self):
        D = self.details
        self.mark_clear(-14, 12, -12, 8, pad=4)
        # the campfire by his head, log benches round it
        z = 0
        D.box((-3, 1, z), (-2, 2, z + 0.44), ('campfire', 'log', 'log'))
        self.lights.append((-2.5, 1.5, 0.6, 7.0))
        self.smoke.append((-2.5, 1.5, 0.5, 1.0))
        self.fill(-5, -4, 0, 2, 0, 1, 'log_y')
        self.fill(-4, -2, 3, 4, 0, 1, 'log_x')
        # his house east of him: oak walls, a dark oak roof, a chimney, the door on the south side
        self.house(4, -4, 7, 7, wall='planks', frame='log', roof='spruce_planks', ridge='y', wall_h=4,
                   door=('ny', 0.33), chimney=(8, 1))
        # crafting table, furnace and chest by the house, a bed outside (he sleeps under the stars)
        self.fill(2, 3, 1, 2, 0, 1, 'crafting')
        self.fill(2, 3, 2, 3, 0, 1, 'furnace')
        self.fill(2, 3, -3, -2, 0, 1, 'chest')
        self.torch(-1, 3)
        self.torch(3, -5)
        self.torch(11, -5)
        # the farm west of him
        self.farm(-13, -10, 8, 7)
        self.torch(-14, -11)
        self.torch(-4, -11)
        # a sheep pen east of the house
        self.fence(14, -4, 7, 7, gap=(14, -1))
        self.mob('sheep', 16.2, -2.0, 0.6)
        self.mob('sheep', 18.6, 0.3, 2.4)
        self.mob('cow', 17.5, 1.6, -1.2)
        # his dog lies beside him
        self.mob('wolf', -1.15, -1.0, -0.12, z=0)
        # flowers in the grass
        self.flowers(0, -2, 15, 0.08, 31)
        # a path from his door south to the village
        self.path_line([(6.5, -5), (5.5, -9), (2, -14), (-3, -22), (-7, -32), (-10, -44), (-11.5, -56),
                        (-11.5, -70)])

    # -- the village -----------------------------------------------------------------------------
    def village(self):
        cx, cy = VILLAGE
        D = self.details
        self.mark_clear(-72, 28, -130, -62, pad=0)
        # roads and the plaza
        self.path_line([(-11.5, -70), (-11.5, -140)], 1.6)
        self.path_line([(-74, -88.5), (27, -88.5)], 1.6)
        self.path_line([(46, -88.5), (64, -88.5), (72, -80)], 1.6)
        self.set_top(cx - 6, cx + 6, cy - 6, cy + 6, 'path')
        # the well: a cobblestone rim, water inside
        z0 = self.zg(cx, cy)
        self.fill(cx - 2, cx + 2, cy - 2, cy + 2, z0, z0 + 1, 'cobblestone')
        i0, j0 = self.ij(cx - 1, cy - 1)
        self.fill(cx - 1, cx + 1, cy - 1, cy + 1, z0, z0 + 1, 'air')
        self.Z[i0:i0 + 2, j0:j0 + 2] = z0 - 2
        self.top[i0:i0 + 2, j0:j0 + 2] = B['cobblestone']
        for a in range(2):
            for b in range(2):
                self.waters.append((i0 + a, j0 + b, z0 + 0.88))
        for (px, py) in ((cx - 2, cy - 2), (cx + 1, cy - 2), (cx - 2, cy + 1), (cx + 1, cy + 1)):
            D.box((px + 0.375, py + 0.375, z0 + 1), (px + 0.625, py + 0.625, z0 + 3), 'fence')
        D.box((cx - 2, cy - 2, z0 + 3), (cx + 2, cy + 2, z0 + 3.5), 'cobblestone')
        # the bell
        D.box((cx + 4.375, cy + 3.375, z0), (cx + 4.625, cy + 3.625, z0 + 1.6), 'fence')
        D.box((cx + 4.2, cy + 3.2, z0 + 1.6), (cx + 4.8, cy + 3.8, z0 + 2.3), 'bell')
        # houses
        H = [(-30, -80, 7, 7, dict(roof='spruce_planks', ridge='x', door=('ny', 0.5), chimney=(-25, -76))),
             (-32, -106, 7, 8, dict(roof='planks', ridge='y', door=('px', 0.5))),
             (-2, -80, 7, 6, dict(roof='planks', ridge='x', door=('ny', 0.3))),
             (-1, -106, 8, 7, dict(roof='spruce_planks', ridge='y', door=('nx', 0.5), chimney=(4, -102))),
             (-50, -84, 6, 6, dict(roof='planks', flat=True, door=('ny', 0.5))),
             (8, -84, 6, 6, dict(roof='spruce_planks', ridge='x', door=('ny', 0.5))),
             (-52, -104, 6, 6, dict(roof='planks', ridge='y', door=('py', 0.5))),
             (10, -104, 7, 6, dict(roof='planks', flat=True, door=('py', 0.4))),
             (-44, -124, 7, 6, dict(roof='spruce_planks', ridge='x', door=('py', 0.5), chimney=(-40, -121))),
             (12, -122, 6, 7, dict(roof='planks', ridge='y', door=('nx', 0.5)))]
        for (x0, y0, w, d, kw) in H:
            self.house(x0, y0, w, d, **kw)
        # the church: stone, a dark roof and a tower with the bell
        z = self.house(-30, -127, 10, 7, wall='stone_bricks', frame='cobblestone', roof='dark_planks', ridge='x',
                       wall_h=5, door=('px', 0.5), windows='glass')
        tx, ty = -24, -126
        self.fill(tx, tx + 4, ty, ty + 4, z, z + 12, 'stone_bricks')
        self.fill(tx, tx + 4, ty, ty + 4, z + 12, z + 13, 'cobblestone')
        for (mx, my) in ((tx, ty), (tx + 3, ty), (tx, ty + 3), (tx + 3, ty + 3)):
            self.fill(mx, mx + 1, my, my + 1, z + 13, z + 14, 'cobblestone')
        self.details.box((tx + 1.6, ty + 1.6, z + 13), (tx + 2.4, ty + 2.4, z + 13.8), 'bell')
        self.mark_clear(tx, tx + 4, ty, ty + 4, pad=3)
        # the blacksmith: cobblestone, a flat roof, a lava pool out front
        z = self.house(-4, -128, 8, 7, wall='cobblestone', frame='log', roof='stone_bricks', flat=True,
                       door=('nx', 0.5), base='stone_bricks', windows=None)
        self.set_top(-9, -5, -127, -122, 'cobblestone', z)
        self.set_top(-8, -6, -126, -123, 'lava')
        self.lights.append((-7, -124.5, z + 0.5, 6.0))
        # farms and pens
        for (x0, y0, w, d, crops) in ((-68, -80, 14, 10, ('wheat', 'wheat')), (-68, -110, 14, 10, ('carrots', 'wheat')),
                                      (12, -74, 12, 9, ('wheat', 'carrots')), (-2, -150, 12, 10, ('wheat', 'wheat'))):
            self.farm(x0, y0, w, d, crops)
            self.torch(x0 - 1, y0 - 1)
        for (x, y) in ((-70, -84), (-69, -84), (-69, -85), (-53, -114), (25, -77)):
            z = self.zg(x, y)
            self.fill(x, x + 1, y, y + 1, z, z + 1, 'hay')
        self.fill(-69, -68, -84, -83, self.zg(-69, -84) + 1, self.zg(-69, -84) + 2, 'hay')
        self.fence(-44, -72, 9, 7, gap=(-40, -72))
        for (x, y, a) in ((-41.5, -69, 0.4), (-38.3, -68.2, 2.2), (-40, -66.7, -1.0)):
            self.mob('cow', x, y, a)
        self.fence(22, -114, 7, 7, gap=(22, -111))
        for (x, y, a) in ((24.5, -112, 1.2), (26.8, -110, -0.4)):
            self.mob('sheep', x, y, a)
        self.mob('pig', 25, -108.6, 2.8)
        # lamp posts along the roads
        for y in range(-72, -140, -12):
            self.torch(-14 if (y // 12) % 2 else -10, y, post=2.0)
        for x in range(-70, 26, 12):
            if abs(x - cx) > 7:
                self.torch(x, -91 if (x // 12) % 2 else -87, post=2.0)
        # the bridge over the river
        zb = 0
        self.set_top(24, 27, -90, -87, 'planks', -1)
        self.set_top(45, 48, -90, -87, 'planks', -1)
        self.fill(27, 45, -91, -86, zb - 1, zb, 'planks')
        self.fill(27, 45, -91, -90, zb - 1, zb, 'log_x')
        self.fill(27, 45, -87, -86, zb - 1, zb, 'log_x')
        for x in range(27, 45):
            for y in (-91, -87):
                D.box((x + 0.375, y + 0.375, zb), (x + 0.625, y + 0.625, zb + 1.0), 'fence')
                if x < 44:
                    D.box((x + 0.625, y + 0.44, zb + 0.7), (x + 1.375, y + 0.56, zb + 0.88), 'fence')
        for x in (30, 36, 42):
            self.fill(x, x + 1, -91, -90, -4, zb - 1, 'log')
            self.fill(x, x + 1, -87, -86, -4, zb - 1, 'log')
        # people and the golem
        for (x, y, a) in ((-9.5, -92.3, 0.5), (-15.2, -84.0, 2.8), (-11.0, -101.5, -0.3), (-12.6, -118.0, 3.1),
                          (-30.0, -88.5, 1.6), (4.8, -89.2, -1.7), (-41.0, -87.6, 1.3), (-6.0, -112.0, 2.2),
                          (-60.0, -76.4, 0.2), (17.0, -71.3, 2.6)):
            self.mob('villager', x, y, a)
        self.mob('golem', -19.5, -95.5, 0.7)
        self.mob('cat', -26.5, -82.6, 1.1)
        self.mob('cat', 2.4, -99.0, -2.0)
        self.flowers(cx, cy, 52, 0.03, 61)

    # -- meshing ---------------------------------------------------------------------------------
    def mesh(self):
        """Visible faces of the grid, the water and the details, sorted into chunks."""
        G = self.G
        solid = G > 0
        quads, keys = [], []
        for d in range(6):
            ax, sg = (0, 1) if d == 0 else (0, -1) if d == 1 else (1, 1) if d == 2 else (1, -1) if d == 3 else \
                (2, 1) if d == 4 else (2, -1)
            nb = np.ones_like(solid)
            if ax == 2 and sg > 0:
                nb[..., -1] = False
            src = [slice(None)] * 3
            dst = [slice(None)] * 3
            if sg > 0:
                dst[ax], src[ax] = slice(0, -1), slice(1, None)
            else:
                dst[ax], src[ax] = slice(1, None), slice(0, -1)
            nb[tuple(dst)] = solid[tuple(src)]
            i, j, k = np.nonzero(solid & ~nb)
            ids = G[i, j, k]
            lo = np.stack([i - HALF, j - HALF, k + ZMIN], -1).astype(np.float32)
            hi = lo + 1.0
            jit = 1.0 + (_hash(i, j, k * 7 + d) - 0.5) * 0.07          # block by block, a little variation
            tint = np.repeat(jit[:, None], 3, 1)
            q = face_quads(lo, hi, d, FACE_LAYER[ids, d], tint)
            quads.append(q)
            keys.append((i // CHUNK) * 1000 + (j // CHUNK))
        # the water's surface
        if self.waters:
            w = np.array(self.waters, np.float64)
            wi, wj, wz = w[:, 0].astype(int), w[:, 1].astype(int), w[:, 2]
            lo = np.stack([wi - HALF, wj - HALF, wz - 1.0], -1).astype(np.float32)
            hi = np.stack([wi - HALF + 1, wj - HALF + 1, wz], -1).astype(np.float32)
            quads.append(face_quads(lo, hi, 4, TL['water'], np.ones((len(w), 3))))
            keys.append((wi // CHUNK) * 1000 + (wj // CHUNK))
        dq = self.details.quads()
        if len(dq):
            c = dq[:, :, 0:2].mean(1)
            di = np.clip((c[:, 0] + HALF).astype(int), 0, self.n - 1)
            dj = np.clip((c[:, 1] + HALF).astype(int), 0, self.n - 1)
            quads.append(dq)
            keys.append((di // CHUNK) * 1000 + (dj // CHUNK))
        Q = np.concatenate(quads)
        K = np.concatenate(keys)
        order = np.argsort(K, kind='stable')
        Q, K = Q[order], K[order]
        V = Q.reshape(-1, 12)
        nq = len(Q)
        base = (np.arange(nq, dtype=np.uint32) * 4)[:, None]
        I = (base + np.array([0, 1, 2, 0, 2, 3], np.uint32)[None]).reshape(-1)
        chunks = []
        uk, start, cnt = np.unique(K, return_index=True, return_counts=True)
        for s, c in zip(start, cnt):
            p = Q[s:s + c, :, 0:3].reshape(-1, 3)
            chunks.append((int(s * 6), int(c * 6), p.min(0), p.max(0)))
        return V, I, chunks


def build(fields):
    """The whole spawn area: (World, vertices, indices, chunks)."""
    import time
    t = time.time()
    W = World(fields)
    W.terrain()
    W.camp()
    W.village()
    W.fill_ground()
    W.apply_ops()
    nt = W.trees()
    V, I, chunks = W.mesh()
    print('[spawn] %d trees, %d quads, %d chunks in %.1fs' % (nt, len(I) // 6, len(chunks), time.time() - t),
          flush=True)
    return W, V, I, chunks
