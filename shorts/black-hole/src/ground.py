"""Destructible ground of the arena: a column height field of 1x1 Minecraft blocks that the black hole peels
away block by block (grass, then dirt, then stone), down into a deep funnel.

Columns start flush with the grass at z = 0. The region is drawn with the same textured-block shader as the
static terrain (grass/dirt/stone layers); its mesh is built fully vectorised, so it can be rebuilt every frame
while thousands of columns are being eaten.
"""
import numpy as np

import world as W

REG = 56                 # columns cover x, y in [-REG, REG)
DEPTH = 26               # deepest level the black hole can eat down to
SAFE_R = 48.0            # beyond this radius the ground stays (trees stand past it)

L_GRASS_TOP, L_DIRT, L_GRASS_SIDE, L_STONE = W.L['grass_top'], W.L['dirt'], W.L['grass_side'], W.L['stone']
B_GRASS, B_DIRT, B_STONE = 0, 1, 2       # block kinds of peeled blocks


def block_kind(k):
    """Kind of the block spanning z in [k, k + 1] (vectorised)."""
    k = np.asarray(k)
    return np.where(k == -1, B_GRASS, np.where(k >= -4, B_DIRT, B_STONE))


def _side_layer(k):
    k = np.asarray(k)
    return np.where(k == -1, L_GRASS_SIDE, np.where(k >= -4, L_DIRT, L_STONE))


def _top_layer(top):
    top = np.asarray(top)
    return np.where(top == 0, L_GRASS_TOP, np.where(top - 1 >= -4, L_DIRT, L_STONE))


def block_colours(k):
    """Flat colours (side, top) for debris spawned from the block at level k."""
    if k == -1:
        return (136, 96, 64), (98, 170, 58)
    if k >= -4:
        return (134, 95, 63), (122, 86, 57)
    return (126, 126, 128), (112, 112, 114)


# corner offsets (x, y, z in units of the face's box) and uv rules per face, as in world.box_face
_FACES = {
    'pz': dict(n=(0, 0, 1)),
    'px': dict(n=(1, 0, 0)),
    'nx': dict(n=(-1, 0, 0)),
    'py': dict(n=(0, 1, 0)),
    'ny': dict(n=(0, -1, 0)),
}


def _quads(face, x0, y0, z0, x1, y1, z1, layer):
    """Vectorised world.box_face: (F, 4, 12) vertices for F faces."""
    F = len(x0)
    if face == 'pz':
        P = [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        UV = [(x0, -y0), (x1, -y0), (x1, -y1), (x0, -y1)]
    elif face == 'px':
        P = [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)]
        UV = [(y0, -z0), (y1, -z0), (y1, -z1), (y0, -z1)]
    elif face == 'nx':
        P = [(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)]
        UV = [(-y1, -z0), (-y0, -z0), (-y0, -z1), (-y1, -z1)]
    elif face == 'py':
        P = [(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)]
        UV = [(-x1, -z0), (-x0, -z0), (-x0, -z1), (-x1, -z1)]
    else:  # 'ny'
        P = [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)]
        UV = [(x0, -z0), (x1, -z0), (x1, -z1), (x0, -z1)]
    v = np.zeros((F, 4, 12), np.float32)
    n = _FACES[face]['n']
    for c in range(4):
        v[:, c, 0], v[:, c, 1], v[:, c, 2] = P[c]
        v[:, c, 3:6] = n
        v[:, c, 6], v[:, c, 7] = UV[c]
        v[:, c, 8] = layer
        v[:, c, 9:12] = 1.0
    return v


class Ground:
    def __init__(self, feet=()):
        """feet: (x0, x1, y0, y1) boxes where the ground is kept while a giant stands on it."""
        n = 2 * REG
        self.n = n
        self.top = np.zeros((n, n), np.int32)
        c = np.arange(n) - REG + 0.5
        X, Y = np.meshgrid(c, c, indexing='ij')
        self.cx, self.cy = X, Y
        self.destructible = np.hypot(X, Y) < SAFE_R
        self.feet = {}
        for key, box in enumerate(feet):
            self.keep(key, box)
        self.dirty = True
        self.version = 0

    def keep(self, key, box):
        x0, x1, y0, y1 = box
        m = (self.cx > x0) & (self.cx < x1) & (self.cy > y0) & (self.cy < y1)
        self.feet[key] = m
        self.destructible &= ~m

    def release(self, key):
        """A giant has gone: the ground under its feet can be eaten too."""
        m = self.feet.pop(key, None)
        if m is not None:
            self.destructible |= m & (np.hypot(self.cx, self.cy) < SAFE_R)

    @staticmethod
    def col_index(x, y):
        return np.floor(x).astype(np.int64) + REG, np.floor(y).astype(np.int64) + REG

    def surface(self, x, y):
        """Top height at world (x, y); 0 outside the region."""
        ix, iy = self.col_index(np.asarray(x), np.asarray(y))
        inb = (ix >= 0) & (ix < 2 * REG) & (iy >= 0) & (iy < 2 * REG)
        out = np.zeros(np.shape(ix), np.float64)
        out[inb] = self.top[ix[inb], iy[inb]]
        return out

    def peel(self, want):
        """Lower columns towards the target tops `want` (n x n ints, deeper = smaller) by at most one block each.
        Returns the removed blocks as (x, y, z, kind) arrays (block centres)."""
        lower = self.destructible & (want < self.top) & (self.top > -DEPTH)
        if not lower.any():
            return np.zeros((0, 3)), np.zeros(0, np.int64)
        ii, jj = np.nonzero(lower)
        k = self.top[ii, jj] - 1                     # the block that goes: z in [k, k + 1]
        self.top[ii, jj] = k
        self.dirty = True
        self.version += 1
        pos = np.stack([ii - REG + 0.5, jj - REG + 0.5, k + 0.5], -1).astype(np.float64)
        return pos, block_kind(k)

    def mesh(self):
        """Vertices/indices (same layout as the static terrain) for the whole region, vectorised."""
        n = self.n
        top = self.top
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        x0 = (ii - REG).ravel().astype(np.float32)
        y0 = (jj - REG).ravel().astype(np.float32)
        t = top.ravel().astype(np.float32)
        parts = [_quads('pz', x0, y0, t - 1, x0 + 1, y0 + 1, t, _top_layer(top.ravel()))]
        # pit walls: faces where a neighbour column is lower
        for face, (dx, dy) in (('px', (1, 0)), ('nx', (-1, 0)), ('py', (0, 1)), ('ny', (0, -1))):
            nb = np.zeros_like(top)
            sx = slice(max(0, -dx), n - max(0, dx))
            sy = slice(max(0, -dy), n - max(0, dy))
            tx = slice(max(0, dx), n - max(0, -dx))
            ty = slice(max(0, dy), n - max(0, -dy))
            nb[sx, sy] = top[tx, ty]
            diff = np.maximum(top - nb, 0).ravel()
            if diff.sum() == 0:
                continue
            cols = np.repeat(np.arange(n * n), diff)
            start = np.repeat(np.cumsum(diff) - diff, diff)
            k = nb.ravel()[cols] + (np.arange(len(cols)) - start)
            xa = x0[cols]
            ya = y0[cols]
            kf = k.astype(np.float32)
            parts.append(_quads(face, xa, ya, kf, xa + 1, ya + 1, kf + 1, _side_layer(k)))
        v = np.concatenate(parts).reshape(-1, 12)
        F = len(v) // 4
        base = (np.arange(F, dtype=np.uint32) * 4)[:, None]
        idx = (base + np.array([0, 1, 2, 0, 2, 3], np.uint32)[None, :]).ravel()
        self.dirty = False
        return v, idx
