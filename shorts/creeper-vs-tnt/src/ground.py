"""Destructible ground around the creeper: a column height field of 1x1 Minecraft blocks.

Columns start flush with the grass at z = 0. Explosions lower them (spherical craters, no overhangs).
The region is drawn with the same textured-block shader as the static terrain (grass/dirt/stone layers).
"""
import numpy as np

import world as W

REG = 28                 # columns cover x, y in [-REG, REG)
DEPTH = 7                # deepest diggable level (the block below stays as bedrock-like floor)
SAFE_R = 25.5            # beyond this radius the ground is indestructible (trees stand past it)
FOOT = (-5.0, 5.0, -7.0, 7.0)   # under the creeper's feet the ground is kept (no floating legs)

L_GRASS_TOP, L_DIRT, L_GRASS_SIDE, L_STONE = W.L['grass_top'], W.L['dirt'], W.L['grass_side'], W.L['stone']


def block_layer(k):
    """Texture layer for the side of the block spanning z in [k, k + 1]."""
    if k == -1:
        return L_GRASS_SIDE
    if k >= -4:
        return L_DIRT
    return L_STONE


def top_layer(top):
    if top == 0:
        return L_GRASS_TOP
    return L_DIRT if top - 1 >= -4 else L_STONE


def block_colours(k):
    """Flat colours (side, top) for debris spawned from the block at level k."""
    if k == -1:
        return (136, 96, 64), (98, 170, 58)
    if k >= -4:
        return (134, 95, 63), (122, 86, 57)
    return (126, 126, 128), (112, 112, 114)


class Ground:
    def __init__(self):
        n = 2 * REG
        self.top = np.zeros((n, n), np.int32)
        c = np.arange(n) - REG + 0.5
        X, Y = np.meshgrid(c, c, indexing='ij')
        self.cx, self.cy = X, Y
        foot = (X > FOOT[0]) & (X < FOOT[1]) & (Y > FOOT[2]) & (Y < FOOT[3])
        self.destructible = (np.hypot(X, Y) < SAFE_R) & ~foot
        self.dirty = True
        self.version = 0

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

    def blast(self, centers, radius):
        """Carve spherical craters. Returns list of (ix, iy, old_top, new_top) for changed columns."""
        changes = []
        r = int(np.ceil(radius)) + 1
        for c in centers:
            if c[2] > radius + 0.3:
                continue
            ix0, iy0 = int(np.floor(c[0])) + REG, int(np.floor(c[1])) + REG
            xs = np.arange(max(0, ix0 - r), min(2 * REG, ix0 + r + 1))
            ys = np.arange(max(0, iy0 - r), min(2 * REG, iy0 + r + 1))
            if len(xs) == 0 or len(ys) == 0:
                continue
            XI, YI = np.meshgrid(xs, ys, indexing='ij')
            d2 = (self.cx[XI, YI] - c[0]) ** 2 + (self.cy[XI, YI] - c[1]) ** 2
            inside = (d2 < radius * radius) & self.destructible[XI, YI]
            if not inside.any():
                continue
            zb = c[2] - np.sqrt(np.maximum(radius * radius - d2, 0.0))
            new_top = np.floor(zb + 0.5).astype(np.int32)
            new_top = np.maximum(new_top, -DEPTH)
            old = self.top[XI, YI]
            lower = inside & (new_top < old)
            if not lower.any():
                continue
            for a, b, o, nt in zip(XI[lower], YI[lower], old[lower], new_top[lower]):
                changes.append((int(a), int(b), int(o), int(nt)))
            self.top[XI[lower], YI[lower]] = new_top[lower]
            self.dirty = True
        if changes:
            self.version += 1
        return changes

    def mesh(self):
        """Vertices/indices (same layout as the static terrain) for the whole region."""
        from world import MeshBuilder, box_face
        mb = MeshBuilder()
        n = 2 * REG
        white = (1.0, 1.0, 1.0)
        top = self.top
        for ix in range(n):
            for iy in range(n):
                t = int(top[ix, iy])
                x0, y0 = ix - REG, iy - REG
                box_face(mb, x0, y0, t - 1, x0 + 1, y0 + 1, t, 'pz', top_layer(t), white)
        # crater walls: faces where a neighbour column is lower
        for face, (dx, dy) in (('px', (1, 0)), ('nx', (-1, 0)), ('py', (0, 1)), ('ny', (0, -1))):
            nb = np.zeros_like(top)
            sx = slice(max(0, -dx), n - max(0, dx))
            sy = slice(max(0, -dy), n - max(0, dy))
            tx = slice(max(0, dx), n - max(0, -dx))
            ty = slice(max(0, dy), n - max(0, -dy))
            nb[:] = 0
            nb[sx, sy] = top[tx, ty]
            ii, jj = np.nonzero(nb < top)
            for ix, iy in zip(ii, jj):
                t = int(top[ix, iy])
                x0, y0 = ix - REG, iy - REG
                for k in range(int(nb[ix, iy]), t):
                    box_face(mb, x0, y0, k, x0 + 1, y0 + 1, k + 1, face, block_layer(k), white)
        self.dirty = False
        return mb.arrays()
