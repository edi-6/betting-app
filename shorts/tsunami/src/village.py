"""The village on the bay, block by block: oak houses with log corners and spruce gable roofs, a cobblestone smithy,
a library, a stone church with a bell tower at the end of the main street, a well in the square, lamp posts, farms
with wheat and hay, a fisherman's hut and a wooden dock out into the sea with a boat tied to it.

Everything here can be knocked down by the water, so it is kept apart from the terrain: a dense grid K[z, y, x] of
block kinds (blocks.KIND_ID, 0 = air) whose cell (0, 0, 0) has its lower corner at ORIGIN. The ground under it is
the terrain's meadow (top at z = 2); `paint` re-textures the meadow's top (streets, the square, farmland, floors).
"""
import numpy as np

import blocks as BL
import world as WD

K_ = BL.KIND_ID
ORIGIN = (-96, -124, -8)
NX, NY, NZ = 192, 168, 44            # x in [-96, 96), y in [-124, 44), z in [-8, 36)
G = WD.PLAIN                         # first block above the meadow sits at z = G


class Village:
    def __init__(self, H):
        self.H = H
        self.K = np.zeros((NZ, NY, NX), np.uint8)
        self.tint = np.ones((NZ, NY, NX, 3), np.float32)
        self.paint = {}                  # (x, y) -> layer for the meadow's top
        self.villagers = []              # (x, y, z, facing, look)
        self.golem = None
        self.bell = None
        self.rng = np.random.default_rng(3)

    # -- primitives ------------------------------------------------------------------------------------------------
    def ground(self, x, y):
        return int(self.H[int(np.floor(y)) - WD.Y0, int(np.floor(x)) - WD.X0])

    def set(self, x, y, z, kind):
        i, j, k = x - ORIGIN[0], y - ORIGIN[1], z - ORIGIN[2]
        if 0 <= i < NX and 0 <= j < NY and 0 <= k < NZ:
            self.K[k, j, i] = 0 if kind is None else K_[kind]

    def get(self, x, y, z):
        i, j, k = x - ORIGIN[0], y - ORIGIN[1], z - ORIGIN[2]
        if 0 <= i < NX and 0 <= j < NY and 0 <= k < NZ:
            return self.K[k, j, i]
        return 0

    def fill(self, x0, y0, z0, x1, y1, z1, kind):
        """Inclusive box."""
        for x in range(min(x0, x1), max(x0, x1) + 1):
            for y in range(min(y0, y1), max(y0, y1) + 1):
                for z in range(min(z0, z1), max(z0, z1) + 1):
                    self.set(x, y, z, kind)

    def paint_rect(self, x0, y0, x1, y1, layer):
        for x in range(min(x0, x1), max(x0, x1) + 1):
            for y in range(min(y0, y1), max(y0, y1) + 1):
                self.paint[(x, y)] = BL.L[layer]

    # -- buildings -------------------------------------------------------------------------------------------------
    def walls(self, x0, y0, x1, y1, z0, z1, kind, corner=None):
        for z in range(z0, z1 + 1):
            for x in range(x0, x1 + 1):
                self.set(x, y0, z, kind)
                self.set(x, y1, z, kind)
            for y in range(y0, y1 + 1):
                self.set(x0, y, z, kind)
                self.set(x1, y, z, kind)
            if corner:
                for (x, y) in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
                    self.set(x, y, z, corner)

    def windows(self, x0, y0, x1, y1, z, every=3, height=1, skip=()):
        """Glass along the middle of each wall."""
        for side in ('n', 's', 'e', 'w'):
            if side in skip:
                continue
            if side in ('n', 's'):
                y = y1 if side == 'n' else y0
                for x in self._spots(x0, x1, every):
                    for dz in range(height):
                        self.set(x, y, z + dz, 'glass')
            else:
                x = x1 if side == 'e' else x0
                for y in self._spots(y0, y1, every):
                    for dz in range(height):
                        self.set(x, y, z + dz, 'glass')

    @staticmethod
    def _spots(a, b, every):
        n = b - a - 1                      # inner length
        if n <= 2:
            return [(a + b) // 2]
        k = max(1, (n + 1) // (every + 1))
        return [a + int(round((i + 1) * (b - a) / (k + 1))) for i in range(k)]

    def door(self, x, y, z):
        self.set(x, y, z, None)
        self.set(x, y, z + 1, None)

    def gable_roof(self, x0, y0, x1, y1, z, kind='spruce', gable='planks', along='x'):
        """Stepped gable roof over the box, ridge along `along`, one block of overhang."""
        if along == 'x':
            k = 0
            while True:
                ya, yb = y0 - 1 + k, y1 + 1 - k
                if ya > yb:
                    break
                for x in range(x0 - 1, x1 + 2):
                    self.set(x, ya, z + k, kind)
                    self.set(x, yb, z + k, kind)
                for y in range(ya + 1, yb):
                    self.set(x0, y, z + k, gable)
                    self.set(x1, y, z + k, gable)
                k += 1
        else:
            k = 0
            while True:
                xa, xb = x0 - 1 + k, x1 + 1 - k
                if xa > xb:
                    break
                for y in range(y0 - 1, y1 + 2):
                    self.set(xa, y, z + k, kind)
                    self.set(xb, y, z + k, kind)
                for x in range(xa + 1, xb):
                    self.set(x, y0, z + k, gable)
                    self.set(x, y1, z + k, gable)
                k += 1

    def pyramid_roof(self, x0, y0, x1, y1, z, kind):
        k = 0
        while x0 - 1 + k <= x1 + 1 - k and y0 - 1 + k <= y1 + 1 - k:
            xa, xb, ya, yb = x0 - 1 + k, x1 + 1 - k, y0 - 1 + k, y1 + 1 - k
            for x in range(xa, xb + 1):
                self.set(x, ya, z + k, kind)
                self.set(x, yb, z + k, kind)
            for y in range(ya, yb + 1):
                self.set(xa, y, z + k, kind)
                self.set(xb, y, z + k, kind)
            k += 1

    def house(self, x0, y0, x1, y1, h=3, wall='planks', corner='log', roof='spruce', base='cobble', door=('n', None),
              along=None, floor='planks', lamp=True):
        """A plains house: a cobblestone course, walls with log corners, windows, a door, a gable roof."""
        z0 = G
        self.walls(x0, y0, x1, y1, z0, z0, base)
        self.walls(x0, y0, x1, y1, z0 + 1, z0 + h, wall, corner)
        self.windows(x0, y0, x1, y1, z0 + 2)
        side, pos = door
        if side == 'n':
            self.door(pos if pos is not None else (x0 + x1) // 2, y1, z0)
        elif side == 's':
            self.door(pos if pos is not None else (x0 + x1) // 2, y0, z0)
        elif side == 'e':
            self.door(x1, pos if pos is not None else (y0 + y1) // 2, z0)
        else:
            self.door(x0, pos if pos is not None else (y0 + y1) // 2, z0)
        self.paint_rect(x0 + 1, y0 + 1, x1 - 1, y1 - 1, 'planks' if floor == 'planks' else 'cobble_floor')
        if along is None:
            along = 'x' if (x1 - x0) >= (y1 - y0) else 'y'
        self.gable_roof(x0, y0, x1, y1, z0 + h + 1, roof, wall, along)
        if lamp:
            # a lamp by the door
            if side in ('n', 's'):
                dx = (pos if pos is not None else (x0 + x1) // 2) + 1
                dy = y1 + 1 if side == 'n' else y0 - 1
                self.set(dx, dy, z0 + 2, 'glowstone')

    def lamp_post(self, x, y, h=3):
        for z in range(G, G + h):
            self.set(x, y, z, 'log')
        self.set(x, y, G + h, 'glowstone')

    def tree(self, x, y):
        blocks = WD.tree_blocks(self.rng, False)
        tint = WD.LEAF_TINTS[self.rng.integers(len(WD.LEAF_TINTS))]
        z0 = self.ground(x + 0.5, y + 0.5)
        for (dx, dy, dz, kind) in blocks:
            if kind == 'log':
                self.set(x + dx, y + dy, z0 + dz, 'log')
            else:
                if self.get(x + dx, y + dy, z0 + dz) == 0:
                    self.set(x + dx, y + dy, z0 + dz, 'leaves')
                    i, j, k = x + dx - ORIGIN[0], y + dy - ORIGIN[1], z0 + dz - ORIGIN[2]
                    if 0 <= i < NX and 0 <= j < NY and 0 <= k < NZ:
                        self.tint[k, j, i] = tint


def keep_clear(x, y):
    """True where the terrain's own trees mustn't grow: the village and its beach."""
    return (abs(x) < 100 and -128 < y < 30) or (abs(x) < 130 and y > -20)


def build(H):
    v = Village(H)
    g = G
    # ---- streets, the square, the beach path ---------------------------------------------------------------------
    v.paint_rect(-74, -28, 74, -26, 'path_top')                  # main street, along the shore
    v.paint_rect(-2, -64, 0, -6, 'path_top')                     # up from the beach to the church
    v.paint_rect(-8, -44, 6, -33, 'path_top')                    # the square
    v.paint_rect(-60, -52, -58, -28, 'path_top')
    v.paint_rect(56, -50, 58, -28, 'path_top')
    v.paint_rect(-40, -68, -38, -28, 'path_top')
    v.paint_rect(28, -70, 30, -28, 'path_top')
    # ---- front row, facing the sea --------------------------------------------------------------------------------
    v.house(-38, -21, -31, -15, h=3, door=('s', None))
    v.house(-14, -20, -8, -14, h=3, door=('s', -11))
    v.house(10, -21, 17, -15, h=4, wall='planks', roof='spruce', door=('s', None))
    v.house(33, -23, 42, -15, h=4, wall='planks', corner='spruce_log', base='cobble', door=('s', 37))
    v.house(56, -20, 62, -14, h=3, door=('s', None))
    v.house(-62, -21, -55, -15, h=3, door=('s', None))
    # ---- the fisherman's hut on the beach and the dock -------------------------------------------------------------
    hx0, hy0, hx1, hy1 = -26, -6, -21, -2
    zb = 1
    for x in range(hx0, hx1 + 1):
        for y in range(hy0, hy1 + 1):
            if x in (hx0, hx1) or y in (hy0, hy1):
                for z in range(zb, zb + 3):
                    v.set(x, y, z, 'planks')
    for (x, y) in ((hx0, hy0), (hx1, hy0), (hx0, hy1), (hx1, hy1)):
        for z in range(zb, zb + 3):
            v.set(x, y, z, 'spruce_log')
    v.door((hx0 + hx1) // 2, hy0, zb)
    v.set(hx0, (hy0 + hy1) // 2, zb + 1, 'glass')
    v.set(hx1, (hy0 + hy1) // 2, zb + 1, 'glass')
    v.gable_roof(hx0, hy0, hx1, hy1, zb + 3, 'spruce', 'planks', 'x')
    for z in range(zb, zb + 2):
        v.set(hx1 + 2, hy0 + 1, z, 'hay')
    v.set(hx1 + 2, hy0 + 2, zb, 'hay')
    # the dock: a deck on log piles out to deep water
    for y in range(-2, 34):
        for x in (6, 7, 8):
            v.set(x, y, 1, 'planks')
        if y % 4 == 0:
            for x in (6, 8):
                floor = v.ground(x + 0.5, y + 0.5)
                for z in range(floor, 1):
                    v.set(x, y, z, 'log')
    for y in range(30, 34):
        for x in (4, 5, 9, 10):
            v.set(x, y, 1, 'planks')
    v.set(6, 33, 2, 'log')
    v.set(6, 33, 3, 'glowstone')
    v.set(8, 33, 2, 'log')
    v.set(8, 33, 3, 'glowstone')
    # a boat tied up beside it (sits in the water)
    bx0, by0 = 11, 16
    for y in range(by0, by0 + 6):
        w = 1 if y in (by0, by0 + 5) else 2
        for x in range(bx0 - w + 1, bx0 + w + 1):
            v.set(x, y, 0, 'spruce')
        if y not in (by0, by0 + 5):
            v.set(bx0 - 1, y, 1, 'spruce')
            v.set(bx0 + 2, y, 1, 'spruce')
    # ---- the square: the well and the bell's lamp posts --------------------------------------------------------------
    wx0, wy0 = -3, -40
    for x in range(wx0, wx0 + 4):
        for y in range(wy0, wy0 + 4):
            edge = x in (wx0, wx0 + 3) or y in (wy0, wy0 + 3)
            v.set(x, y, g, 'cobble' if edge else None)
            if edge:
                v.set(x, y, g + 1, 'cobble')
    v.paint_rect(wx0 + 1, wy0 + 1, wx0 + 2, wy0 + 2, 'water')
    for (x, y) in ((wx0, wy0), (wx0 + 3, wy0), (wx0, wy0 + 3), (wx0 + 3, wy0 + 3)):
        v.set(x, y, g + 2, 'log')
        v.set(x, y, g + 3, 'log')
    v.fill(wx0, wy0, g + 4, wx0 + 3, wy0 + 3, g + 4, 'spruce')
    v.set(wx0 + 1, wy0 + 1, g + 5, 'spruce')
    v.set(wx0 + 2, wy0 + 2, g + 5, 'spruce')
    v.set(wx0 + 1, wy0 + 2, g + 5, 'spruce')
    v.set(wx0 + 2, wy0 + 1, g + 5, 'spruce')
    for (x, y) in ((-9, -29), (7, -29), (-9, -45), (7, -45), (-3, -9), (1, -9), (-41, -29), (31, -29), (-61, -29),
                   (59, -29), (-3, -63), (1, -63)):
        v.lamp_post(x, y, 3)
    # ---- second row ---------------------------------------------------------------------------------------------------
    # the smithy: cobblestone, a flat roof
    sx0, sy0, sx1, sy1 = -36, -50, -27, -38
    v.walls(sx0, sy0, sx1, sy1, g, g + 4, 'cobble', 'log')
    v.windows(sx0, sy0, sx1, sy1, g + 2, every=3)
    v.door((sx0 + sx1) // 2, sy1, g)
    v.door((sx0 + sx1) // 2 + 1, sy1, g)
    v.fill(sx0 - 1, sy0 - 1, g + 5, sx1 + 1, sy1 + 1, g + 5, 'cobble')
    v.fill(sx0, sy0, g + 6, sx1, sy0, g + 6, 'cobble')
    v.paint_rect(sx0 + 1, sy0 + 1, sx1 - 1, sy1 - 1, 'cobble_floor')
    # the library: two storeys of oak with bookshelves inside, a big roof
    lx0, ly0, lx1, ly1 = 12, -55, 22, -43
    v.walls(lx0, ly0, lx1, ly1, g, g, 'cobble')
    v.walls(lx0, ly0, lx1, ly1, g + 1, g + 7, 'planks', 'log')
    v.windows(lx0, ly0, lx1, ly1, g + 2, every=2, height=2)
    v.windows(lx0, ly0, lx1, ly1, g + 5, every=2, height=1)
    for y in range(ly0 + 1, ly1):
        for z in range(g + 1, g + 4):
            v.set(lx0 + 1, y, z, 'bookshelf')
            v.set(lx1 - 1, y, z, 'bookshelf')
    v.door((lx0 + lx1) // 2, ly1, g)
    v.paint_rect(lx0 + 1, ly0 + 1, lx1 - 1, ly1 - 1, 'planks')
    v.gable_roof(lx0, ly0, lx1, ly1, g + 8, 'spruce', 'planks', 'y')
    # more houses
    v.house(36, -49, 42, -42, h=3, door=('n', None))
    v.house(-22, -48, -15, -41, h=3, door=('n', None))
    v.house(-62, -50, -55, -42, h=3, door=('e', None))
    v.house(52, -48, 60, -41, h=4, door=('w', None))
    v.house(24, -82, 31, -74, h=3, door=('n', None))
    v.house(-31, -82, -24, -74, h=3, door=('n', None))
    v.house(44, -70, 50, -63, h=3, door=('w', None))
    v.house(-50, -70, -44, -63, h=3, door=('e', None))
    # ---- the church: stone bricks, tall windows, a bell tower with the bell at the top ----------------------------------
    cx0, cy0, cx1, cy1 = -5, -84, 3, -68
    v.walls(cx0, cy0, cx1, cy1, g, g, 'cobble')
    v.walls(cx0, cy0, cx1, cy1, g + 1, g + 7, 'bricks', 'cobble')
    for y in range(cy0 + 2, cy1 - 1, 3):
        for z in range(g + 2, g + 6):
            v.set(cx0, y, z, 'glass')
            v.set(cx1, y, z, 'glass')
    v.door(cx0 + 4, cy1, g)
    v.set(cx0 + 4, cy1, g + 2, None)
    for z in range(g + 4, g + 7):
        v.set(cx0 + 4, cy1, z, 'glass')
    v.paint_rect(cx0 + 1, cy0 + 1, cx1 - 1, cy1 - 1, 'cobble_floor')
    v.gable_roof(cx0, cy0 + 5, cx1, cy1, g + 8, 'spruce', 'bricks', 'y')
    # tower over the back end
    tx0, ty0, tx1, ty1 = cx0 + 1, cy0, cx1 - 1, cy0 + 4
    v.walls(tx0, ty0, tx1, ty1, g + 1, g + 15, 'bricks', 'mossy_bricks')
    for z in range(g + 12, g + 15):                              # the belfry: open arches
        for (x, y) in (((tx0 + tx1) // 2, ty0), ((tx0 + tx1) // 2, ty1)):
            v.set(x, y, z, None)
        for (x, y) in ((tx0, (ty0 + ty1) // 2), (tx1, (ty0 + ty1) // 2)):
            v.set(x, y, z, None)
    for x in range(tx0 + 1, tx1):
        for y in range(ty0 + 1, ty1):
            v.set(x, y, g + 11, 'bricks')
    v.set((tx0 + tx1) // 2, (ty0 + ty1) // 2, g + 13, 'gold')     # the bell
    v.bell = ((tx0 + tx1) // 2 + 0.5, (ty0 + ty1) // 2 + 0.5, g + 13.5)
    v.pyramid_roof(tx0, ty0, tx1, ty1, g + 16, 'spruce')
    v.set((tx0 + tx1) // 2, (ty0 + ty1) // 2, g + 19, 'glowstone')
    # ---- farms ----------------------------------------------------------------------------------------------------------
    for (fx0, fy0, fx1, fy1) in ((-80, -70, -66, -58), (64, -66, 80, -54), (-82, -100, -66, -88), (66, -98, 82, -86)):
        v.walls(fx0, fy0, fx1, fy1, g, g, 'log')
        v.paint_rect(fx0 + 1, fy0 + 1, fx1 - 1, fy1 - 1, 'farmland')
        for x in range(fx0 + 1, fx1):
            for y in range(fy0 + 1, fy1):
                if (y - fy0) % 4 == 0:
                    v.paint[(x, y)] = BL.L['water']
                else:
                    v.set(x, y, g, 'wheat')
    for (x, y, n) in ((-64, -60, 3), (-64, -62, 2), (62, -58, 2), (62, -60, 3), (-23, -10, 2), (45, -28, 2)):
        for z in range(n):
            v.set(x, y, g + z, 'hay')
    # ---- trees in the village --------------------------------------------------------------------------------------------
    for (x, y) in ((-48, -33), (48, -34), (-72, -40), (74, -40), (-14, -60), (14, -64), (-46, -88), (44, -90),
                   (-86, -20), (86, -24), (-70, -112), (8, -100), (-18, -104), (30, -108), (60, -110), (-90, -78),
                   (90, -72)):
        v.tree(x, y)
    # ---- who lives here ------------------------------------------------------------------------------------------------
    v.villagers = [(-3.5, -1.0, 1.0, 0.0), (7.0, 20.5, 2.0, 1.6), (12.5, -27.0, 2.0, 3.1), (4.0, -36.5, 2.0, 2.3),
                   (-12.5, -27.5, 2.0, 0.2), (-7.0, -64.0, 2.0, 1.4), (-70.0, -62.0, 2.0, 1.0),
                   (24.0, -40.0, 2.0, 0.5)]
    v.golem = (-7.0, -37.0, 2.0, 0.6)
    return v


if __name__ == '__main__':
    import time
    H = WD.make_heightmap()
    t = time.time()
    v = build(H)
    n = int((v.K > 0).sum())
    print('village blocks', n, f'{time.time() - t:.1f}s')
    t = time.time()
    vv, ii = WD.mesh_blocks(v.K, ORIGIN, tint=v.tint)
    print('mesh verts', len(vv), 'tris', len(ii) // 3, f'{time.time() - t:.2f}s')
