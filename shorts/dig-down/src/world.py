"""The world in cross-section: a slice 41 blocks wide from the sky (y = 80) to bedrock (y = -64), with the set pieces
Steve digs past or into, placed along the shaft at x = 0.

Grid rows go down: row r is the block at y = Y_TOP - r (the block occupies [y, y + 1)). `fg` is the cut face (the
blocks), `bg` the wall behind (seen where the face is air: a cave's back wall, the sky), `deco` sprites on cells.
"""
import numpy as np

import art as A

B = A.B
X0, X1 = -45, 45                 # columns
Y_TOP, Y_BOT = 80, -64
NX = X1 - X0 + 1
NY = Y_TOP - Y_BOT + 1
SURFACE = 64                     # the grass block's top
SKY = -1                         # bg value for sky

# where things are (y of the block, x)
CAVE_CEIL, CAVE_FLOOR = 54, 46            # the cave he falls into: air from 47 to 53 at the shaft
DUNGEON = (3, 33)                         # x0, y0 of a little dungeon beside the shaft
MINE_Y = 18                               # mineshaft floor (air at 18..20)
GEODE = (-1.0, -17.5)
CITY_Y = -50                              # ancient city floor (air above from -49 to -44)
DIAMONDS = [(0, -54), (1, -54), (0, -55), (-1, -55), (1, -55)]
LAVA_TOP = -59                            # lava fills y -62..-60; the air pocket above it is y = -59..-58
LAST_FLOOR = -57                          # the last block he stands on is y = -57 (top at -56)


def row(y):
    return Y_TOP - y


def col(x):
    return x - X0


class World:
    def __init__(self, seed=3):
        rng = np.random.default_rng(seed)
        self.rng = rng
        fg = np.full((NY, NX), B['stone'], np.int16)
        bg = np.full((NY, NX), B['stone'], np.int16)
        ys = Y_TOP - np.arange(NY)[:, None] + np.zeros((1, NX))
        xs = X0 + np.arange(NX)[None, :] + np.zeros((NY, 1))
        self.ys, self.xs = ys, xs
        # rock: stone, deepslate below 0 (dithered), blobs of granite, diorite, andesite, tuff, gravel
        from scipy.ndimage import gaussian_filter
        n1 = gaussian_filter(rng.random((NY, NX)), 2.2)
        n2 = gaussian_filter(rng.random((NY, NX)), 2.0)
        n3 = gaussian_filter(rng.random((NY, NX)), 1.6)
        q = lambda n: (n - n.mean()) / n.std()
        n1, n2, n3 = q(n1), q(n2), q(n3)
        deep = (ys < 0) | ((ys < 8) & (rng.random((NY, NX)) < (8 - ys) / 9.0))
        fg[deep] = B['deepslate']
        fg[(n1 > 1.25) & ~deep] = B['granite']
        fg[(n2 > 1.35) & ~deep] = B['diorite']
        fg[(n1 < -1.35) & ~deep] = B['andesite']
        fg[(n3 > 1.5) & (ys > 10)] = B['gravel']
        fg[(n2 > 1.3) & deep] = B['tuff']
        bg[:] = fg
        # ores by depth
        def veins(name, deep_name, ylo, yhi, count, size):
            for _ in range(count):
                y = int(rng.integers(ylo, yhi))
                x = int(rng.integers(X0, X1 + 1))
                for _ in range(size):
                    r, c = row(y) + int(rng.integers(-1, 2)), col(x) + int(rng.integers(-1, 2))
                    if 0 <= r < NY and 0 <= c < NX and fg[r, c] in (B['stone'], B['deepslate'], B['andesite'],
                                                                     B['granite'], B['diorite'], B['tuff']):
                        fg[r, c] = B[deep_name] if fg[r, c] in (B['deepslate'], B['tuff']) and deep_name else B[name]
        veins('coal_ore', 'deep_coal', 5, 62, 40, 6)
        veins('iron_ore', 'deep_iron', -20, 56, 26, 4)
        veins('copper_ore', None, 0, 50, 16, 4)
        veins('gold_ore', 'deep_gold', -60, 20, 12, 3)
        veins('redstone_ore', 'deep_redstone', -60, -10, 14, 4)
        veins('lapis_ore', 'deep_lapis', -50, 30, 8, 3)
        # the surface: grass, dirt, the sky
        for c in range(NX):
            x = X0 + c
            top = SURFACE - 1
            fg[:row(top), c] = B['air']
            bg[:row(top), c] = SKY
            fg[row(top), c] = B['grass']
            for d in range(1, 4 + int(rng.integers(0, 2))):
                fg[row(top - d), c] = B['dirt']
                bg[row(top - d), c] = B['dirt']
            bg[row(top), c] = B['dirt']
        self.surface_top = {X0 + c: SURFACE for c in range(NX)}
        # a tree on the left
        tx = -6
        for y in range(SURFACE, SURFACE + 5):
            fg[row(y), col(tx)] = B['log']
        for y in range(SURFACE + 3, SURFACE + 7):
            for x in range(tx - 2, tx + 3):
                if fg[row(y), col(x)] == B['air'] and not (abs(x - tx) == 2 and y == SURFACE + 6):
                    fg[row(y), col(x)] = B['leaves']
        self.fg, self.bg = fg, bg
        self.deco = {}                    # (x, y) -> sprite name
        self.lights = {}                  # (x, y) -> block light level
        self.water = set()
        # ores on his way down (what's in the shaft's column is what he digs through)
        for (y, x, name) in ((42, 0, 'coal_ore'), (41, -1, 'coal_ore'), (35, 0, 'iron_ore'), (34, 1, 'iron_ore'),
                             (29, 0, 'copper_ore'), (28, -1, 'copper_ore'), (12, 0, 'iron_ore'), (11, 0, 'iron_ore'),
                             (5, 0, 'lapis_ore'), (4, 1, 'lapis_ore'), (-4, 0, 'deep_redstone'),
                             (-5, 0, 'deep_redstone'), (-5, -1, 'deep_redstone'), (-9, 0, 'deep_gold'),
                             (-29, 0, 'deep_iron'), (-30, 0, 'deep_iron'), (-34, 0, 'deep_gold'),
                             (-35, 1, 'deep_gold'), (-38, 0, 'deep_redstone'), (-39, -1, 'deep_redstone')):
            self.fg[row(y), col(x)] = B[name]
        self._cave()
        self._dungeon()
        self._mineshaft()
        self._geode()
        self._city()
        self._diamonds_lava()
        self._bedrock()

    # -- helpers ------------------------------------------------------------------------------------
    def set(self, x, y, name, bg=None):
        r, c = row(y), col(x)
        if 0 <= r < NY and 0 <= c < NX:
            self.fg[r, c] = B[name]
            if bg is not None:
                self.bg[r, c] = B[bg]

    def get(self, x, y):
        r, c = row(y), col(x)
        if 0 <= r < NY and 0 <= c < NX:
            return int(self.fg[r, c])
        return B['bedrock']

    def carve(self, mask, bg=None):
        self.fg[mask] = B['air']
        if bg is not None:
            self.bg[mask] = B[bg]

    def floor_below(self, x, y):
        """The first solid block below y (exclusive); returns its y."""
        yy = y - 1
        while self.get(x, yy) == B['air'] or self.get(x, yy) in (B['water'],):
            yy -= 1
        return yy

    # -- set pieces ----------------------------------------------------------------------------------
    def _cave(self):
        """A winding cave crossing the shaft, tall where he falls into it; a zombie lives here."""
        xs, ys = self.xs, self.ys
        yc = 49.5 + 2.5 * np.sin(xs / 4.3) + 1.2 * np.sin(xs / 1.9 + 1.0)
        hh = 2.2 + 1.6 * np.exp(-(xs / 4.0) ** 2) + 0.6 * np.sin(xs / 3.1)
        m = np.abs(ys + 0.5 - yc) < hh
        m &= (xs > -18)
        self.carve(m)
        # exactly: at the shaft the cave is open from 47 to 53
        for y in range(CAVE_FLOOR + 1, CAVE_CEIL):
            self.set(0, y, 'air')
        self.set(0, CAVE_CEIL, 'stone')
        self.set(0, CAVE_FLOOR, 'stone')
        for x in (-1, 1):
            for y in range(CAVE_FLOOR + 1, CAVE_CEIL):
                self.set(x, y, 'air')
        self.deco[(7, int(np.floor(self._floor_y(7))) + 1)] = 'lichen'
        self.lights[(-9, 49)] = 7
        self.deco[(-9, 49)] = 'lichen'

    def _floor_y(self, x):
        y = CAVE_CEIL + 2
        while self.get(x, y) != B['air'] and y > CAVE_FLOOR - 8:
            y -= 1
        while self.get(x, y - 1) == B['air']:
            y -= 1
        return y - 1

    def _dungeon(self):
        x0, y0 = DUNGEON
        for x in range(x0, x0 + 7):
            for y in range(y0, y0 + 5):
                edge = x in (x0, x0 + 6) or y in (y0, y0 + 4)
                self.set(x, y, 'stone' if not edge else ('andesite' if (x + y) % 3 else 'stone'))
                if not edge:
                    self.set(x, y, 'air', 'stone')
        self.set(x0 + 3, y0 + 1, 'spawner')
        self.lights[(x0 + 3, y0 + 1)] = 3
        self.deco[(x0 + 5, y0 + 1)] = 'chest'

    def _mineshaft(self):
        y0 = MINE_Y
        for x in range(X0, X1 + 1):
            for y in range(y0, y0 + 3):
                self.set(x, y, 'air')
            if x % 2 == 0:
                self.deco[(x, y0)] = 'rail'
        for x in range(-16, 17, 4):
            if x == 0:
                continue
            for y in (y0, y0 + 1):
                self.deco[(x, y)] = 'fence'
            for xx in (x - 1, x, x + 1):
                if xx != 0:
                    self.set(xx, y0 + 2, 'planks')
        for (x, y) in ((-6, y0 + 2), (-11, y0 + 1), (5, y0 + 2), (10, y0 + 2), (13, y0 + 1), (-2, y0 + 2)):
            self.deco[(x, y)] = 'cobweb'
        self.set(-9, y0, 'spawner')
        self.lights[(-9, y0)] = 3
        self.lights[(9, y0 + 1)] = 12
        self.deco[(9, y0 + 1)] = 'torch'

    def _geode(self):
        gx, gy = GEODE
        d = np.hypot((self.xs + 0.5 - gx) * 1.0, (self.ys + 0.5 - gy) * 1.15)
        for r, name in ((7.4, 'basalt'), (6.4, 'calcite'), (5.4, 'amethyst')):
            m = (d < r) & (self.fg != B['air'])
            self.fg[m] = B[name]
            self.bg[m] = B[name]
        hollow = d < 3.6
        self.carve(hollow, 'amethyst')
        rng = self.rng
        inner = (d >= 3.6) & (d < 4.6)
        for (r, c) in zip(*np.nonzero(inner)):
            if rng.random() < 0.3:
                self.fg[r, c] = B['budding']
        for (r, c) in zip(*np.nonzero(hollow)):
            y, x = Y_TOP - r, X0 + c
            if self.get(x, y - 1) != B['air'] and rng.random() < 0.8:
                self.deco[(x, y)] = 'cluster'
                self.lights[(x, y)] = 5

    def _city(self):
        y0 = CITY_Y
        # the hall: a deepslate-tiles floor, brick walls and pillars, open air inside
        for x in range(2, X1 + 1):
            for y in range(y0 + 1, y0 + 7):
                self.set(x, y, 'air', 'deep_tiles')
            self.set(x, y0, 'deep_tiles')
            self.set(x, y0 - 1, 'deep_bricks')
            self.set(x, y0 + 7, 'deep_bricks')
        for y in range(y0, y0 + 8):
            self.set(1, y, 'deep_bricks')
        for x in (7, 13, 19):
            for y in range(y0 + 1, y0 + 7):
                self.set(x, y, 'deep_bricks')
        for x in range(2, X1 + 1):
            if self.rng.random() < 0.55:
                self.set(x, y0, 'sculk')
        for x in (4, 10, 16):
            self.deco[(x, y0 + 1)] = 'sensor'
            self.lights[(x, y0 + 1)] = 2
        self.deco[(11, y0 + 1)] = 'shrieker'
        for x in (5, 9, 15):
            self.deco[(x, y0 + 6)] = 'lantern'
            self.lights[(x, y0 + 6)] = 10
        self.deco[(3, y0 + 1)] = 'candle'
        self.lights[(3, y0 + 1)] = 6
        # sculk creeping out of it along the rock
        for x in range(1, X1 + 1):
            for y in (y0 - 2, y0 + 8):
                if self.rng.random() < 0.45 and self.get(x, y) in (B['deepslate'], B['tuff']):
                    self.set(x, y, 'sculk')
        for (x, y) in ((1, y0 - 1), (1, y0 - 2), (2, y0 - 2), (1, y0 + 8), (1, y0 + 9)):
            self.set(x, y, 'sculk')

    def _diamonds_lava(self):
        for (x, y) in DIAMONDS:
            self.set(x, y, 'deep_diamond', 'deepslate')
        # the lava lake right under where he stops
        xs, ys = self.xs, self.ys
        d = np.hypot((xs + 0.5) / 7.5, (ys + 0.5 - (LAVA_TOP - 1.5)) / 3.2)
        lake = (d < 1.0) & (ys <= LAVA_TOP + 1) & (ys > -64)
        self.fg[lake] = B['air']
        self.bg[lake] = B['deepslate']
        lava = lake & (ys <= LAVA_TOP - 1)
        self.fg[lava] = B['lava']
        for x in range(-1, 2):
            self.set(x, LAST_FLOOR, 'deepslate')
        for x in range(-1, 2):
            self.set(x, LAVA_TOP, 'air')
            self.set(x, LAVA_TOP + 1, 'air')
        self.set(0, LAST_FLOOR + 1, 'deepslate')
        self.set(0, LAST_FLOOR, 'deepslate')

    def _bedrock(self):
        for c in range(NX):
            h = 1 + int(self.rng.integers(0, 4))
            for k in range(h):
                y = Y_BOT + k
                if self.fg[row(y), c] != B['lava']:
                    self.fg[row(y), c] = B['bedrock']
                self.bg[row(y), c] = B['bedrock']
            self.fg[row(Y_BOT), c] = B['bedrock']


if __name__ == '__main__':
    w = World()
    names = A.BLOCKS
    import collections
    print(collections.Counter(names[v] for v in w.fg.ravel()).most_common(20))
    print('decos', len(w.deco), 'lights', len(w.lights))
    print('floor under the surface at x=0 from 54:', w.floor_below(0, CAVE_CEIL))
