"""The world of NOAH_FINAL.

Variant 'village' (the perfect world, and the night):
  * plains at z = 0 (grass tops), hills and woods further out, a lake to the west for the sunset;
  * the village: the house with the rules (south edge, the oldest building), the map house, homes, a library, a
    smithy, a tower, farms, a pen, a well, lamp posts and paths;
  * under the house a trapdoor, a ladder and a long torch-lit stair down to the underground room: a cavern holding
    24 copies of the house on a street grid, each with a date sign, lanterns on chains, a row of 19 lecterns with the
    DAY books, chests, and a pit with a ladder down to the deep chamber, where the still figure stands.
Variant 'changed' (after the freeze): the same land under a green sky, houses moved and turned, dead spruces, signs
  saying LEFT, and at spawn a giant crater with his real house at the bottom.
Variant 'finale': the village at night, with his real room inside the house.

Everything the story needs to find is recorded in world.points; signs, item frames and painted wall text in
world.signs / world.frames / world.decals (drawn as decals by the renderer).
"""
import numpy as np

import blocks as BL
import voxel as VX
from noise import fbm2d

SIZE = (384, 384, 160)
ORIGIN = (-192, -192, -100)
SPAWN = (6.5, -72.5, 0.0)
TODAY = '2026-04-03'
TOMORROW = '2026-04-04'
HOUSE_DATES = ['2018-06-14', '2018-09-02', '2019-02-02', '2019-07-19', '2019-12-24', '2020-04-04', '2020-11-03',
               '2021-01-19', '2021-06-30', '2021-10-31', '2022-03-03', '2022-08-19', '2023-01-01', '2023-05-19',
               '2023-10-04', '2024-02-29', '2024-07-07', '2024-12-19', '2025-04-04', '2025-07-01', 'NOAH_404',
               '2026-01-19', TODAY, TOMORROW]
S, E, N, W = 0, 1, 2, 3          # facings: front towards -y, +x, +y, -x


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------------------------------------
# building in a local frame
# ---------------------------------------------------------------------------------------------
class Builder:
    """Local coordinates (u, v, z) with the building's front towards -v; rotated by r quarter turns (CCW) about z
    and placed at (ox, oy, oz). Facings in states are local and rotate with the frame."""

    def __init__(self, w, ox, oy, oz=0, r=0):
        self.w, self.ox, self.oy, self.oz, self.r = w, ox, oy, oz, r % 4

    def xy(self, u, v):
        r = self.r
        if r == 0:
            return self.ox + u, self.oy + v
        if r == 1:
            return self.ox - v, self.oy + u
        if r == 2:
            return self.ox - u, self.oy - v
        return self.ox + v, self.oy - u

    def wf(self, f):
        return (f + self.r) % 4

    def st(self, state):
        return (state & ~3) | self.wf(state & 3)

    def set(self, u, v, z, name, state=0):
        x, y = self.xy(u, v)
        self.w.set(x, y, self.oz + z, name, self.st(state))

    def fill(self, u0, v0, z0, u1, v1, z1, name, state=0):
        x0, y0 = self.xy(u0, v0)
        x1, y1 = self.xy(u1, v1)
        self.w.fill(x0, y0, self.oz + z0, x1, y1, self.oz + z1, name, self.st(state))

    def get(self, u, v, z):
        x, y = self.xy(u, v)
        return self.w.get(x, y, self.oz + z)

    def world(self, u, v, z):
        """World position of a local point (block corners: add 0.5 for centres)."""
        r = self.r
        if r == 0:
            return np.array([self.ox + u, self.oy + v, self.oz + z], float)
        if r == 1:
            return np.array([self.ox - v + 1, self.oy + u, self.oz + z], float)
        if r == 2:
            return np.array([self.ox - u + 1, self.oy - v + 1, self.oz + z], float)
        return np.array([self.ox + v, self.oy - u + 1, self.oz + z], float)

    def centre(self, u, v, z):
        x, y = self.xy(u, v)
        return np.array([x + 0.5, y + 0.5, self.oz + z], float)

    def sign(self, u, v, z, facing, lines, wall=True, key=None):
        x, y = self.xy(u, v)
        f = self.wf(facing)
        self.w.set(x, y, self.oz + z, 'oak_sign', f | (BL.WALL if wall else 0))
        self.w.signs.append(dict(pos=(x, y, self.oz + z), facing=f, wall=wall, lines=list(lines), key=key))

    def frame(self, u, v, z, facing, content):
        x, y = self.xy(u, v)
        f = self.wf(facing)
        self.w.set(x, y, self.oz + z, 'item_frame', f)
        self.w.frames.append(dict(pos=(x, y, self.oz + z), facing=f, content=content))

    def decal(self, u, v, z, facing, key, size, off=0.02):
        """A painted image on the face of the wall block (u, v) that looks towards local `facing`'s front."""
        c = self.centre(u, v, z + 0.5)
        f = self.wf(facing)
        d = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}[f]
        pos = c + np.array([d[0], d[1], 0.0]) * (0.5 + off)
        self.w.decals.append(dict(pos=tuple(pos), facing=f, size=size, key=key))

    def point(self, name, u, v, z):
        self.w.points[name] = self.centre(u, v, z)

    def dyn(self, name, u, v, z):
        """A block that moves in the story (a door, a lid, the block that breaks): hidden from the static mesh and
        recorded (world position, block, state) so the shots draw it as a prop."""
        x, y = self.xy(u, v)
        zz = self.oz + z
        i, j, k = self.w.ix(x, y, zz)
        self.w.dyn[name] = dict(pos=(x, y, zz), block=BL.REG[self.w.ids[i, j, k]].name,
                                state=int(self.w.state[i, j, k]))
        self.w.hidden[i, j, k] = True


def new_world():
    w = VX.World(SIZE, ORIGIN)
    w.points = {}
    w.decals = []
    w.dyn = {}
    return w


# ---------------------------------------------------------------------------------------------
# terrain
# ---------------------------------------------------------------------------------------------
VILLAGE_C = np.array([4.0, -10.0])


def heightmap(w, seed=5, crater=None):
    X0, Y0, _ = w.origin
    xs = np.arange(w.size[0]) + X0 + 0.5
    ys = np.arange(w.size[1]) + Y0 + 0.5
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    d = np.hypot((X - VILLAGE_C[0]) / 1.15, Y - VILLAGE_C[1])
    rise = smoothstep(62.0, 120.0, d)
    hills = 3.0 + 9.0 * fbm2d(X / 55.0, Y / 55.0, octaves=3, seed=seed) + 3.0 * fbm2d(X / 19.0, Y / 19.0, octaves=2,
                                                                                    seed=seed + 1)
    far = smoothstep(120.0, 190.0, d) * (8.0 + 14.0 * np.abs(fbm2d(X / 70.0, Y / 70.0, octaves=3, seed=seed + 2)))
    h = rise * np.maximum(hills, 0.0) + far
    # gentle swells near the village (never under it)
    h += smoothstep(45.0, 70.0, d) * np.maximum(1.5 * fbm2d(X / 21.0, Y / 21.0, octaves=2, seed=seed + 4), 0.0)
    H = np.floor(h + 0.5).astype(np.int32)
    # the lake to the west, where the sun sets
    lk = np.hypot((X + 92.0) / 1.4, Y + 28.0)
    lake = lk < 24.0
    H[lake] = -3
    shore = (lk >= 24.0) & (lk < 30.0)
    H[shore] = np.minimum(H[shore], 0)
    if crater is not None:
        cx, cy, r, depth = crater
        dc = np.hypot(X - cx, Y - cy)
        bowl = dc < r
        prof = -depth * np.sqrt(np.clip(1 - (dc / r) ** 2, 0, 1)) ** 0.9
        H[bowl] = np.minimum(H[bowl], np.floor(prof[bowl]).astype(np.int32))
    return H, lake


def fill_terrain(w, H, lake):
    X, Y, Z = w.size
    oz = w.origin[2]
    z = np.arange(Z) + oz
    Hb = H[:, :, None]
    ids = np.zeros((X, Y, Z), np.uint16)
    ids[:] = BL.B['stone']
    ids[:, :, z < -58] = BL.B['deepslate']
    dirt = (z[None, None, :] >= Hb - 4) & (z[None, None, :] < Hb - 1)
    ids = np.where(dirt, BL.B['dirt'], ids)
    top = z[None, None, :] == Hb - 1
    ids = np.where(top, BL.B['grass_block'], ids)
    air = z[None, None, :] >= Hb
    ids = np.where(air, 0, ids)
    # lake: sand bed, water up to z = -1 (surface at the grass level)
    lk = lake[:, :, None]
    ids = np.where(lk & (z[None, None, :] == Hb - 1), BL.B['sand'], ids)
    ids = np.where(lk & (z[None, None, :] >= Hb) & (z[None, None, :] <= -1), BL.B['water'], ids)
    ids[:, :, 0] = BL.B['obsidian']
    w.ids[:] = ids
    # sand around the lake
    near = np.zeros_like(lake)
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            near |= np.roll(np.roll(lake, dx, 0), dy, 1)
    shore = near & ~lake
    k = (-1 - oz)
    sel = shore & (H == 0)
    w.ids[:, :, k][sel] = BL.B['sand']


def path(w, pts, width=3, rng=None, block='dirt_path'):
    rng = rng or np.random.default_rng(0)
    for (a, b) in zip(pts[:-1], pts[1:]):
        a, b = np.array(a, float), np.array(b, float)
        n = int(np.ceil(np.linalg.norm(b - a) * 2)) + 1
        for t in np.linspace(0, 1, n):
            p = a + (b - a) * t
            for dx in range(-(width // 2) - 1, width // 2 + 2):
                for dy in range(-(width // 2) - 1, width // 2 + 2):
                    rr = max(abs(dx), abs(dy))
                    if rr > width // 2 and rng.random() > 0.35:
                        continue
                    x, y = int(np.floor(p[0])) + dx, int(np.floor(p[1])) + dy
                    if w.get(x, y, -1) == 'grass_block' and w.get(x, y, 0) in ('air', 'tall_grass', 'poppy',
                                                                                 'dandelion', 'cornflower'):
                        w.set(x, y, -1, block)
                        w.set(x, y, 0, 'air')


# ---------------------------------------------------------------------------------------------
# trees and plants
# ---------------------------------------------------------------------------------------------
def tree(w, x, y, z, rng, kind='oak'):
    if kind == 'dead':
        h = int(rng.integers(5, 9))
        w.fill(x, y, z, x, y, z + h, 'dead_log')
        for _ in range(int(rng.integers(1, 4))):
            dz = int(rng.integers(2, h))
            dx, dy = [(1, 0), (-1, 0), (0, 1), (0, -1)][rng.integers(4)]
            for k in range(1, int(rng.integers(2, 4))):
                w.set(x + dx * k, y + dy * k, z + dz + (k > 1), 'dead_log')
        return
    if kind == 'spruce':
        h = int(rng.integers(7, 12))
        w.fill(x, y, z, x, y, z + h - 1, 'spruce_log')
        for k in range(2, h + 1):
            r = max(0, int((h + 1 - k) * 0.45)) if k % 2 == 0 else max(0, int((h + 1 - k) * 0.3))
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) + abs(dy) <= r + (r > 1) and (dx or dy or k >= h):
                        if w.get(x + dx, y + dy, z + k) == 'air':
                            w.set(x + dx, y + dy, z + k, 'spruce_leaves')
        w.set(x, y, z + h, 'spruce_leaves')
        w.set(x, y, z + h + 1, 'spruce_leaves')
        return
    birch = kind == 'birch'
    h = int(rng.integers(5, 8)) if birch else int(rng.integers(4, 7))
    w.fill(x, y, z, x, y, z + h - 1, 'birch_log' if birch else 'oak_log')
    leaf = 'birch_leaves' if birch else 'oak_leaves'
    for dz in (h - 3, h - 2):
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if abs(dx) == 2 and abs(dy) == 2 and rng.random() < 0.6:
                    continue
                if w.get(x + dx, y + dy, z + dz) == 'air':
                    w.set(x + dx, y + dy, z + dz, leaf)
    for dz in (h - 1, h):
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if abs(dx) == 1 and abs(dy) == 1 and (dz == h or rng.random() < 0.4):
                    continue
                if w.get(x + dx, y + dy, z + dz) == 'air':
                    w.set(x + dx, y + dy, z + dz, leaf)
    if not birch and rng.random() < 0.35:            # a bigger oak: a second canopy lobe
        ox, oy = [(2, 1), (-2, 1), (1, -2), (-1, 2)][rng.integers(4)]
        for dz in (h - 2, h - 1):
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if w.get(x + ox + dx, y + oy + dy, z + dz) == 'air':
                        w.set(x + ox + dx, y + oy + dy, z + dz, leaf)


def poisson(rng, box, mind, accept, n_try=30000):
    x0, y0, x1, y1 = box
    pts = []
    cell = mind / np.sqrt(2)
    grid = {}
    for _ in range(n_try):
        p = (rng.uniform(x0, x1), rng.uniform(y0, y1))
        if not accept(*p):
            continue
        gx, gy = int(p[0] / cell), int(p[1] / cell)
        ok = True
        for i in range(gx - 2, gx + 3):
            for j in range(gy - 2, gy + 3):
                q = grid.get((i, j))
                if q is not None and (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 < mind * mind:
                    ok = False
        if ok:
            grid[(gx, gy)] = p
            pts.append(p)
    return pts


def plant_trees(w, H, rng, kinds=('oak', 'oak', 'oak', 'birch'), keep_out=None, density=1.0):
    X0, Y0, _ = w.origin

    def ht(x, y):
        i, j = int(np.floor(x)) - X0, int(np.floor(y)) - Y0
        if 0 <= i < w.size[0] and 0 <= j < w.size[1]:
            return H[i, j]
        return -99

    def accept(x, y):
        h = ht(x, y)
        if h < 0 or abs(x) > 186 or abs(y) > 186:
            return False
        for dx in (-2, 2):
            for dy in (-2, 2):
                if abs(ht(x + dx, y + dy) - h) > 1:
                    return False
        if w.get(np.floor(x), np.floor(y), h - 1) != 'grass_block' or w.get(np.floor(x), np.floor(y), h) != 'air':
            return False
        if keep_out is not None and keep_out(x, y):
            return False
        d = np.hypot((x - VILLAGE_C[0]) / 1.15, y - VILLAGE_C[1])
        dens = np.clip((d - 58.0) / 40.0, 0.0, 1.0) * 0.9 + 0.04
        # woods to the north and east; open plains towards the lake and the sunset
        if x < -40:
            dens *= 0.25
        return rng.random() < dens * density

    pts = poisson(rng, (-186, -186, 186, 186), 6.0, accept)
    for (x, y) in pts:
        xi, yi = int(np.floor(x)), int(np.floor(y))
        tree(w, xi, yi, ht(x, y), rng, kinds[rng.integers(len(kinds))])
    return pts


def scatter_plants(w, H, rng, frac=0.16, keep_out=None, flowers=True):
    X0, Y0, _ = w.origin
    oz = w.origin[2]
    for i in range(0, w.size[0]):
        col_ids = w.ids[i]
        for j in range(0, w.size[1]):
            h = H[i, j]
            if h < 0:
                continue
            k = h - oz
            if k >= w.size[2] or col_ids[j, k - 1] != BL.B['grass_block'] or col_ids[j, k] != 0:
                continue
            r = rng.random()
            if r > frac:
                continue
            x, y = i + X0, j + Y0
            if keep_out is not None and keep_out(x, y):
                continue
            if flowers and r < frac * 0.07:
                name = ('poppy', 'dandelion', 'cornflower')[rng.integers(3)]
            else:
                name = 'tall_grass'
            col_ids[j, k] = BL.B[name]


# ---------------------------------------------------------------------------------------------
# buildings
# ---------------------------------------------------------------------------------------------
def gable_roof(b, u0, u1, v0, v1, zb, stairs='spruce_stairs', fill='spruce_planks', ridge='spruce_slab'):
    """A pitched roof over u0..u1 (ridge along v), one block of overhang all round, starting at z = zb."""
    k = 0
    while True:
        ua, ub = u0 - 1 + k, u1 + 1 - k
        if ub - ua < 0:
            break
        z = zb + k
        if ub - ua <= 0:
            b.fill(ua, v0 - 1, z, ua, v1 + 1, z, fill)
            break
        if ub - ua == 1:
            b.fill(ua, v0 - 1, z, ub, v1 + 1, z, fill)
            break
        b.fill(ua, v0 - 1, z, ua, v1 + 1, z, stairs, W)       # rises towards +u
        b.fill(ub, v0 - 1, z, ub, v1 + 1, z, stairs, E)
        # gable ends and the roof's underside
        if ua + 1 <= ub - 1:
            for vv in (v0, v1):
                b.fill(ua + 1, vv, z, ub - 1, vv, z, fill)
        k += 1
    return zb + k


def house_shell(b, W_, D_, wall='oak_planks', base='cobblestone', corner='oak_log', floor='oak_planks', h=3,
                door=None, windows=(), roof=True, stairs='spruce_stairs', roof_fill='spruce_planks',
                door_block='oak_door', ceiling='spruce_planks'):
    """Walls round u 0..W_, v 0..D_ (inclusive), floor at z = -1, walls z = 0..h-1, flat ceiling at z = h."""
    b.fill(0, 0, -3, W_, D_, -2, 'cobblestone')
    b.fill(0, 0, -1, W_, D_, -1, base)
    b.fill(1, 1, -1, W_ - 1, D_ - 1, -1, floor)
    b.fill(0, 0, 0, W_, D_, h - 1, wall)
    b.fill(0, 0, 0, W_, 0, 0, base)
    b.fill(0, D_, 0, W_, D_, 0, base)
    b.fill(0, 0, 0, 0, D_, 0, base)
    b.fill(W_, 0, 0, W_, D_, 0, base)
    for (u, v) in ((0, 0), (W_, 0), (0, D_), (W_, D_)):
        b.fill(u, v, 0, u, v, h, corner)
    b.fill(1, 1, 0, W_ - 1, D_ - 1, h + 3, 'air')
    b.fill(0, 0, h, W_, D_, h, ceiling)
    b.fill(0, 0, h, W_, 0, h, corner)
    b.fill(0, D_, h, W_, D_, h, corner)
    if door is not None:
        du = door
        b.set(du, 0, 0, door_block, S)
        b.set(du, 0, 1, door_block, S | BL.HALF_TOP)
        b.fill(du, -1, -1, du, -1, -1, 'cobblestone')
    for (u, v, z0, z1) in windows:
        b.fill(u, v, z0, u, v, z1, 'glass_pane')
    top = None
    if roof:
        top = gable_roof(b, 0, W_, 0, D_, h + 1, stairs, roof_fill)
    return top


def rules_house(w, ox, oy, r=0, variant='today', date=None, decay=0.0, rng=None, chimney_smoke=True):
    """The house with the rules. 9 x 11 (u 0..8, v 0..10), door in the middle of the front.
    variant: today (the original), copy_<date> in the cavern, finale (his real room inside)."""
    rng = rng or np.random.default_rng(1)
    b = Builder(w, ox, oy, 0 if variant in ('today', 'finale') else w._copy_z, r)
    win = [(0, 2, 1, 2), (0, 3, 1, 2), (0, 4, 1, 2), (0, 7, 1, 2), (8, 7, 1, 2), (3, 10, 1, 1)]
    house_shell(b, 8, 10, door=4, windows=win)
    # chimney at the back left, up through the roof
    b.fill(1, 10, 0, 1, 10, 9, 'cobblestone')
    b.fill(0, 11, -1, 2, 11, -1, 'cobblestone')
    if variant == 'finale':
        real_room_interior(b, 1, 1, 7, 9, 3)
        b.point('house_door', 4, -1, 0)
        b.point('house_in', 4, 2, 0)
        b.set(3, -1, 1, 'torch', BL.WALL | N)
        b.set(5, -1, 1, 'torch', BL.WALL | N)
        b.dyn('house_door_bottom', 4, 0, 0)
        b.dyn('house_door_top', 4, 0, 1)
        return b
    # interior
    b.set(6, 9, 0, 'red_bed', S | BL.HALF_TOP)
    b.set(6, 8, 0, 'red_bed', S)
    b.set(2, 9, 0, 'chest', S)
    b.set(3, 9, 0, 'lit_furnace' if variant == 'today' else 'furnace', S)
    b.set(4, 9, 0, 'furnace', S)
    b.set(7, 6, 0, 'crafting_table', W)
    b.set(7, 7, 0, 'bookshelf')
    b.set(7, 5, 0, 'bookshelf')
    b.set(7, 5, 1, 'bookshelf')
    b.set(1, 5, 0, 'oak_fence')                      # a little table with a lantern on it
    b.set(1, 5, 1, 'lantern') if variant != 'today' else None
    b.set(2, 2, -1, 'oak_trapdoor', BL.HALF_TOP | S)
    b.set(2, 2, 0, 'red_carpet')
    b.set(1, 1, 0, 'red_carpet')
    b.set(2, 1, 0, 'red_carpet')
    b.set(1, 2, 0, 'red_carpet')
    # torches: by the bed and by the door; outside by the door
    b.set(7, 8, 1, 'torch', BL.WALL | W)
    b.set(1, 8, 1, 'torch', BL.WALL | E)
    b.set(3, -1, 1, 'torch', BL.WALL | N)
    b.set(5, -1, 1, 'torch', BL.WALL | N)
    if variant not in ('copy_today', 'copy_tomorrow'):
        b.frame(3, 9, 2, S, ('map', 'TODAY_SMALL'))
        b.frame(5, 9, 2, S, ('item', 'clock'))
    b.frame(1, 6, 2, E, ('item', 'wooden_pickaxe'))
    b.frame(7, 2, 2, W, ('item', 'iron_axe'))
    if variant == 'today':
        b.point('house_door', 4, -1, 0)
        b.point('house_in', 4, 2, 0)
        b.point('house_bed', 6, 8, 0)
        b.point('house_chest', 2, 9, 0)
        b.point('house_furnace', 3, 9, 0)
        b.point('house_trapdoor', 2, 2, -1)
        b.point('house_ceiling_block', 4, 5, 3)
        b.point('house_centre', 4, 5, 0)
        b.point('house_window_w', 0, 3, 1)
        b.point('house_window_e', 8, 7, 1)
        # the shaft under the trapdoor
        b.fill(2, 2, -6, 2, 2, -2, 'air')
        for z in range(-6, -1):
            b.set(2, 1, z, 'stone_bricks')
            b.set(2, 3, z, 'stone_bricks')
            b.set(2, 2, z, 'ladder', N)          # on the south wall: he climbs down facing it, then turns round
        b.w._house_frame = b
        for name, (u, v, z) in (('house_door_bottom', (4, 0, 0)), ('house_door_top', (4, 0, 1)),
                                ('house_chest', (2, 9, 0)), ('house_ceiling', (4, 5, 3)),
                                ('house_trapdoor', (2, 2, -1)), ('house_carpet', (2, 2, 0))):
            b.dyn(name, u, v, z)
    else:
        # the copies: dated, some decayed; TODAY's has what happened tonight
        if variant == 'copy_today':
            b.set(4, 5, 3, 'air')                          # the broken ceiling block
            b.set(4, 5, 0, 'oak_sign', S)                  # the fallen sign, on the floor
            w.signs.append(dict(pos=tuple(b.xy(4, 5)) + (b.oz,), facing=b.wf(S), wall=False,
                                lines=['GOOD.', "YOU DIDN'T", 'TURN AROUND.'], key='sign_good'))
            b.set(5, 9, 0, 'red_bed', S | BL.HALF_TOP)     # a second bed
            b.set(5, 8, 0, 'red_bed', S)
            b.set(4, 9, 0, 'air')
            b.set(3, 10, 1, 'oak_planks')                  # no back window: the wall is painted
            b.decal(4, 10, 1, S, 'wall_sleep', (5.0, 1.4))
            b.set(1, 5, 1, 'air')
            b.point('copy_today_door', 4, -1, 0)
            b.point('copy_today_in', 4, 2, 0)
            b.dyn('copy_door_bottom', 4, 0, 0)
            b.dyn('copy_door_top', 4, 0, 1)
            b.point('copy_today_bed2', 5, 8, 0)
            b.point('copy_today_wall', 4, 9, 1)
        if variant == 'copy_tomorrow':
            b.set(1, 5, 1, 'lantern')
            b.set(4, 5, 3, 'air')
            b.set(5, 9, 0, 'red_bed', S | BL.HALF_TOP)
            b.set(5, 8, 0, 'red_bed', S)
            b.set(4, 9, 0, 'air')
        if decay > 0:
            decay_box(b, 0, 0, -1, 8, 10, 9, decay, rng)
        if date is not None:
            b.sign(4, -2, 0, S, [date], wall=False, key=f'date_{date}')
            if variant == 'copy_today':
                b.point('copy_today_sign', 4, -2, 0)
            if variant == 'copy_tomorrow':
                b.point('copy_tomorrow_sign', 4, -2, 0)
    return b


def decay_box(b, u0, v0, z0, u1, v1, z1, amount, rng):
    """Age a copy: mossy cobblestone, cobwebs, missing blocks, no torches."""
    for u in range(u0, u1 + 1):
        for v in range(v0, v1 + 1):
            for z in range(z0, z1 + 1):
                n = b.get(u, v, z)
                if n == 'cobblestone' and rng.random() < amount:
                    b.set(u, v, z, 'mossy_cobblestone')
                elif n in ('oak_planks', 'spruce_planks', 'spruce_stairs', 'glass_pane') and rng.random() < \
                        amount * 0.25:
                    b.set(u, v, z, 'air')
                elif n in ('torch', 'lantern', 'lit_furnace') and rng.random() < 0.8:
                    b.set(u, v, z, 'air' if n != 'lit_furnace' else 'furnace')
                elif n == 'air' and 1 <= u <= u1 - 1 and 1 <= v <= v1 - 1 and z >= 1 and rng.random() < amount * 0.05:
                    b.set(u, v, z, 'cobweb')


def real_room_interior(b, u0, v0, u1, v1, h):
    """His real bedroom, in blocks: white walls, grey carpet, a desk with the computer, a blue bed, a window with
    the blinds down, a poster, a lamp."""
    b.fill(u0, v0, -1, u1, v1, -1, 'light_gray_concrete')
    b.fill(u0, v0, 0, u1, v1, 0, 'gray_carpet')
    b.fill(u0 - 1, v0 - 1, 0, u1 + 1, v0 - 1, h - 1, 'white_concrete')
    b.fill(u0 - 1, v1 + 1, 0, u1 + 1, v1 + 1, h - 1, 'white_concrete')
    b.fill(u0 - 1, v0 - 1, 0, u0 - 1, v1 + 1, h - 1, 'white_concrete')
    b.fill(u1 + 1, v0 - 1, 0, u1 + 1, v1 + 1, h - 1, 'white_concrete')
    b.fill(u0, v0, h, u1, v1, h, 'white_concrete')
    # desk along the back wall: slabs on the carpet, the monitor is a prop (screen), a chair
    uc = (u0 + u1) // 2
    b.fill(uc - 1, v1, 0, uc + 1, v1, 0, 'air')
    b.fill(uc - 1, v1, 0, uc + 1, v1, 0, 'white_slab', BL.HALF_TOP)
    b.set(uc, v1 - 1, 0, 'spruce_stairs', N)                                     # the chair faces the desk
    b.set(u1, v0 + 1, 0, 'blue_bed', W | BL.HALF_TOP) if False else None
    b.set(u1, v0 + 2, 0, 'blue_bed', S | BL.HALF_TOP)
    b.set(u1, v0 + 1, 0, 'blue_bed', S)
    b.set(u0, v1, 0, 'bookshelf')
    b.set(u0, v1, 1, 'lantern')
    b.decal(uc, v1 + 1, 1, S, 'poster_space', (1.0, 1.3))
    b.decal(u0 - 1, v0 + 2, 1, E, 'poster_band', (1.0, 1.3))
    b.point('room_desk', uc, v1, 0)
    b.point('room_chair', uc, v1 - 1, 0)
    b.point('room_monitor', uc, v1, 1)
    b.point('room_bed', u1, v0 + 1, 0)
    b.point('room_door_in', (u0 + u1) // 2, v0, 0)


def small_house(w, ox, oy, r, rng, style=0):
    b = Builder(w, ox, oy, 0, r)
    W_, D_ = [(6, 6), (6, 8), (8, 6), (5, 7)][style % 4]
    mats = [('oak_planks', 'oak_log', 'spruce_stairs', 'spruce_planks'),
            ('spruce_planks', 'spruce_log', 'oak_stairs', 'oak_planks'),
            ('cobblestone', 'oak_log', 'spruce_stairs', 'spruce_planks'),
            ('oak_planks', 'birch_log', 'oak_stairs', 'dark_oak_planks')][style % 4]
    wall, corner, stairs, rf = mats
    win = [(0, D_ // 2, 1, 1), (W_, D_ // 2, 1, 1), (W_ // 2 + 1, D_, 1, 1)]
    house_shell(b, W_, D_, wall=wall, corner=corner, door=W_ // 2, windows=win, stairs=stairs, roof_fill=rf,
                door_block='spruce_door' if style % 2 else 'oak_door')
    b.set(W_ - 1, D_ - 1, 0, 'red_bed' if style % 2 == 0 else 'blue_bed', S | BL.HALF_TOP)
    b.set(W_ - 1, D_ - 2, 0, 'red_bed' if style % 2 == 0 else 'blue_bed', S)
    b.set(1, D_ - 1, 0, 'crafting_table')
    b.set(1, 1, 0, 'chest', N)
    b.set(W_ // 2 - 1, -1, 1, 'torch', BL.WALL | N)
    b.set(1, D_ // 2, 1, 'torch', BL.WALL | E)
    for u in range(1, W_):
        if rng.random() < 0.5 and u != W_ // 2:
            b.set(u, -1, 0, ('poppy', 'dandelion', 'cornflower')[rng.integers(3)])
    return b


def map_house(w, ox, oy, r):
    b = Builder(w, ox, oy, 0, r)
    W_, D_ = 12, 8
    win = [(0, 4, 1, 2), (W_, 4, 1, 2), (3, 0, 1, 2), (9, 0, 1, 2)]
    house_shell(b, W_, D_, wall='spruce_planks', corner='spruce_log', base='stone_bricks', door=6, windows=win,
                stairs='cobblestone_stairs', roof_fill='cobblestone', h=4)
    # the wall of maps: 5 x 2 frames on the back wall, the four labelled ones in the middle
    maps = [('map', 'OLD_1'), ('map', 'BEFORE'), ('map', 'ABANDONED'), ('map', 'BURNED'), ('map', 'OLD_2'),
            ('map', 'OLD_3'), ('map', 'TODAY'), ('map', 'OLD_4'), ('map', 'OLD_5'), ('map', 'OLD_6')]
    k = 0
    for z in (2, 1):
        for u in range(4, 9):
            b.frame(u, D_ - 1, z, S, maps[k])
            if maps[k][1] in ('BEFORE', 'ABANDONED', 'BURNED', 'TODAY'):
                b.point('map_' + maps[k][1], u, D_ - 1, z)
            k += 1
    # labels under the top row and on the lower row's frames' sides are painted on the wall above the frames
    for u, lab in ((5, 'BEFORE'), (6, 'ABANDONED'), (7, 'BURNED')):
        b.decal(u, D_, 3, S, 'label_' + lab, (1.0, 0.28))
    b.decal(5, D_, 0, S, 'label_TODAY', (1.0, 0.28))
    b.set(1, D_ - 1, 0, 'crafting_table')
    b.set(W_ - 1, D_ - 1, 0, 'lectern', S)
    b.set(1, 1, 0, 'bookshelf')
    b.set(1, 2, 0, 'bookshelf')
    b.set(W_ - 1, 1, 0, 'chest', N)
    b.set(3, D_ - 1, 3, 'lantern', BL.WALL)
    b.set(9, D_ - 1, 3, 'lantern', BL.WALL)
    b.set(5, -1, 1, 'torch', BL.WALL | N)
    b.set(7, -1, 1, 'torch', BL.WALL | N)
    b.point('maphouse_door', 6, -1, 0)
    b.dyn('maphouse_door_bottom', 6, 0, 0)
    b.dyn('maphouse_door_top', 6, 0, 1)
    b.point('maphouse_in', 6, 1, 0)
    b.point('maphouse_wall', 6, D_ - 2, 0)
    return b


def library(w, ox, oy, r):
    b = Builder(w, ox, oy, 0, r)
    W_, D_ = 10, 8
    house_shell(b, W_, D_, wall='cobblestone', corner='oak_log', base='stone_bricks', door=5,
                windows=[(0, 4, 1, 2), (W_, 4, 1, 2)], h=4)
    for u in range(1, W_):
        for z in range(0, 3):
            if u != 5:
                b.set(u, D_ - 1, z, 'bookshelf')
    b.set(5, D_ - 2, 0, 'lectern', S)
    b.set(2, 3, 3, 'lantern', BL.WALL)
    b.set(8, 3, 3, 'lantern', BL.WALL)
    return b


def smithy(w, ox, oy, r):
    b = Builder(w, ox, oy, 0, r)
    W_, D_ = 8, 6
    house_shell(b, W_, D_, wall='cobblestone', corner='oak_log', base='stone_bricks', door=4, windows=[(0, 3, 1, 1)],
                stairs='cobblestone_stairs', roof_fill='cobblestone')
    b.set(1, D_ - 1, 0, 'lit_furnace', S)
    b.set(2, D_ - 1, 0, 'lit_furnace', S)
    b.set(3, D_ - 1, 0, 'furnace', S)
    b.set(W_ - 1, 1, 0, 'chest', W)
    b.set(3, -1, 1, 'torch', BL.WALL | N)
    return b


def tower(w, ox, oy):
    b = Builder(w, ox, oy, 0, 0)
    b.fill(0, 0, -2, 4, 4, 14, 'cobblestone')
    b.fill(1, 1, 0, 3, 3, 13, 'air')
    b.fill(0, 0, 14, 4, 4, 14, 'stone_bricks')
    for u in (0, 2, 4):
        b.set(u, 0, 15, 'cobblestone')
        b.set(u, 4, 15, 'cobblestone')
        b.set(0, u, 15, 'cobblestone')
        b.set(4, u, 15, 'cobblestone')
    b.set(2, 0, 0, 'oak_door', S)
    b.set(2, 0, 1, 'oak_door', S | BL.HALF_TOP)
    for z in (5, 9, 12):
        b.set(2, 0, z, 'glass_pane')
        b.set(0, 2, z, 'glass_pane')
        b.set(4, 2, z, 'glass_pane')
    for z in range(0, 14):
        b.set(3, 3, z, 'ladder', S)
    b.set(2, 2, 15, 'lantern')
    b.point('tower_top', 2, 2, 15)
    return b


def well(w, ox, oy):
    b = Builder(w, ox, oy, 0, 0)
    b.fill(0, 0, -4, 4, 4, 0, 'cobblestone')
    b.fill(1, 1, -4, 3, 3, -1, 'water')
    b.fill(1, 1, 0, 3, 3, 0, 'air')
    for (u, v) in ((0, 0), (4, 0), (0, 4), (4, 4)):
        b.fill(u, v, 1, u, v, 2, 'oak_fence')
    b.fill(0, 0, 3, 4, 4, 3, 'cobblestone_slab')
    b.point('well', 2, 2, 0)
    return b


def farm(w, ox, oy, W_, D_, young=False):
    b = Builder(w, ox, oy, 0, 0)
    b.fill(-1, -1, -1, W_ + 1, D_ + 1, -1, 'oak_log')
    b.fill(0, 0, -1, W_, D_, -1, 'farmland')
    b.fill(0, 0, 0, W_, D_, 0, 'air')
    b.fill(W_ // 2, 0, -1, W_ // 2, D_, -1, 'water')
    rng = np.random.default_rng(abs(ox * 7 + oy * 131))
    for u in range(0, W_ + 1):
        if u == W_ // 2:
            continue
        for v in range(0, D_ + 1):
            b.set(u, v, 0, 'wheat_young' if (young or rng.random() < 0.2) else 'wheat')
    return b


def pen(w, ox, oy, W_, D_):
    b = Builder(w, ox, oy, 0, 0)
    b.fill(0, 0, 0, W_, 0, 0, 'oak_fence')
    b.fill(0, D_, 0, W_, D_, 0, 'oak_fence')
    b.fill(0, 0, 0, 0, D_, 0, 'oak_fence')
    b.fill(W_, 0, 0, W_, D_, 0, 'oak_fence')
    b.set(W_ // 2, 0, 0, 'air')
    b.set(1, D_ - 1, 0, 'hay_block')
    b.set(2, D_ - 1, 0, 'hay_block')
    b.set(1, D_ - 1, 1, 'hay_block')
    b.fill(1, 1, 0, W_ - 1, D_ - 1, 0, 'air') if False else None
    for u in range(1, W_):
        for v in range(1, D_):
            if b.get(u, v, 0) in ('tall_grass', 'poppy', 'dandelion', 'cornflower'):
                b.set(u, v, 0, 'air')
    b.point('pen', W_ // 2, D_ // 2, 0)
    return b


def lamp_post(w, x, y, hang=False):
    w.fill(x, y, 0, x, y, 2, 'oak_fence')
    w.set(x, y, 3, 'lantern')


# ---------------------------------------------------------------------------------------------
# underground
# ---------------------------------------------------------------------------------------------
CAVE = dict(x0=-46, x1=46, y0=6, y1=90, zf=-40, zc=-13)        # floor top at zf, ceiling underside at zc
CHAMBER = dict(c=(0.0, 58.0), zf=-96, zc=-52, r=36.0)


def stair_tunnel(w, x, y_start, z_start, z_end):
    """Down along +y from (x, y_start, z_start) to z_end, one block down per block forward, torches every 5."""
    y = y_start
    z = z_start
    k = 0
    while z > z_end:
        w.fill(x - 1, y, z - 1, x + 1, y, z + 4, 'stone_bricks')
        w.fill(x, y, z, x, y, z + 2, 'air')
        w.set(x, y, z - 1, 'cobblestone')
        w.set(x, y, z, 'cobblestone_stairs', N)
        if k % 5 == 0:
            w.set(x - 1, y, z + 1, 'torch', BL.WALL | 1)      # on the -x wall (the first one at the ladder's foot)
        k += 1
        y += 1
        z -= 1
    w.fill(x - 1, y, z - 1, x + 1, y + 2, z + 4, 'stone_bricks')
    w.fill(x, y, z, x, y + 2, z + 2, 'air')
    return y, z


def cavern(w, rng):
    c = CAVE
    x0, x1, y0, y1, zf, zc = c['x0'], c['x1'], c['y0'], c['y1'], c['zf'], c['zc']
    # rough rock volume, then the room
    w.fill(x0 - 3, y0 - 3, zf - 3, x1 + 3, y1 + 3, zc + 3, 'stone')
    w.fill(x0, y0, zf, x1, y1, zc - 1, 'air')
    w.fill(x0, y0, zf - 1, x1, y1, zf - 1, 'smooth_stone')
    # the ceiling: uneven, with deepslate patches; the walls: stone and cobblestone
    X = np.arange(x0, x1 + 1)
    Y = np.arange(y0, y1 + 1)
    for xi in X[::1]:
        for yi in Y[::1]:
            n = fbm2d(np.array(xi / 9.0), np.array(yi / 9.0), octaves=2, seed=41)
            drop = int(max(0, n * 5 + 1))
            if drop:
                w.fill(xi, yi, zc - drop, xi, yi, zc - 1, 'stone')
    for xi in range(x0, x1 + 1):
        for z in range(zf, zc):
            if rng.random() < 0.2:
                w.set(xi, y0 - 1, z, 'cobblestone')
                w.set(xi, y1 + 1, z, 'cobblestone')
    # streets of copies: 4 rows x 6 columns, 9 wide x 11 deep, facing the stair (south)
    w._copy_z = zf
    k = 0
    rows = [12, 29, 46, 63]
    cols = [-42, -27, -12, 3, 18, 33]
    positions = []
    for ri, yy in enumerate(rows):
        for ci, xx in enumerate(cols):
            positions.append((xx, yy))
    order = list(range(len(positions)))
    for k, (xx, yy) in enumerate(positions):
        date = HOUSE_DATES[k]
        variant = 'copy'
        if date == TODAY:
            variant = 'copy_today'
        elif date == TOMORROW:
            variant = 'copy_tomorrow'
        age = 1.0 - k / len(positions)
        dec = 0.55 * age ** 1.3 if variant == 'copy' and k < 19 else 0.0
        rules_house(w, xx, yy, 0, variant=variant, date=date, decay=dec, rng=rng)
    del order
    # lanterns on chains from the ceiling along the streets, and on posts at crossings
    for yy in (8, 24, 41, 58, 75):
        for xx in range(-44, 46, 6):
            zc_here = zc - 1
            while w.get(xx, yy, zc_here) != 'air' and zc_here > zf + 4:
                zc_here -= 1
            for z in range(zf + 6, zc_here + 1):
                w.set(xx, yy, z, 'chain')
            w.set(xx, yy, zf + 5, 'lantern', BL.WALL)
    # lantern posts at the crossings between the rows
    for yy in (8, 24, 41, 58, 75):
        for xx in (-31, -16, 1, 14, 29):
            w.fill(xx, yy + 2, zf, xx, yy + 2, zf + 1, 'oak_fence')
            w.set(xx, yy + 2, zf + 2, 'lantern')
    # the lecterns: DAY 1..19 along the north wall, a sign on the wall above each
    xs = np.linspace(-36, 36, 19).round().astype(int)
    for d, xx in enumerate(xs):
        yy = y1 - 3
        w.set(xx, yy, zf, 'lectern', S)
        w.set(xx, yy + 3, zf + 2, 'oak_sign', S | BL.WALL)
        w.fill(xx - 1, yy + 4, zf, xx + 1, yy + 4, zf + 4, 'stone_bricks')
        w.signs.append(dict(pos=(xx, yy + 3, zf + 2), facing=S, wall=True, lines=[f'DAY {d + 1}'],
                            key=f'day_{d + 1}'))
        w.points[f'lectern_{d + 1}'] = np.array([xx + 0.5, yy + 0.5, zf], float)
        if d % 3 == 1:
            w.set(xx, yy + 2, zf, 'chest', S)
            w.points[f'lchest_{d + 1}'] = np.array([xx + 0.5, yy + 2.5, zf], float)
        w.set(xx, yy + 1, zf + 3, 'soul_lantern', BL.WALL) if d % 4 == 0 else None
    w.fill(x0, y1 + 1, zf, x1, y1 + 1, zf + 5, 'stone_bricks')
    for xx in range(-36, 40, 8):
        yy = y1 - 5
        ztop = zc - 1
        while w.get(xx, yy, ztop) != 'air' and ztop > zf + 5:
            ztop -= 1
        for z in range(zf + 5, ztop + 1):
            w.set(xx, yy, z, 'chain')
        w.set(xx, yy, zf + 4, 'lantern', BL.WALL)
    # the pit: a 3 x 3 hole in the floor behind the middle lectern, a ladder down its side
    px, py = 0, y1 - 1
    w.fill(px - 1, py - 1, zf - 1, px + 1, py + 1, zf - 1, 'air')
    w.points['pit'] = np.array([px + 0.5, py + 0.5, zf], float)
    return (px, py)


def chamber(w, rng, pit):
    c = CHAMBER
    cx, cy = c['c']
    zf, zc, r = c['zf'], c['zc'], c['r']
    X0, Y0, Z0 = w.origin
    i0, i1 = int(cx - r - 3) - X0, int(cx + r + 3) - X0
    j0, j1 = int(cy - r - 3) - Y0, int(cy + r + 3) - Y0
    k0, k1 = zf - 2 - Z0, zc + 2 - Z0
    xs = np.arange(i0, i1) + X0 + 0.5
    ys = np.arange(j0, j1) + Y0 + 0.5
    zs = np.arange(k0, k1) + Z0 + 0.5
    Xg, Yg, Zg = np.meshgrid(xs, ys, zs, indexing='ij')
    zmid = (zf + zc) / 2
    d = np.sqrt(((Xg - cx) / r) ** 2 + ((Yg - cy) / r) ** 2 + ((Zg - zmid) / ((zc - zf) / 2 + 3)) ** 2)
    noise = 0.06 * np.sin(Xg * 0.37 + Yg * 0.21) * np.cos(Zg * 0.3 + Xg * 0.11)
    inside = (d + noise < 1.0) & (Zg >= zf)
    sub = w.ids[i0:i1, j0:j1, k0:k1]
    sub[:] = np.where(inside, 0, BL.B['deepslate'])
    # a flat floor of deepslate tiles and a low round platform in the middle
    w.fill(int(cx - r), int(cy - r), zf - 1, int(cx + r), int(cy + r), zf - 1, 'deepslate')
    for xi in range(int(cx - 5), int(cx + 6)):
        for yi in range(int(cy - 5), int(cy + 6)):
            if (xi + 0.5 - cx) ** 2 + (yi + 0.5 - cy) ** 2 < 5.2 ** 2:
                w.set(xi, yi, zf - 1, 'deepslate_tiles')
    # soul lanterns in a ring, far apart: the only light down here
    for k in range(8):
        a = k / 8 * 2 * np.pi
        x = int(round(cx + 22 * np.cos(a)))
        y = int(round(cy + 22 * np.sin(a)))
        if k % 2 == 0:
            w.set(x, y, zf, 'soul_lantern')
    for k in range(4):
        a = k / 4 * 2 * np.pi + 0.4
        w.set(int(round(cx + 7 * np.cos(a))), int(round(cy + 7 * np.sin(a))), zf, 'soul_lantern')
    # the ladder: from the pit in the cavern floor straight down to the chamber floor, on a pillar
    px, py = pit
    zcav = CAVE['zf']
    for z in range(zf, zcav):
        w.set(px, py, z, 'air')
        w.set(px, py + 1, z, 'cobbled_deepslate')
        w.set(px, py, z, 'ladder', S)
    w.fill(px - 1, py - 1, zcav - 1, px + 1, py + 1, zcav - 1, 'air')
    for z in range(zcav - 12, zcav):
        for (dx, dy) in ((-1, 0), (1, 0), (0, -1)):
            if w.get(px + dx, py + dy, z) != 'air':
                continue
    w.points['chamber_ladder_foot'] = np.array([px + 0.5, py + 0.5, zf], float)
    w.points['figure'] = np.array([cx + 0.5, cy + 0.5, zf], float)
    w.points['chamber_centre'] = np.array([cx, cy, zf], float)


# ---------------------------------------------------------------------------------------------
# the real house (at the bottom of the crater)
# ---------------------------------------------------------------------------------------------
def real_house(w, ox, oy, oz, r=0):
    b = Builder(w, ox, oy, oz, r)
    W_, D_ = 10, 9
    b.fill(-1, -1, -2, W_ + 1, D_ + 1, -1, 'smooth_stone')
    b.fill(0, 0, 0, W_, D_, 3, 'white_concrete')
    b.fill(1, 1, 0, W_ - 1, D_ - 1, 3, 'air')
    b.fill(-1, -1, 4, W_ + 1, D_ + 1, 4, 'gray_concrete')
    b.fill(0, 0, 5, W_, D_, 5, 'gray_concrete')
    b.set(5, 0, 0, 'spruce_door', S)
    b.set(5, 0, 1, 'spruce_door', S | BL.HALF_TOP)
    for u in (2, 3, 7, 8):
        b.fill(u, 0, 1, u, 0, 2, 'glass_pane')
    b.fill(0, 4, 1, 0, 5, 2, 'glass_pane')
    b.fill(W_, 4, 1, W_, 5, 2, 'glass_pane')
    b.set(3, -1, 2, 'lantern', BL.WALL) if False else None
    real_room_interior(b, 1, 1, W_ - 1, D_ - 1, 4)
    b.point('real_door', 5, -1, 0)
    b.dyn('real_door_bottom', 5, 0, 0)
    b.dyn('real_door_top', 5, 0, 1)
    b.point('real_in', 5, 1, 0)
    return b


# ---------------------------------------------------------------------------------------------
# variants
# ---------------------------------------------------------------------------------------------
HOUSE_O = (-4, -34)                  # the rules house's corner (u 0..8, v 0..10), door at (0, -34)
MAPH_O = (16, 12)                     # map house (rotated to face west)


def village_layout(changed=False):
    """Buildings: (kind, x, y, r, style)."""
    if not changed:
        return [('rules', HOUSE_O[0], HOUSE_O[1], 0, 0), ('well', 2, -2, 0, 0), ('map', MAPH_O[0], MAPH_O[1], 3, 0),
                ('small', -20, -6, 1, 0), ('small', -20, 10, 1, 1), ('small', -14, 22, 0, 2),
                ('small', 16, -24, 3, 3), ('small', 26, -8, 3, 1), ('small', 8, 22, 0, 0),
                ('library', -8, 34, 0, 0), ('smithy', 30, 20, 3, 0), ('tower', 20, 36, 0, 0),
                ('farm', -40, -18, 12, 9), ('farm', -40, 0, 12, 9), ('farm', 32, -40, 10, 8),
                ('pen', 18, -46, 11, 9)]
    # the changed world: the same pieces, moved and turned
    return [('small', HOUSE_O[0] + 9, HOUSE_O[1] + 6, 2, 0), ('well', -12, 8, 0, 0), ('map', -24, -8, 1, 0),
            ('small', 18, -4, 2, 0), ('small', -6, 16, 3, 1), ('small', 24, 20, 1, 2),
            ('small', -24, -26, 0, 3), ('small', 6, 30, 2, 1), ('rules', 10, -12, 1, 0),
            ('library', 30, -26, 3, 0), ('smithy', -30, 26, 0, 0), ('tower', -2, -6, 0, 0),
            ('farm', 34, 2, 9, 12), ('pen', -40, 6, 10, 10)]


def build(variant='village', seed=11):
    rng = np.random.default_rng(seed)
    w = new_world()
    w._copy_z = CAVE['zf']
    changed = variant == 'changed'
    crater = (SPAWN[0], SPAWN[1] + 6.0, 34.0, 36.0) if changed else None
    H, lake = heightmap(w, crater=crater)
    fill_terrain(w, H, lake)
    lay = village_layout(changed)
    blds = []
    for (kind, x, y, r, st) in lay:
        if kind == 'rules':
            if changed:
                b = small_house(w, x, y, r, rng, 2)
            else:
                b = rules_house(w, x, y, r, variant='finale' if variant == 'finale' else 'today', rng=rng)
        elif kind == 'well':
            b = well(w, x, y)
        elif kind == 'map':
            b = map_house(w, x, y, r)
        elif kind == 'small':
            b = small_house(w, x, y, r, rng, st)
        elif kind == 'library':
            b = library(w, x, y, r)
        elif kind == 'smithy':
            b = smithy(w, x, y, r)
        elif kind == 'tower':
            b = tower(w, x, y)
        elif kind == 'farm':
            b = farm(w, x, y, r, st, young=changed)
        elif kind == 'pen':
            b = pen(w, x, y, r, st)
        blds.append((kind, x, y, r, b))
    # paths
    if not changed:
        path(w, [(6, -70), (5, -56), (3, -44), (1, -36)], 3, rng)
        path(w, [(3, -44), (6, -30), (5, -14), (4, -4)], 3, rng)
        path(w, [(4, 4), (4, 20), (8, 30), (21, 34)], 3, rng)
        path(w, [(-22, 2), (-8, 2), (0, 2)], 3, rng)
        path(w, [(8, 2), (15, 6)], 3, rng)
        path(w, [(6, -12), (15, -18), (22, -22)], 2, rng)
        path(w, [(4, -14), (-19, -14)], 2, rng)
        path(w, [(-2, 20), (-8, 30)], 2, rng)
        lamps = [(8, -40), (8, -26), (7, -12), (1, 6), (8, 16), (-12, 4), (12, 8), (6, 28), (-4, -48)]
    else:
        path(w, [(6, -70), (6, -40), (-10, -30), (-12, 0), (-4, 18)], 3, rng)
        path(w, [(-12, 0), (12, -2), (30, -10)], 3, rng)
        path(w, [(12, -2), (24, 14), (10, 30)], 2, rng)
        lamps = [(4, -38), (-14, -12), (10, 2), (22, 16)]
    for (x, y) in lamps:
        lamp_post(w, x, y)
    # keep trees and grass off the village and the spawn
    occupied = np.zeros(w.size[:2], bool)
    X0, Y0 = w.origin[:2]
    for (kind, x, y, r, b) in blds:
        occupied[x - 14 - X0:x + 16 - X0, y - 14 - Y0:y + 16 - Y0] = True

    def keep_out(x, y):
        i, j = int(np.floor(x)) - X0, int(np.floor(y)) - Y0
        if occupied[i, j]:
            return True
        if not changed and x < HOUSE_O[0] and HOUSE_O[1] - 4 < y < HOUSE_O[1] + 12:
            return True             # the low moon's way in through the rules house's west window (act 3)
        return np.hypot(x - SPAWN[0], y - SPAWN[1]) < 10

    if not changed:
        plant_trees(w, H, rng, keep_out=keep_out)
        scatter_plants(w, H, rng, 0.15, keep_out=lambda x, y: False)
    else:
        plant_trees(w, H, rng, kinds=('spruce', 'dead', 'dead', 'spruce'), keep_out=keep_out, density=0.8)
        scatter_plants(w, H, rng, 0.05, flowers=False)
    # the signs of the changed world: LEFT, LEFT, LEFT, then the wrong way
    if changed:
        route = [((-11, -34), 'LEFT'), ((-16, -18), 'LEFT'), ((-6, -44), 'LEFT'),
                 ((2, -46), "YOU'RE GOING THE WRONG WAY")]
        for k, ((x, y), text) in enumerate(route):
            w.fill(x, y, 0, x, y, 0, 'oak_fence')
            lines = [text] if len(text) < 12 else ["YOU'RE GOING", 'THE WRONG', 'WAY']
            w.set(x, y, 1, 'oak_sign', S)
            w.signs.append(dict(pos=(x, y, 1), facing=S, wall=False, lines=lines, key=f'route_{k}'))
            w.points[f'route_{k}'] = np.array([x + 0.5, y + 0.5, 0.0])
        cx, cy, cr, cd = crater
        real_house(w, int(cx) - 5, int(cy) - 4, -int(cd) + 1)
        w.points['crater'] = np.array([cx, cy, 0.0])
    # underground (only the village: the finale never goes down)
    if variant == 'village':
        hf = w._house_frame
        x, y = hf.xy(2, 2)
        y_end, z_end = stair_tunnel(w, x, y + 1, -6, CAVE['zf'])
        pit = cavern(w, rng)
        # connect the stair to the cavern's south wall
        w.fill(x - 1, y_end, CAVE['zf'], x + 1, CAVE['y0'], CAVE['zf'] + 3, 'air')
        w.fill(x - 1, y_end, CAVE['zf'] - 1, x + 1, CAVE['y0'], CAVE['zf'] - 1, 'stone_bricks')
        w.points['stair_top'] = np.array([x + 0.5, y + 1.5, -6.0])
        w.points['stair_bottom'] = np.array([x + 0.5, y_end + 0.5, float(CAVE['zf'])])
        chamber(w, rng, pit)
    w.points['spawn'] = np.array(SPAWN)
    w.heightmap = H
    return w


if __name__ == '__main__':
    import sys
    import time
    t = time.time()
    w = build(sys.argv[1] if len(sys.argv) > 1 else 'village')
    print('built %.1fs' % (time.time() - t), len(w.signs), 'signs', len(w.frames), 'frames', len(w.decals), 'decals')
    for k, v in sorted(w.points.items()):
        print(' ', k, np.round(v, 1))
