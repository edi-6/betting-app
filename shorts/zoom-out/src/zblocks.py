"""The block textures of the spawn area (16x16, drawn here): the engine's first layers in the order the renderer
expects (grass top 0, grass side 2, leaves 7, water 9, birch leaves 12), and the rest for Steve's camp and the
village: cobblestone, glass, doors, farmland and crops, the dirt path, hay, spruce and oak planks for roofs, stone
bricks, the campfire, beds, the crafting table, chest and furnace, wool, flowers in the grass, lava, flames.

The alpha channel marks glowing texels (the renderer's glow material): the campfire's and the torches' flames, lava.
"""
import numpy as np

from textures import make_block_textures

LAYERS = ['grass_top', 'dirt', 'grass_side', 'stone', 'stone2', 'log', 'log_top', 'leaves', 'sand', 'water',
          'planks', 'birch_log', 'birch_leaves', 'gravel', 'snow', 'snow_side', 'crater',
          'cobblestone', 'glass', 'door', 'farmland', 'wheat', 'carrots', 'path', 'hay_top', 'hay_side',
          'crafting_top', 'chest_top', 'furnace_top', 'campfire', 'spruce_planks', 'stone_bricks', 'bell',
          'poppy_grass', 'dandelion_grass', 'cornflower_grass', 'white_wool', 'cow', 'pink', 'bed_red', 'torch',
          'dark_planks', 'mossy_cobble', 'water_still', 'lava', 'flame', 'fence']
L = {n: i for i, n in enumerate(LAYERS)}
RNG = np.random.default_rng


def _speckle(base, var, rng, n=16):
    img = np.ones((n, n, 3)) * np.array(base, float)
    img *= 1.0 + (rng.random((n, n, 1)) - 0.5) * 2 * var
    return img


def _planks(base, rng, dark=0.72):
    p = _speckle(base, 0.06, rng)
    for r in (3, 7, 11, 15):
        p[r] *= dark
    for r0 in (0, 4, 8, 12):
        c = int(rng.integers(2, 14))
        p[r0:r0 + 3, c] *= 0.84
        p[r0 + 1, (c + 6) % 16] *= 0.9
    return p


def _cobble(rng, base=(124, 124, 124), mortar=(78, 78, 80)):
    yy, xx = np.mgrid[0:16, 0:16]
    pts = rng.uniform(0, 16, (11, 2))
    d = np.full((16, 16), 1e9)
    d2 = np.full((16, 16), 1e9)
    lab = np.zeros((16, 16), int)
    for k, (px, py) in enumerate(pts):
        for ox in (-16, 0, 16):
            for oy in (-16, 0, 16):
                dd = np.hypot(xx - px - ox, yy - py - oy)
                closer = dd < d
                d2 = np.where(closer, d, np.minimum(d2, dd))
                d = np.where(closer, dd, d)
                lab = np.where(closer, k, lab)
    shade = 0.82 + 0.3 * rng.random(11)
    img = np.array(base, float)[None, None, :] * shade[lab][..., None]
    img *= 1.0 + (rng.random((16, 16, 1)) - 0.5) * 0.1
    img[(d2 - d) < 1.1] = mortar
    return img


def _grass_flowers(grass, cols, rng, n=3):
    """Flowers seen from above on a grass block: little round blossoms with a darker or yellow middle."""
    g = grass.copy()
    spots = [(3, 3), (10, 5), (6, 11), (12, 12)][:n + 1]
    for k, (x, y) in enumerate(spots[:n]):
        x = int(np.clip(x + rng.integers(-1, 2), 1, 13))
        y = int(np.clip(y + rng.integers(-1, 2), 1, 13))
        c = np.array(cols[k % len(cols)], float)
        g[y + 2, x + 1] = (58, 104, 34)
        for dy, dx in ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)):
            if (dy, dx) in ((0, 0), (2, 2)) and rng.random() < 0.5:
                continue
            g[y + dy, x + dx] = c * (0.82 + 0.18 * rng.random())
        g[y + 1, x + 1] = (240, 200, 60) if c[0] < 200 or c[1] < 120 else c * 0.6
    return g


def world_textures(seed=4):
    rng = RNG(seed)
    t = make_block_textures()
    tex = {k: t[k] for k in ('grass_top', 'dirt', 'grass_side', 'stone', 'stone2', 'log', 'log_top', 'leaves')}
    for k in ('grass_top', 'grass_side', 'leaves'):
        g = tex[k].astype(float)
        grey = g.mean(-1, keepdims=True)
        tex[k] = grey + (g - grey) * 0.78
    tex['grass_top'] = tex['grass_top'] * np.array([0.95, 0.9, 0.9])
    s = _speckle((219, 207, 160), 0.06, rng)
    d = rng.random((16, 16))
    s[d < 0.1] = (200, 188, 140)
    s[d > 0.94] = (232, 222, 180)
    tex['sand'] = s
    yy, xx = np.mgrid[0:16, 0:16]
    w = np.ones((16, 16, 3)) * np.array((44, 92, 196), float)
    ripple = (np.sin((xx + 2 * np.sin(yy * 0.9)) * 0.8) > 0.55)
    w[ripple] = (70, 124, 222)
    w *= 1.0 + (rng.random((16, 16, 1)) - 0.5) * 0.08
    tex['water'] = w
    tex['water_still'] = w
    tex['planks'] = _planks((162, 130, 78), rng)
    b = _speckle((216, 214, 204), 0.05, rng)
    for _ in range(9):
        r, c = int(rng.integers(0, 16)), int(rng.integers(0, 14))
        b[r, c:c + int(rng.integers(2, 4))] = (40, 40, 38)
    tex['birch_log'] = b
    bl = _speckle((104, 150, 64), 0.18, rng)
    d = rng.random((16, 16))
    bl[d < 0.15] = (58, 96, 36)
    bl[d > 0.9] = (140, 186, 92)
    tex['birch_leaves'] = bl
    g = _speckle((128, 122, 118), 0.16, rng)
    d = rng.random((16, 16))
    g[d < 0.18] = (96, 90, 86)
    g[d > 0.85] = (160, 152, 146)
    tex['gravel'] = g
    tex['snow'] = _speckle((238, 244, 250), 0.03, rng)
    ss = tex['dirt'].copy()
    lip = 4 + (rng.random(16) < 0.5).astype(int)
    for c in range(16):
        ss[:lip[c], c] = tex['snow'][:lip[c], c]
    tex['snow_side'] = ss
    tex['crater'] = tex['dirt'] * 0.55
    # the camp and the village
    tex['cobblestone'] = _cobble(rng)
    tex['mossy_cobble'] = _cobble(rng)
    m = rng.random((16, 16)) < 0.3
    tex['mossy_cobble'][m] = tex['mossy_cobble'][m] * 0.5 + np.array([70, 110, 45]) * 0.5
    gl = np.ones((16, 16, 3)) * np.array((196, 222, 232), float)
    gl[0, :] = gl[15, :] = gl[:, 0] = gl[:, 15] = (236, 246, 250)
    for k in range(3, 9):
        gl[k, k + 2] = (250, 252, 255)
        gl[k + 1, k + 2] = (236, 246, 250)
    tex['glass'] = gl
    dr = _planks((150, 112, 64), rng, 0.8)
    dr[2:7, 3:7] = (180, 210, 220)
    dr[2:7, 9:13] = (180, 210, 220)
    dr[10, 12:14] = (60, 60, 60)
    tex['door'] = dr
    fm = _speckle((96, 62, 38), 0.08, rng)
    fm[::3] *= 0.78
    tex['farmland'] = fm
    wh = fm.copy()
    for x in range(0, 16, 2):
        for y in range(16):
            if rng.random() < 0.8:
                wh[y, x] = (200, 178, 70) if rng.random() < 0.7 else (150, 170, 60)
    tex['wheat'] = wh
    ca = fm.copy()
    for x in range(1, 16, 3):
        for y in range(1, 16, 3):
            ca[y:y + 2, x:x + 2] = (66, 140, 44)
            ca[y, x] = (90, 170, 54)
    tex['carrots'] = ca
    pa = _speckle((148, 120, 70), 0.08, rng)
    d = rng.random((16, 16))
    pa[d < 0.15] = (122, 98, 56)
    pa[d > 0.9] = (170, 142, 88)
    tex['path'] = pa
    ht = _speckle((206, 168, 60), 0.08, rng)
    ht[[0, 15]] = ht[:, [0, 15]].mean() * 0 + np.array((150, 104, 40))
    ht[:, [0, 15]] = (150, 104, 40)
    tex['hay_top'] = ht
    hs = _speckle((200, 164, 58), 0.08, rng)
    hs[4:6] = (150, 50, 40)
    hs[10:12] = (150, 50, 40)
    tex['hay_side'] = hs
    ct = _planks((170, 132, 80), rng)
    ct[2:14, 2:14] = ct[2:14, 2:14] * 0.82
    ct[3:13, 7] = (90, 64, 38)
    ct[7, 3:13] = (90, 64, 38)
    tex['crafting_top'] = ct
    ch = _planks((160, 110, 50), rng)
    ch[0, :] = ch[15, :] = ch[:, 0] = ch[:, 15] = (70, 46, 22)
    ch[6:10, 7:9] = (180, 180, 180)
    tex['chest_top'] = ch
    tex['furnace_top'] = _speckle((120, 120, 120), 0.08, rng)
    cf = tex['grass_top'].copy() * 0.6
    for k in range(16):
        cf[k, k] = cf[k, 15 - k] = (96, 70, 40)
        if 0 < k < 15:
            cf[k, k - 1] = cf[k, 16 - k] = (80, 58, 32)
    cf[5:11, 5:11] = (255, 168, 40)
    cf[6:10, 6:10] = (255, 220, 110)
    tex['campfire'] = cf
    tex['spruce_planks'] = _planks((104, 76, 46), rng)
    tex['dark_planks'] = _planks((70, 48, 30), rng)
    sb = _speckle((122, 122, 122), 0.06, rng)
    for r in (0, 4, 8, 12):
        sb[r] = (86, 86, 86)
    for r0, off in ((0, 0), (4, 8), (8, 0), (12, 8)):
        sb[r0:r0 + 4, off % 16] = (86, 86, 86)
        sb[r0:r0 + 4, (off + 8) % 16] = (86, 86, 86) if off == 0 else sb[r0:r0 + 4, (off + 8) % 16]
    tex['stone_bricks'] = sb
    tex['bell'] = _speckle((236, 196, 64), 0.08, rng)
    grass = tex['grass_top']
    tex['poppy_grass'] = _grass_flowers(grass, [(210, 40, 36)], rng)
    tex['dandelion_grass'] = _grass_flowers(grass, [(250, 220, 40)], rng)
    tex['cornflower_grass'] = _grass_flowers(grass, [(80, 110, 230), (110, 140, 240)], rng)
    tex['white_wool'] = _speckle((232, 232, 228), 0.05, rng)
    cw = _speckle((80, 56, 40), 0.08, rng)
    for _ in range(6):
        r, c = int(rng.integers(0, 13)), int(rng.integers(0, 13))
        cw[r:r + 3, c:c + 4] = (226, 222, 214)
    tex['cow'] = cw
    tex['pink'] = _speckle((236, 160, 156), 0.05, rng)
    bd = _speckle((176, 36, 36), 0.05, rng)
    bd[:5] = (232, 232, 226)
    tex['bed_red'] = bd
    to = np.ones((16, 16, 3)) * np.array((110, 82, 50), float)
    to[5:11, 5:11] = (255, 220, 90)
    to[6:10, 6:10] = (255, 250, 200)
    tex['torch'] = to
    lv = _speckle((214, 84, 20), 0.08, rng)
    blob = rng.random((16, 16))
    lv[blob > 0.72] = (246, 150, 40)
    lv[blob > 0.92] = (255, 214, 96)
    tex['lava'] = lv
    fl = np.ones((16, 16, 3)) * np.array((255, 170, 50), float)
    fl[3:13, 3:13] = (255, 214, 110)
    fl[5:11, 5:11] = (255, 246, 196)
    tex['flame'] = fl
    fe = _planks((156, 124, 74), rng)
    tex['fence'] = fe
    glow = {n: np.zeros((16, 16)) for n in LAYERS}
    glow['lava'][:] = 150
    glow['flame'][:] = 150
    glow['torch'][5:11, 5:11] = 150
    glow['campfire'][5:11, 5:11] = 150
    out = []
    for n in LAYERS:
        rgb = np.clip(tex[n], 0, 255)
        out.append(np.concatenate([rgb, glow[n][..., None]], -1).astype(np.uint8))
    return out
