"""The pixel art, drawn in code: 16 x 16 block textures (RGB, plus a glow mask), sprites with transparency (torches,
rails, cobwebs, crystals, lanterns, sculk sensors), the mining cracks, item icons, and the characters seen from the
side (Steve with his pickaxe, a zombie, cave spiders, the Warden)."""
import numpy as np

N = 16


def _rng(seed):
    return np.random.default_rng(seed)


def noise(rng, base, var, coarse=0.4, n=N):
    fine = rng.random((n, n)) - 0.5
    blob = np.kron(rng.random((n // 4, n // 4)) - 0.5, np.ones((4, 4)))
    k = fine * (1 - coarse) + blob * coarse
    return np.clip(np.array(base, float)[None, None] * (1 + 2 * var * k[..., None]), 0, 255)


def _specks(img, rng, col, p, size=1):
    m = rng.random((N, N)) < p
    if size > 1:
        m2 = m.copy()
        m2[1:, :] |= m[:-1, :]
        m2[:, 1:] |= m[:, :-1]
        m = m2
    img[m] = col
    return img


# ---------------------------------------------------------------------------------------------
# blocks
# ---------------------------------------------------------------------------------------------
def stone(rng):
    img = noise(rng, (126, 126, 126), 0.08, 0.5)
    _specks(img, rng, (104, 104, 104), 0.10, 2)
    _specks(img, rng, (142, 142, 142), 0.05)
    return img


def deepslate(rng):
    img = noise(rng, (104, 104, 112), 0.09, 0.3)
    for r in range(1, N, 3):
        img[r, :] *= 0.84 + 0.1 * rng.random((N, 1))
    _specks(img, rng, (68, 68, 76), 0.08)
    return img


def dirt(rng):
    img = noise(rng, (134, 96, 67), 0.10, 0.5)
    _specks(img, rng, (102, 72, 50), 0.12)
    _specks(img, rng, (160, 118, 84), 0.05)
    _specks(img, rng, (120, 120, 116), 0.02)
    return img


def grass_side(rng):
    img = dirt(rng)
    g = noise(rng, (100, 160, 60), 0.10, 0.3)
    depth = 3 + (rng.random(N) < 0.5).astype(int) + (rng.random(N) < 0.25).astype(int)
    for c in range(N):
        img[:depth[c], c] = g[:depth[c], c]
    img[0, :] = img[0, :] * 1.12
    return img


def ore(base, rng, col, hi, n=4, lo=None):
    """Rock with chunky ore nuggets: each a little blob with a lit corner and a dark edge."""
    img = base.copy()
    glow = np.zeros((N, N))
    lo = lo if lo is not None else tuple(int(v * 0.55) for v in col)
    shapes = [((0, 0), (0, 1), (1, 0), (1, 1), (0, 2)), ((0, 0), (0, 1), (1, 1), (1, 2), (2, 1)),
              ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)), ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2))]
    spots = [(2, 2), (2, 10), (8, 5), (10, 11), (6, 12), (12, 2)]
    rng.shuffle(spots)
    for (r, c) in spots[:n]:
        r, c = r + int(rng.integers(-1, 2)), c + int(rng.integers(-1, 2))
        sh = shapes[int(rng.integers(len(shapes)))]
        for k, (dr, dc) in enumerate(sh):
            rr, cc = int(np.clip(r + dr, 0, 15)), int(np.clip(c + dc, 0, 15))
            img[rr, cc] = hi if k == 0 else col
            glow[rr, cc] = 1.0
        # a dark edge under the nugget
        for (dr, dc) in sh:
            rr, cc = r + dr + 1, c + dc
            if 0 <= rr < N and 0 <= cc < N and glow[rr, cc] == 0:
                img[rr, cc] = lo
    return img, glow


def granite(rng):
    img = noise(rng, (154, 106, 89), 0.08, 0.5)
    _specks(img, rng, (176, 128, 110), 0.08, 2)
    _specks(img, rng, (120, 80, 66), 0.06)
    return img


def diorite(rng):
    img = noise(rng, (190, 190, 192), 0.06, 0.4)
    _specks(img, rng, (120, 120, 124), 0.10)
    _specks(img, rng, (230, 230, 232), 0.06)
    return img


def andesite(rng):
    img = noise(rng, (136, 136, 138), 0.06, 0.6)
    _specks(img, rng, (112, 112, 114), 0.12, 2)
    _specks(img, rng, (160, 160, 162), 0.05)
    return img


def gravel(rng):
    img = noise(rng, (128, 124, 122), 0.12, 0.3)
    for _ in range(14):
        r, c = rng.integers(0, 15, 2)
        v = rng.choice([86, 100, 150, 170])
        img[r:r + 2, c:c + 2] = (v, v - 4, v - 6)
    return img


def tuff(rng):
    img = noise(rng, (108, 109, 102), 0.07, 0.5)
    _specks(img, rng, (90, 92, 86), 0.1, 2)
    return img


def calcite(rng):
    img = noise(rng, (222, 224, 220), 0.03, 0.5)
    _specks(img, rng, (200, 202, 198), 0.08)
    return img


def smooth_basalt(rng):
    img = noise(rng, (72, 72, 80), 0.06, 0.6)
    _specks(img, rng, (60, 60, 66), 0.05)
    return img


def amethyst(rng):
    img = noise(rng, (136, 96, 200), 0.06, 0.4)
    for k in range(-16, 16, 5):
        for c in range(N):
            r = c + k
            if 0 <= r < N:
                img[r, c] = (176, 136, 236)
    _specks(img, rng, (98, 66, 160), 0.08)
    return img


def budding(rng):
    img = amethyst(rng)
    for _ in range(4):
        r, c = rng.integers(2, 13, 2)
        img[r:r + 2, c:c + 2] = (70, 40, 120)
    return img


def planks(rng, base=(162, 130, 78)):
    img = noise(rng, base, 0.06, 0.3)
    for r in (3, 7, 11, 15):
        img[r] *= 0.7
    for r0 in (0, 4, 8, 12):
        c = int(rng.integers(1, 15))
        img[r0:r0 + 3, c] *= 0.8
    return img


def log_side(rng):
    img = noise(rng, (104, 82, 50), 0.08, 0.2)
    for c in range(0, N, 3):
        img[:, c] *= 0.78
    return img


def leaves(rng):
    img = noise(rng, (60, 120, 40), 0.18, 0.3)
    _specks(img, rng, (40, 86, 26), 0.2)
    _specks(img, rng, (90, 150, 60), 0.08)
    return img


def deepslate_bricks(rng):
    img = noise(rng, (70, 70, 76), 0.06, 0.3)
    for r in (0, 4, 8, 12):
        img[r, :] = (44, 44, 48)
    for r0, cs in ((0, (0, 8)), (4, (4, 12)), (8, (0, 8)), (12, (4, 12))):
        for c in cs:
            img[r0:r0 + 4, c] = (44, 44, 48)
    return img


def deepslate_tiles(rng):
    img = noise(rng, (56, 56, 60), 0.06, 0.3)
    for r in (0, 8):
        img[r, :] = (34, 34, 38)
    for c in (0, 8):
        img[:, c] = (34, 34, 38)
    return img


def sculk(rng):
    img = noise(rng, (14, 34, 42), 0.25, 0.4)
    glow = np.zeros((N, N))
    for _ in range(9):
        r, c = rng.integers(0, 16, 2)
        img[r, c] = (40, 220, 220)
        glow[r, c] = 1.0
    _specks(img, rng, (8, 20, 26), 0.15)
    return img, glow


def bedrock(rng):
    img = noise(rng, (86, 86, 86), 0.35, 0.55)
    img[rng.random((N, N)) < 0.18] = (40, 40, 40)
    img[rng.random((N, N)) > 0.9] = (150, 150, 150)
    return img


def obsidian(rng):
    img = noise(rng, (22, 16, 36), 0.25, 0.5)
    _specks(img, rng, (60, 40, 90), 0.05)
    return img


def spawner_block(rng):
    img = np.zeros((N, N, 3)) + (24, 30, 36)
    for k in (0, 4, 8, 12, 15):
        img[k, :] = (40, 56, 70)
        img[:, k] = (40, 56, 70)
    return img


def lava_frames(n=32, seed=5):
    """Animated lava (it flows downwards slowly): n frames of RGB."""
    rng = _rng(seed)
    big = rng.random((64, 16))
    from scipy.ndimage import gaussian_filter
    big = gaussian_filter(big, (2.0, 1.2), mode='wrap')
    big = (big - big.min()) / (big.max() - big.min())
    frames = []
    for f in range(n):
        sh = int(round(f * 64 / n))
        v = np.roll(big, sh, 0)[:16]
        col = np.zeros((N, N, 3))
        col[:] = (210, 82, 14)
        col[v > 0.45] = (238, 120, 24)
        col[v > 0.62] = (252, 170, 50)
        col[v > 0.8] = (255, 222, 110)
        frames.append(col)
    return frames


def water_frames(n=32, seed=6):
    rng = _rng(seed)
    from scipy.ndimage import gaussian_filter
    big = gaussian_filter(rng.random((64, 16)), (1.5, 1.0), mode='wrap')
    big = (big - big.min()) / (big.max() - big.min())
    frames = []
    for f in range(n):
        v = np.roll(big, int(round(f * 64 / n)), 0)[:16]
        col = np.zeros((N, N, 3))
        col[:] = (44, 84, 196)
        col[v > 0.6] = (66, 110, 220)
        col[v > 0.82] = (110, 150, 240)
        frames.append(col)
    return frames


# ---------------------------------------------------------------------------------------------
# sprites (RGBA, alpha 0 = see-through); drawn from character maps
# ---------------------------------------------------------------------------------------------
def from_rows(rows, pal):
    h, w = len(rows), max(len(r) for r in rows)
    img = np.zeros((h, w, 4), np.float32)
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch in pal:
                img[r, c, :3] = pal[ch]
                img[r, c, 3] = 255
    return img


TORCH = ["......", "..YY..", ".YWWY.", ".YWWY.", "..OO..", "..bb..", "..bb..", "..bb..", "..bb..", "..bb..",
         "..bb.."]
TORCH_PAL = {'Y': (255, 200, 60), 'W': (255, 250, 210), 'O': (230, 120, 30), 'b': (110, 80, 44)}
LANTERN = ["..kk..", ".k..k.", "kkkkkk", "kCWWCk", "kCWWCk", "kCCCCk", "kkkkkk"]
LANTERN_PAL = {'k': (40, 44, 52), 'C': (70, 210, 230), 'W': (200, 255, 255)}
COBWEB = ["#...........#...", ".#....#....#....", "..#...#...#.....", "...#..#..#......", "....#.#.#.......",
          "######*#########", "....#.#.#.......", "...#..#..#......", "..#...#...#.....", ".#....#....#....",
          "#.....#.....#...", "......#......#..", "......#.......#.", "....................", "......#.........",
          "......#........."]
WEB_PAL = {'#': (230, 230, 236), '*': (255, 255, 255)}
CLUSTER = ["................", "................", "......c.........", ".....cC....c....", ".....cC...cC....",
           "..c..cCc..cC....", "..cC.cCc.cCc..c.", "..cCccCcccCc.cC.", ".ccCcCCccCCc.cC.", ".cCCcCCcCCCccCC."]
CLUSTER_PAL = {'c': (190, 150, 250), 'C': (236, 210, 255)}
CANDLE = [".Y.", ".W.", "ccc", "ccc", "ccc", "ccc"]
CANDLE_PAL = {'Y': (255, 210, 90), 'W': (255, 250, 220), 'c': (210, 200, 180)}
SENSOR = ["..t..........t..", ".tCt........tCt.", "..t..........t..", "..t..........t..", "SSSSSSSSSSSSSSSS",
          "SsSSsSSSSsSSsSSS", "SSSSSSSSSSSSSSSS", "kkkkkkkkkkkkkkkk"]
SENSOR_PAL = {'t': (20, 90, 100), 'C': (90, 255, 255), 'S': (20, 70, 84), 's': (60, 200, 210), 'k': (14, 30, 38)}
RAIL = ["................", "................", "................", "................", "................",
        "................", "................", "................", "................", "................",
        "................", "................", "rrrrrrrrrrrrrrrr", "wwwwwwwwwwwwwwww", ".w..w..w..w..w..",
        "................"]
RAIL_PAL = {'r': (150, 150, 156), 'w': (110, 82, 50)}
FENCE = ["......pp......", "......pp......", "......pp......", "......pp......", "......pp......",
         "......pp......", "......pp......", "......pp......", "......pp......", "......pp......",
         "......pp......", "......pp......", "......pp......", "......pp......", "......pp......",
         "......pp......"]
FENCE_PAL = {'p': (162, 130, 78)}
LICHEN = ["..g...........g.", ".gGg.......g.gGg", "..g.......gGg.g.", "...........g....", "................",
          ".......g........", "......gGg.......", ".......g........"]
LICHEN_PAL = {'g': (120, 160, 120), 'G': (220, 250, 200)}
DIAMOND = ["................", "................", "......oooo......", ".....oHHcco.....", "....oHwcccco....",
           "...oHwcccccCo...", "...oHcccccCCo...", "...ocwcccCCCo...", "....occccCCo....", ".....occCCo.....",
           "......oCCo......", ".......oo.......", "................", "................", "................",
           "................"]
DIAMOND_PAL = {'o': (14, 50, 50), 'H': (232, 255, 255), 'w': (168, 248, 240), 'c': (74, 222, 208),
               'C': (36, 160, 150)}
COAL = ["................", "................", "................", "......kk........", "....kkKkk.......",
        "...kKKkkkk......", "...kkkkkKk......", "....kkKkkk......", ".....kkkk.......", "................"]
COAL_PAL = {'k': (36, 36, 40), 'K': (80, 80, 86)}
PICKAXE = [".......cccc.....", "......cHccCc....", "......c....cc...", ".........bb.c...", "........bb...c..",
           ".......bb....c..", "......bb.....c..", ".....bb.........", "....bb..........", "...bb...........",
           "..bb............", ".bb.............", "bb..............", "b..............."]
PICK_PAL = {'c': (90, 220, 210), 'H': (220, 255, 255), 'C': (40, 150, 140), 'b': (110, 80, 44)}
IRON_PICK_PAL = {'c': (200, 200, 204), 'H': (255, 255, 255), 'C': (130, 130, 136), 'b': (110, 80, 44)}


CHEST = ["................", "................", ".oooooooooooooo.", ".ollllllllllllo.", ".ollllllllllllo.",
         ".ollllllllllllo.", ".oooooooggooooo.", ".oddddddggddddo.", ".odddddddddddddo", ".odddddddddddddo",
         ".odddddddddddddo", ".odddddddddddddo", ".odddddddddddddo", ".oooooooooooooo."]
CHEST_PAL = {'o': (60, 36, 14), 'l': (176, 120, 50), 'd': (150, 98, 40), 'g': (200, 200, 200)}
SHRIEKER = ["................", "................", "....bb....bb....", "...bWWb..bWWb...", "...bWb....bWb...",
            "..bbWbbbbbbWbb..", "..bWWWWWWWWWWb..", "..bWddddddddWb..", "..bWdCCCCCCdWb..", "..bWddddddddWb..",
            "SSSSSSSSSSSSSSSS", "SsSSSsSSSSsSSSsS", "SSSSSSSSSSSSSSSS", "kkkkkkkkkkkkkkkk"]
SHRIEKER_PAL = {'b': (150, 140, 120), 'W': (226, 218, 196), 'd': (30, 40, 44), 'C': (90, 255, 255),
                'S': (20, 70, 84), 's': (60, 200, 210), 'k': (14, 30, 38)}


def sprite(name):
    table = {'torch': (TORCH, TORCH_PAL), 'lantern': (LANTERN, LANTERN_PAL), 'cobweb': (COBWEB, WEB_PAL),
             'cluster': (CLUSTER, CLUSTER_PAL), 'candle': (CANDLE, CANDLE_PAL), 'sensor': (SENSOR, SENSOR_PAL),
             'rail': (RAIL, RAIL_PAL), 'fence': (FENCE, FENCE_PAL), 'lichen': (LICHEN, LICHEN_PAL),
             'diamond': (DIAMOND, DIAMOND_PAL), 'coal': (COAL, COAL_PAL), 'pickaxe': (PICKAXE, IRON_PICK_PAL),
             'chest': (CHEST, CHEST_PAL), 'shrieker': (SHRIEKER, SHRIEKER_PAL)}
    rows, pal = table[name]
    return from_rows(rows, pal)


GLOWING_SPRITES = {'torch': 1.0, 'lantern': 1.0, 'candle': 0.8, 'cluster': 0.5, 'sensor': 0.6, 'lichen': 0.6,
                   'shrieker': 0.5}


def cracks(seed=9):
    """The ten stages of a block being mined: (10, 16, 16) darkness masks."""
    rng = _rng(seed)
    order = np.zeros((N, N))
    # grow cracks from the middle along random walks; each pixel gets the stage it appears at
    order[:] = 99
    walkers = [(8.0, 8.0, a) for a in rng.uniform(0, 2 * np.pi, 5)]
    step = 0
    while step < 120:
        for k, (r, c, a) in enumerate(walkers):
            a += rng.normal(0, 0.6)
            r, c = r + np.sin(a), c + np.cos(a)
            if 0 <= r < N and 0 <= c < N:
                rr, cc = int(r), int(c)
                order[rr, cc] = min(order[rr, cc], step / 12.0)
            else:
                r, c, a = 8.0 + rng.normal(0, 2), 8.0 + rng.normal(0, 2), rng.uniform(0, 6.28)
            walkers[k] = (r, c, a)
        step += 1
    out = np.zeros((10, N, N))
    for s in range(10):
        out[s] = (order <= s * 1.0).astype(float) * 0.55
    return out


# ---------------------------------------------------------------------------------------------
# characters, from the side (facing right); parts are separate so they can swing
# ---------------------------------------------------------------------------------------------
STEVE_HEAD = ["HHHHHHHH", "HHHHHHHH", "HHHHHSSH", "HHSSSSSS", "HHSSSWPS", "HSSSSSSN", "HSSSSMMS", "HSSSSSSS"]
STEVE_PAL = {'H': (64, 42, 26), 'S': (200, 146, 104), 'W': (250, 250, 250), 'P': (78, 60, 140),
             'N': (150, 96, 62), 'M': (110, 66, 46), 'T': (22, 178, 182), 't': (14, 140, 146),
             'J': (58, 56, 160), 'j': (44, 42, 130), 'O': (100, 100, 104)}
STEVE_BODY = ["TTTT"] * 11 + ["tttt"]
STEVE_ARM = ["TTTT"] * 4 + ["SSSS"] * 8
STEVE_LEG = ["JJJJ"] * 9 + ["jjjj", "OOOO", "OOOO"]
ZOMBIE_PAL = {'H': (44, 96, 44), 'S': (90, 150, 80), 'W': (20, 30, 20), 'P': (20, 30, 20), 'N': (70, 120, 60),
              'M': (40, 80, 40), 'T': (22, 150, 160), 't': (14, 110, 120), 'J': (60, 50, 140), 'j': (46, 38, 110),
              'O': (60, 60, 64)}
ZOMBIE_HEAD = ["HHHHHHHH", "HHSSHSSH", "HSSSSSSS", "HSSSSSSS", "HSSSSWWS", "HSSSSSSN", "HSSSMMMS", "HSSSSSSS"]
WARDEN_PAL = {'d': (30, 74, 88), 'D': (20, 52, 64), 'g': (40, 210, 220), 'G': (160, 255, 250), 'b': (40, 98, 114),
              'r': (84, 146, 156)}
WARDEN = [  # 16 wide, 46 tall side view: horn-like ears, glowing ribcage
    "..gg......gg....", ".gGg......gGg...", "..ddddddddddd...", "..ddddddddddd...", "..ddddddddddd...",
    "..dddddddddddd..", "..ddddddddddddd.", "..ddddddddddDDd.", "...ddddddddddd..", "...ddddddddddd..",
    "....bbbbbbbbb...", "...bbbbbbbbbbb..", "..bbbbbbbbbbbbb.", "..bbgbbbbbbgbbb.", "..bbgGgbbbgGgbb.",
    "..bbbgbbbbbgbbb.", "..bbbbgGGGgbbbb.", "..bbbbbgGgbbbbb.", "..bbbbbbbbbbbbb.", "..bbbbbbbbbbbbb.",
    "..rbbbbbbbbbbbr.", "..rrbbbbbbbbbrr.", "...rbbbbbbbbbr..", "...bbbbbbbbbbb..", "...bbbbbbbbbbb..",
    "...bbbb...bbbb..", "...bbbb...bbbb..", "...bbbb...bbbb..", "...bbbb...bbbb..", "...bbbb...bbbb..",
    "...bbbb...bbbb..", "...DDDD...DDDD..", "...DDDD...DDDD..", "..DDDDD...DDDDD."]
SPIDER = ["................", "....r.r.........", ".kkkkkkkkk......", "kkkkkkkkkkkkk...", "kkRkkRkkkkkkkk..",
          "kkkkkkkkkkkkkk..", ".kkkkkkkkkkkk...", "k.k.k.k.k.k.k...", "k..k..k..k..k...", "k...k...k...k..."]
SPIDER_PAL = {'k': (20, 44, 52), 'R': (220, 30, 30), 'r': (60, 90, 100)}
BAT = ["k..........k", "kk...kk...kk", "kkk.kkkk.kkk", ".kkkkRRkkkk.", "..kkkkkkkk..", "....kkkk...."]
BAT_PAL = {'k': (60, 48, 40), 'R': (200, 60, 60)}


def steve_parts(pal=None, head=None):
    pal = pal or STEVE_PAL
    return {'head': from_rows(head or STEVE_HEAD, pal), 'body': from_rows(STEVE_BODY, pal),
            'arm': from_rows(STEVE_ARM, pal), 'leg': from_rows(STEVE_LEG, pal)}


def characters():
    return {'steve': steve_parts(), 'zombie': steve_parts(ZOMBIE_PAL, ZOMBIE_HEAD),
            'warden': from_rows(WARDEN, WARDEN_PAL), 'spider': from_rows(SPIDER, SPIDER_PAL),
            'bat': from_rows(BAT, BAT_PAL), 'pickaxe': sprite('pickaxe')}


# ---------------------------------------------------------------------------------------------
# the block table
# ---------------------------------------------------------------------------------------------
BLOCKS = ['air', 'grass', 'dirt', 'stone', 'granite', 'diorite', 'andesite', 'gravel', 'coal_ore', 'iron_ore',
          'copper_ore', 'gold_ore', 'redstone_ore', 'lapis_ore', 'diamond_ore', 'deepslate', 'deep_iron',
          'deep_gold', 'deep_redstone', 'deep_diamond', 'deep_lapis', 'tuff', 'calcite', 'basalt', 'amethyst',
          'budding', 'planks', 'log', 'leaves', 'deep_bricks', 'deep_tiles', 'sculk', 'bedrock', 'obsidian',
          'spawner', 'water', 'lava', 'deep_coal']
B = {n: i for i, n in enumerate(BLOCKS)}
ORE_GLOW = {'redstone_ore': 0.6, 'deep_redstone': 0.6, 'diamond_ore': 0.25, 'deep_diamond': 0.25}


def block_textures(seed=4):
    """(n_blocks, 16, 16, 3) float RGB and (n_blocks, 16, 16) glow."""
    rng = _rng(seed)
    st = stone(rng)
    ds = deepslate(rng)
    tex, glow = {}, {}
    tex['air'] = np.zeros((N, N, 3))
    tex['grass'] = grass_side(rng)
    tex['dirt'] = dirt(rng)
    tex['stone'] = st
    tex['granite'] = granite(rng)
    tex['diorite'] = diorite(rng)
    tex['andesite'] = andesite(rng)
    tex['gravel'] = gravel(rng)
    tex['coal_ore'], _ = ore(st, rng, (44, 44, 46), (70, 70, 72), 4, (24, 24, 26))
    tex['iron_ore'], _ = ore(st, rng, (216, 175, 147), (236, 206, 182), 4)
    tex['copper_ore'], _ = ore(st, rng, (224, 128, 98), (120, 200, 160), 4)
    tex['gold_ore'], _ = ore(st, rng, (252, 220, 70), (255, 250, 170), 4)
    tex['redstone_ore'], glow['redstone_ore'] = ore(st, rng, (230, 20, 20), (255, 120, 120), 5)
    tex['lapis_ore'], _ = ore(st, rng, (30, 70, 190), (90, 130, 230), 5)
    tex['diamond_ore'], glow['diamond_ore'] = ore(st, rng, (92, 236, 230), (220, 255, 255), 5, (30, 150, 150))
    tex['deepslate'] = ds
    tex['deep_coal'], _ = ore(ds, rng, (30, 30, 32), (60, 60, 62), 4, (16, 16, 18))
    tex['deep_iron'], _ = ore(ds, rng, (216, 175, 147), (236, 206, 182), 4)
    tex['deep_gold'], _ = ore(ds, rng, (252, 220, 70), (255, 250, 170), 4)
    tex['deep_redstone'], glow['deep_redstone'] = ore(ds, rng, (230, 20, 20), (255, 120, 120), 5)
    tex['deep_diamond'], glow['deep_diamond'] = ore(ds, rng, (92, 236, 230), (220, 255, 255), 5, (30, 150, 150))
    tex['deep_lapis'], _ = ore(ds, rng, (30, 70, 190), (90, 130, 230), 5)
    tex['tuff'] = tuff(rng)
    tex['calcite'] = calcite(rng)
    tex['basalt'] = smooth_basalt(rng)
    tex['amethyst'] = amethyst(rng)
    tex['budding'] = budding(rng)
    tex['planks'] = planks(rng)
    tex['log'] = log_side(rng)
    tex['leaves'] = leaves(rng)
    tex['deep_bricks'] = deepslate_bricks(rng)
    tex['deep_tiles'] = deepslate_tiles(rng)
    tex['sculk'], glow['sculk'] = sculk(rng)
    tex['bedrock'] = bedrock(rng)
    tex['obsidian'] = obsidian(rng)
    tex['spawner'] = spawner_block(rng)
    tex['water'] = water_frames(1)[0]
    tex['lava'] = lava_frames(1)[0]
    T = np.stack([tex[n] for n in BLOCKS]).astype(np.float32)
    G = np.stack([glow[n] * ORE_GLOW.get(n, 0.8) if n in glow else np.zeros((N, N)) for n in BLOCKS]).astype(
        np.float32)
    G[B['lava']] = 1.0
    return T, G


def item_icon(T, b):
    """A dropped block: its texture at half size with a dark edge."""
    t = T[b][::2, ::2].copy()
    img = np.zeros((8, 8, 4), np.float32)
    img[..., :3] = t
    img[..., 3] = 255
    img[0, :, :3] *= 1.15
    img[-1, :, :3] *= 0.6
    img[:, -1, :3] *= 0.7
    return img
