"""Procedural 16x16 pixel art in the Minecraft style, drawn from scratch in code: the nine blocks that go under the
press, their insides (what a crushed block is made of), the press itself (iron, steel, chrome, hazard stripes)
and the studio (deepslate tiles, stone bricks, redstone lamps). Plus the loot sprites that burst out of the chest
and isometric inventory icons for the hotbar."""
import numpy as np
from PIL import Image

from textures import make_block_textures
import tnt as TN

N = 16


def _rng(seed):
    return np.random.default_rng(seed)


def _noise(rng, base, var, coarse=0.4):
    fine = rng.random((N, N)) - 0.5
    blob = np.kron(rng.random((N // 4, N // 4)) - 0.5, np.ones((4, 4)))
    k = fine * (1 - coarse) + blob * coarse
    return np.clip(np.array(base, float)[None, None] * (1 + 2 * var * k[..., None]), 0, 255)


def _rgba(img, glow=None):
    a = np.zeros((N, N, 1)) if glow is None else glow[..., None] * 255.0
    return np.concatenate([np.clip(img, 0, 255), a], -1)


# ---------------------------------------------------------------------------------------------
# the nine test blocks: (top, side, bottom) faces
# ---------------------------------------------------------------------------------------------
GLINTS = ((3, 5, 3), (3, 7, 2), (10, 13, 3))


def glass_alpha():
    """Which pixels of the glass texture are opaque: the frame and the glints (the rest is see-through)."""
    a = np.zeros((N, N), bool)
    a[0, :] = a[-1, :] = a[:, 0] = a[:, -1] = True
    a[1, 1:-1] = a[1:-1, 1] = a[-2, 1:-1] = a[1:-1, -2] = True
    for (r0, c0, L) in GLINTS:
        for k in range(L):
            a[r0 + k, c0 - k] = True
    return a


def glass(rng):
    img = np.full((N, N, 3), (206, 232, 240), float)
    img *= (1 + (rng.random((N, N, 1)) - 0.5) * 0.04)
    # frame
    img[0, :] = img[-1, :] = img[:, 0] = img[:, -1] = (236, 244, 248)
    img[1, 1:-1] = img[1:-1, 1] = (250, 252, 252)
    img[-2, 1:-1] = img[1:-1, -2] = (178, 206, 218)
    # the diagonal glints
    for (r0, c0, L) in GLINTS:
        for k in range(L):
            img[r0 + k, c0 - k] = (255, 255, 255)
    return img


def melon_side(rng):
    img = np.zeros((N, N, 3))
    for c in range(N):
        stripe = (c % 4) in (0, 1)
        img[:, c] = (92, 154, 36) if stripe else (138, 186, 50)
    img = img * (1 + (rng.random((N, N, 1)) - 0.5) * 0.12)
    img[rng.random((N, N)) < 0.06] *= 0.8
    return img


def melon_top(rng):
    img = _noise(rng, (120, 172, 46), 0.08, 0.3)
    yy, xx = np.mgrid[0:N, 0:N]
    ang = np.arctan2(yy - 7.5, xx - 7.5)
    img[(np.sin(ang * 5) > 0.55)] *= 0.78
    img[6:10, 6:10] = (96, 130, 40)
    img[7:9, 7:9] = (74, 96, 30)
    return img


def slime_face(rng):
    img = np.full((N, N, 3), (120, 200, 96), float)
    img *= (1 + (rng.random((N, N, 1)) - 0.5) * 0.06)
    img[0, :] = img[-1, :] = img[:, 0] = img[:, -1] = (150, 222, 122)
    img[1:-1, 1] = img[1, 1:-1] = (138, 214, 112)
    # the inner cube you can see through the jelly
    img[4:12, 4:12] = (86, 162, 70)
    img[5:11, 5:11] = (96, 176, 78)
    img[4, 4:12] = img[4:12, 4] = (70, 138, 56)
    img[3, 3:6] = (190, 240, 170)
    img[3:5, 3] = (190, 240, 170)
    return img


def planks(rng, base=(160, 122, 72)):
    img = _noise(rng, base, 0.05, 0.2)
    for r in (3, 7, 11, 15):
        img[r, :] *= 0.72
    for r0, c in ((0, 5), (4, 12), (8, 3), (12, 9)):
        img[r0:r0 + 3, c] *= 0.8
    return img


def chest_side(rng, front=False):
    img = planks(rng, (158, 110, 52))
    img[0, :] = img[-1, :] = img[:, 0] = img[:, -1] = (64, 42, 20)
    img[5, :] = (54, 36, 18)                       # the lid seam
    img[4, 1:-1] *= 0.8
    if front:
        img[4:8, 7:9] = (200, 204, 210)            # the latch
        img[4:8, 7] = (160, 164, 170)
        img[7, 7:9] = (120, 122, 128)
        img[5, 7:9] = (228, 230, 236)
    return img


def chest_top(rng):
    img = planks(rng, (166, 118, 58))
    img[0, :] = img[-1, :] = img[:, 0] = img[:, -1] = (64, 42, 20)
    img[1, 1:-1] *= 1.08
    return img


def diamond_block(rng):
    img = np.full((N, N, 3), (94, 220, 214), float)
    img *= (1 + (rng.random((N, N, 1)) - 0.5) * 0.08)
    img[0, :] = img[:, 0] = (180, 250, 246)
    img[-1, :] = img[:, -1] = (44, 150, 150)
    # facets
    for r in range(2, 14):
        for c in range(2, 14):
            if (r + c) % 7 == 0 or (r - c) % 9 == 0:
                img[r, c] = (150, 244, 238)
    for (r, c) in ((3, 3), (4, 10), (9, 5), (11, 12), (7, 8)):
        img[r, c] = (240, 255, 255)
        img[r + 1, c] = (200, 252, 250)
    img[13, 2:14] = (60, 176, 172)
    img[2:14, 13] = (60, 176, 172)
    return img


def obsidian(rng):
    img = _noise(rng, (36, 22, 56), 0.3, 0.45)
    d = rng.random((N, N))
    img[d < 0.10] = (16, 10, 26)
    img[(d > 0.84) & (d < 0.93)] = (80, 52, 122)
    img[d > 0.96] = (132, 96, 186)
    for _ in range(3):
        r, c = rng.integers(0, N, 2)
        L = rng.integers(2, 5)
        img[r, c:c + L] = (96, 64, 142)
    return img


def bedrock(rng):
    img = _noise(rng, (118, 118, 118), 0.3, 0.55)
    d = rng.random((N, N))
    img[d < 0.16] = (50, 50, 50)
    img[(d > 0.16) & (d < 0.25)] = (78, 78, 78)
    img[d > 0.86] = (176, 176, 176)
    blob = np.kron(rng.random((4, 4)), np.ones((4, 4)))
    img[blob > 0.8] *= 0.62
    return img


def make_items(seed=11):
    """kind -> dict(faces=(top, side, bottom[, front]), inner=[palette], inner_p=[weights], ...)"""
    rng = _rng(seed)
    bt = make_block_textures()
    tnt_side, tnt_top, tnt_bot = TN.side(rng), TN.top(rng), TN.top(rng, bottom=True)
    sl = slime_face(rng)
    gl = glass(rng)
    dia = diamond_block(rng)
    obs = obsidian(rng)
    bed = bedrock(rng)
    items = {
        'grass': dict(label='Grass Block', faces=(bt['grass_top'], bt['grass_side'], bt['dirt']),
                      inner=[(136, 96, 64), (118, 82, 54), (152, 110, 74), (98, 68, 44)], inner_p=[0.4, 0.3, 0.2, 0.1]),
        'glass': dict(label='Glass', faces=(gl, gl, gl), alpha=glass_alpha(),
                      inner=[(214, 236, 244), (236, 246, 250), (190, 222, 234)], inner_p=[0.5, 0.3, 0.2]),
        'melon': dict(label='Melon', faces=(melon_top(rng), melon_side(rng), melon_top(rng)),
                      inner=[(222, 48, 52), (236, 78, 70), (200, 36, 42), (30, 22, 18)],
                      inner_p=[0.45, 0.3, 0.17, 0.08],
                      rind=(206, 226, 150)),
        'slime': dict(label='Slime Block', faces=(sl, sl, sl),
                      inner=[(110, 190, 90), (96, 176, 78), (128, 204, 104)], inner_p=[0.5, 0.3, 0.2]),
        'tnt': dict(label='TNT', faces=(tnt_top, tnt_side, tnt_bot),
                    inner=[(88, 86, 84), (64, 62, 60), (112, 108, 104), (190, 40, 30)], inner_p=[0.4, 0.3, 0.2, 0.1]),
        'chest': dict(label='Chest', faces=(chest_top(rng), chest_side(rng), chest_top(rng), chest_side(rng, True)),
                      inner=[(92, 64, 34), (70, 48, 24), (110, 78, 42)], inner_p=[0.5, 0.3, 0.2]),
        'diamond': dict(label='Block of Diamond', faces=(dia, dia, dia),
                        inner=[(150, 240, 236), (200, 252, 250), (110, 226, 220)], inner_p=[0.5, 0.25, 0.25]),
        'obsidian': dict(label='Obsidian', faces=(obs, obs, obs),
                         inner=[(30, 18, 46), (20, 12, 32), (60, 38, 90)], inner_p=[0.5, 0.35, 0.15]),
        'bedrock': dict(label='Bedrock', faces=(bed, bed, bed),
                        inner=[(70, 70, 70), (40, 40, 40), (110, 110, 110)], inner_p=[0.5, 0.3, 0.2]),
    }
    return items


ORDER = ('grass', 'glass', 'melon', 'slime', 'tnt', 'chest', 'diamond', 'obsidian', 'bedrock')


# ---------------------------------------------------------------------------------------------
# the press and the studio (block texture layers)
# ---------------------------------------------------------------------------------------------
def iron_block(rng):
    img = np.full((N, N, 3), (176, 178, 182), float)
    img *= (1 + (rng.random((N, N, 1)) - 0.5) * 0.05)
    img[0, :] = img[:, 0] = (204, 206, 210)
    img[-1, :] = img[:, -1] = (116, 118, 122)
    for r in (4, 8, 12):
        img[r, 1:-1] = (156, 158, 162)
    img[1:-1, 8] = (164, 166, 170)
    return img


def steel(rng):
    img = _noise(rng, (92, 96, 104), 0.08, 0.5)
    img[0, :] = img[:, 0] = (124, 128, 136)
    img[-1, :] = img[:, -1] = (58, 60, 66)
    for (r, c) in ((2, 2), (2, 13), (13, 2), (13, 13)):
        img[r, c] = (150, 154, 162)                 # bolt heads
        img[r + 1, c] = (48, 50, 54)
    return img


def chrome(rng):
    img = np.zeros((N, N, 3))
    prof = np.array([150, 170, 196, 226, 244, 236, 214, 190, 172, 158, 146, 138, 132, 128, 140, 120], float)
    for c in range(N):
        img[:, c] = prof[c] * np.array([0.96, 0.98, 1.0])
    img *= (1 + (rng.random((N, N, 1)) - 0.5) * 0.02)
    return img


def hazard(rng):
    img = np.zeros((N, N, 3))
    yy, xx = np.mgrid[0:N, 0:N]
    band = ((xx + yy) // 4) % 2 == 0
    img[band] = (232, 188, 24)
    img[~band] = (30, 28, 30)
    img *= (1 + (rng.random((N, N, 1)) - 0.5) * 0.06)
    return img


def deepslate_tiles(rng):
    img = _noise(rng, (58, 58, 64), 0.1, 0.4)
    for k in (0, 8):
        img[k, :] = (34, 34, 38)
        img[:, k] = (34, 34, 38)
    img[1, :] *= 1.18
    img[:, 1] *= 1.12
    img[9, :] *= 1.18
    img[:, 9] *= 1.12
    return img


def stone_bricks(rng):
    img = _noise(rng, (80, 80, 84), 0.08, 0.4)
    for r in (0, 4, 8, 12):
        img[r, :] = (48, 48, 51)
    for r0, cs in ((0, (0, 8)), (4, (4, 12)), (8, (0, 8)), (12, (4, 12))):
        for c in cs:
            img[r0:r0 + 4, c] = (48, 48, 51)
    return img


def redstone_lamp(rng):
    img = np.full((N, N, 3), (236, 128, 48), float)
    img *= (1 + (rng.random((N, N, 1)) - 0.5) * 0.14)
    glow = np.ones((N, N))
    frame = np.zeros((N, N), bool)
    frame[0, :] = frame[-1, :] = frame[:, 0] = frame[:, -1] = True
    frame[1, 1] = frame[1, -2] = frame[-2, 1] = frame[-2, -2] = True
    for (r, c) in ((4, 4), (4, 11), (11, 4), (11, 11), (7, 7), (8, 8)):
        frame[r, c] = True
    img[frame] = (110, 64, 32)
    glow[frame] = 0.0
    hot = rng.random((N, N)) > 0.86
    img[hot & ~frame] = (255, 196, 110)
    return img, glow


def dark_oak(rng):
    return planks(rng, (66, 44, 26))


def black_rubber(rng):
    return _noise(rng, (26, 26, 28), 0.15, 0.3)


def gauge_face(rng):
    """A round pressure gauge on the press column (static, decorative)."""
    img = np.full((N, N, 3), (40, 40, 44), float)
    yy, xx = np.mgrid[0:N, 0:N]
    r = np.hypot(xx - 7.5, yy - 7.5)
    img[r < 6.5] = (232, 230, 220)
    img[(r >= 6.5) & (r < 7.5)] = (180, 180, 186)
    img[(r > 3.5) & (r < 5.5) & (xx > 9)] = (210, 60, 40)
    img[7:9, 3:8] = (30, 30, 30)
    return img


STUDIO_LAYERS = ['deepslate', 'stone_bricks', 'iron', 'steel', 'chrome', 'hazard', 'lamp', 'dark_oak', 'rubber',
                 'gauge', 'iron_dark']
SL = {n: i for i, n in enumerate(STUDIO_LAYERS)}


def studio_textures(seed=21):
    rng = _rng(seed)
    lamp, lamp_glow = redstone_lamp(rng)
    iron = iron_block(rng)
    t = {
        'deepslate': _rgba(deepslate_tiles(rng)),
        'stone_bricks': _rgba(stone_bricks(rng)),
        'iron': _rgba(iron),
        'steel': _rgba(steel(rng)),
        'chrome': _rgba(chrome(rng)),
        'hazard': _rgba(hazard(rng)),
        'lamp': _rgba(lamp, lamp_glow),
        'dark_oak': _rgba(dark_oak(rng)),
        'rubber': _rgba(black_rubber(rng)),
        'gauge': _rgba(gauge_face(rng)),
        'iron_dark': _rgba(iron * 0.62),
    }
    return [t[k] for k in STUDIO_LAYERS], {'iron': iron, 'steel': steel(_rng(seed + 1)),
                                          'chrome': chrome(_rng(seed + 2)),
                                          'rubber': black_rubber(_rng(seed + 3)), 'hazard': hazard(_rng(seed + 4))}


# ---------------------------------------------------------------------------------------------
# Minecraft's block breaking animation: ten crack stages over a face
# ---------------------------------------------------------------------------------------------
def crack_order(seed=5):
    """(16, 16) float: the crack stage (0..1) at which each pixel of a face turns into a crack. Cracks grow from
    a few seeds along random branching walks, like the game's destroy stages."""
    rng = _rng(seed)
    order = np.full((N, N), 2.0)
    step = 0
    fronts = [(int(r), int(c)) for r, c in rng.integers(3, 13, (3, 2))]
    total = 0
    visited = np.zeros((N, N), bool)
    walk = []
    while fronts and total < 150:
        nxt = []
        for (r, c) in fronts:
            if 0 <= r < N and 0 <= c < N and not visited[r, c]:
                visited[r, c] = True
                walk.append((r, c))
                total += 1
                for _ in range(1 + int(rng.random() < 0.35)):
                    dr, dc = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)][rng.integers(8)]
                    nxt.append((r + dr, c + dc))
        fronts = nxt[:12]
        step += 1
    for k, (r, c) in enumerate(walk):
        order[r, c] = (k + 1) / (len(walk) + 1)
    return order


# ---------------------------------------------------------------------------------------------
# loot (extruded 16x16 item sprites)
# ---------------------------------------------------------------------------------------------
DIAMOND_ROWS = [
    "................",
    "................",
    "....oooooooo....",
    "...oHHwwccccoo..",
    "..oHwwccccCCcco.",
    ".oHwccccccCCCcco",
    ".occcccccccCCCo.",
    "..occccccCCCCo..",
    "...occcccCCCo...",
    "....occccCCo....",
    ".....occCCo.....",
    "......occo......",
    ".......oo.......",
    "................",
    "................",
    "................",
]
INGOT_ROWS = [
    "................",
    "................",
    "................",
    "................",
    "................",
    ".....oooooooo...",
    "....oHHHHHHHHo..",
    "...oHhhhhhhhhGo.",
    "..oHhhhhhhhhGGo.",
    ".oHhhhhhhhhGGo..",
    ".oGGGGGGGGGGo...",
    "..oooooooooo....",
    "................",
    "................",
    "................",
    "................",
]
APPLE_ROWS = [
    "................",
    ".......b........",
    "......bg........",
    ".....ggb........",
    "...rrrrbrrr.....",
    "..rRRrrrrrrr....",
    ".rRRrrrrrrrrr...",
    ".rRrrrrrrrrrrd..",
    ".rrrrrrrrrrrrd..",
    ".rrrrrrrrrrrdd..",
    ".rrrrrrrrrrrdd..",
    "..rrrrrrrrrdd...",
    "..rrrrrrrrddd...",
    "...rrdddddd.....",
    "....ddd.dd......",
    "................",
]
EMERALD_ROWS = [
    "................",
    "......oooo......",
    ".....oHHggo.....",
    "....oHgggggo....",
    "...oHgggggggo...",
    "...oHggggGGGo...",
    "..oHggggggGGGo..",
    "..oHgggggggGGo..",
    "..oggggggggGGo..",
    "..ogggggggGGGo..",
    "...oggggggGGo...",
    "...ogggggGGGo...",
    "....oggggGGo....",
    ".....oggGGo.....",
    "......oooo......",
    "................",
]
PALETTES = {
    'diamond': {'o': (18, 60, 60), 'H': (232, 255, 255), 'w': (168, 248, 240), 'c': (74, 222, 208),
                'C': (40, 170, 160)},
    'gold': {'o': (84, 56, 10), 'H': (255, 252, 180), 'h': (250, 214, 62), 'G': (208, 150, 20)},
    'iron': {'o': (54, 54, 58), 'H': (250, 250, 250), 'h': (214, 214, 216), 'G': (160, 160, 164)},
    'apple': {'b': (84, 56, 24), 'g': (80, 176, 40), 'r': (222, 34, 30), 'R': (255, 140, 120), 'd': (160, 20, 20)},
    'emerald': {'o': (10, 70, 30), 'H': (200, 255, 210), 'g': (40, 200, 90), 'G': (20, 140, 60)},
}


def sprite(name):
    rows = {'diamond': DIAMOND_ROWS, 'gold': INGOT_ROWS, 'iron': INGOT_ROWS, 'apple': APPLE_ROWS,
            'emerald': EMERALD_ROWS}[name]
    pal = PALETTES[name]
    img = np.zeros((N, N, 4), np.uint8)
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch in pal:
                img[r, c, :3] = pal[ch]
                img[r, c, 3] = 255
    return img


# ---------------------------------------------------------------------------------------------
# isometric inventory icons
# ---------------------------------------------------------------------------------------------
def iso_icon(top, side, size=64, front=None):
    """A block drawn like an inventory icon: top face lit, left face mid, right face dark (nearest-neighbour)."""
    s = size
    out = Image.new('RGBA', (s, s), (0, 0, 0, 0))

    def face(img, shade):
        a = np.clip(np.asarray(img, float)[..., :3] * shade, 0, 255).astype(np.uint8)
        return Image.fromarray(np.concatenate([a, np.full((N, N, 1), 255, np.uint8)], -1), 'RGBA')

    h = s / 2.0
    q = s / 4.0
    # top face: square -> rhombus with corners (h,0) (s,q) (h,h) (0,q)
    top_i = face(top, 1.0).resize((256, 256), Image.NEAREST)
    coeffs = _affine_coeffs([(0, 0), (256, 0), (0, 256)], [(0, q), (h, 0), (h, h)])
    t_img = top_i.transform((s, s), Image.AFFINE, coeffs, resample=Image.NEAREST)
    left_i = face(side, 0.78).resize((256, 256), Image.NEAREST)
    coeffs = _affine_coeffs([(0, 0), (256, 0), (0, 256)], [(0, q), (h, h), (0, q + h)])
    l_img = left_i.transform((s, s), Image.AFFINE, coeffs, resample=Image.NEAREST)
    right_i = face(front if front is not None else side, 0.6).resize((256, 256), Image.NEAREST)
    coeffs = _affine_coeffs([(0, 0), (256, 0), (0, 256)], [(h, h), (s, q), (h, s)])
    r_img = right_i.transform((s, s), Image.AFFINE, coeffs, resample=Image.NEAREST)
    for im in (l_img, r_img, t_img):
        out.alpha_composite(im)
    return np.asarray(out)


def _affine_coeffs(src, dst):
    """PIL AFFINE takes the inverse map (output -> input); solve it from three point pairs."""
    A = []
    b = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([u, v, 1, 0, 0, 0])
        A.append([0, 0, 0, u, v, 1])
        b.extend([x, y])
    sol = np.linalg.solve(np.array(A, float), np.array(b, float))
    return tuple(sol)


def item_icons(items, size=64):
    out = {}
    for k in ORDER:
        f = items[k]['faces']
        out[k] = iso_icon(f[0], f[1], size, front=f[3] if len(f) > 3 else None)
    return out
