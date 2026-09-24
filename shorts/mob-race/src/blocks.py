"""16 x 16 pixel-art block textures for the race, drawn in code: the cliff the marble run is built on (grass, dirt,
stone, deepslate, ores, moss), the course (stone bricks, oak planks, packed ice, slime, gold, nether bricks,
netherrack, iron bars, end rods, trapdoors, pistons, TNT) and lava. Layers are RGBA: alpha is how much a pixel glows.
"""
import numpy as np

import tnt as TN
from textures import make_block_textures

N = 16


def _rng(seed):
    return np.random.default_rng(seed)


def _noise(rng, base, var, coarse=0.4):
    fine = rng.random((N, N)) - 0.5
    blob = np.kron(rng.random((N // 4, N // 4)) - 0.5, np.ones((4, 4)))
    k = fine * (1 - coarse) + blob * coarse
    return np.clip(np.array(base, float)[None, None] * (1 + 2 * var * k[..., None]), 0, 255)


def _rgba(img, glow=None):
    a = np.zeros((N, N, 1)) if glow is None else np.asarray(glow, float)[..., None] * 255.0
    return np.concatenate([np.clip(img, 0, 255), a], -1)


def ore(stone, rng, col, hi, n=4, glow=False):
    """Stone with clusters of ore pixels."""
    img = stone.copy()
    g = np.zeros((N, N))
    for _ in range(n):
        r, c = rng.integers(2, 14, 2)
        for dr, dc in ((0, 0), (0, 1), (1, 0), (1, 1), (-1, 0), (0, -1)):
            if rng.random() < 0.75:
                img[r + dr, c + dc] = col if rng.random() < 0.7 else hi
                g[r + dr, c + dc] = 0.35 if glow else 0.0
    return img, g


def deepslate(rng):
    img = _noise(rng, (74, 74, 80), 0.1, 0.3)
    for r in range(0, N, 4):
        img[r, :] *= 0.8
    img[rng.random((N, N)) < 0.08] = (52, 52, 58)
    return img


def stone_bricks(rng):
    img = _noise(rng, (126, 126, 128), 0.07, 0.4)
    for r in (0, 4, 8, 12):
        img[r, :] = (90, 90, 92)
    for r0, cs in ((0, (0, 8)), (4, (4, 12)), (8, (0, 8)), (12, (4, 12))):
        for c in cs:
            img[r0:r0 + 4, c] = (90, 90, 92)
    img[1::4, :] *= 1.06
    return img


def planks(rng, base=(176, 134, 80)):
    img = _noise(rng, base, 0.05, 0.2)
    for r in (3, 7, 11, 15):
        img[r, :] *= 0.72
    for r0, c in ((0, 5), (4, 12), (8, 3), (12, 9)):
        img[r0:r0 + 3, c] *= 0.8
    return img


def packed_ice(rng):
    img = _noise(rng, (150, 190, 246), 0.05, 0.5)
    for (r, c, L) in ((2, 3, 5), (6, 9, 4), (10, 2, 6), (13, 10, 4)):
        for k in range(L):
            if 0 <= r + k < N and 0 <= c + k < N:
                img[r + k, c + k] = (214, 234, 255)
    img[0, :] = img[:, 0] = (190, 218, 255)
    return img


def slime(rng):
    img = np.full((N, N, 3), (118, 200, 94), float) * (1 + (rng.random((N, N, 1)) - 0.5) * 0.06)
    img[0, :] = img[-1, :] = img[:, 0] = img[:, -1] = (150, 224, 124)
    img[1:-1, 1] = img[1, 1:-1] = (138, 216, 112)
    img[4:12, 4:12] = (86, 164, 70)
    img[5:11, 5:11] = (98, 180, 80)
    img[3, 3:6] = (196, 244, 176)
    img[3:5, 3] = (196, 244, 176)
    return img


def gold(rng):
    img = np.full((N, N, 3), (246, 206, 60), float) * (1 + (rng.random((N, N, 1)) - 0.5) * 0.06)
    img[0, :] = img[:, 0] = (255, 244, 150)
    img[-1, :] = img[:, -1] = (200, 150, 30)
    for (r, c) in ((3, 3), (4, 11), (9, 6), (12, 12)):
        img[r, c] = (255, 252, 210)
    img[13, 1:15] = (214, 168, 40)
    img[1:15, 13] = (214, 168, 40)
    return img


def netherrack(rng):
    img = _noise(rng, (112, 44, 44), 0.16, 0.45)
    img[rng.random((N, N)) < 0.12] = (80, 26, 28)
    img[rng.random((N, N)) < 0.06] = (150, 70, 66)
    return img


def nether_bricks(rng):
    img = _noise(rng, (58, 26, 32), 0.1, 0.3)
    for r in (0, 4, 8, 12):
        img[r, :] = (30, 12, 16)
    for r0, cs in ((0, (0, 8)), (4, (4, 12)), (8, (0, 8)), (12, (4, 12))):
        for c in cs:
            img[r0:r0 + 4, c] = (30, 12, 16)
    return img


def iron_bars(rng):
    img = np.full((N, N, 3), (38, 40, 44), float)
    for c in (1, 2, 7, 8, 13, 14):
        img[:, c] = (150, 152, 158)
    img[:, (2, 8, 14)] = (110, 112, 118)
    img[(0, 15), :] = (130, 132, 138)
    return img


def end_rod(rng):
    img = np.full((N, N, 3), (240, 234, 226), float)
    g = np.ones((N, N)) * 0.55
    img[:, :2] = img[:, -2:] = (200, 190, 176)
    g[:, :2] = g[:, -2:] = 0.2
    return img, g


def trapdoor(rng):
    img = planks(rng, (170, 128, 76))
    img[0, :] = img[-1, :] = img[:, 0] = img[:, -1] = (110, 80, 46)
    for (r0, r1, c0, c1) in ((3, 7, 3, 7), (3, 7, 9, 13), (9, 13, 3, 7), (9, 13, 9, 13)):
        img[r0:r1, c0:c1] = (74, 52, 30)
    img[7:9, 7:9] = (90, 90, 96)
    return img


def piston_side(rng):
    img = _noise(rng, (120, 120, 122), 0.08, 0.4)
    img[:4, :] = planks(rng)[:4, :]
    img[4, :] = (74, 74, 76)
    return img


def piston_head(rng):
    img = planks(rng)
    img[5:11, 5:11] = (150, 150, 154)
    img[6:10, 6:10] = (120, 120, 124)
    return img


def bedrock(rng):
    img = _noise(rng, (86, 86, 86), 0.35, 0.55)
    img[rng.random((N, N)) < 0.18] = (42, 42, 42)
    img[rng.random((N, N)) > 0.9] = (150, 150, 150)
    return img


def mossy(stone, rng):
    img = stone.copy()
    m = rng.random((N, N)) < 0.45
    img[m] = _noise(rng, (84, 128, 52), 0.15, 0.4)[m]
    return img


def lava(rng):
    """The still-frame lava for the static basins' walls to glow against (the lava surfaces are animated voxels)."""
    img = _noise(rng, (236, 110, 20), 0.12, 0.5)
    img[rng.random((N, N)) < 0.2] = (255, 190, 60)
    return img, np.ones((N, N))


LAYERS = ['stone', 'stone2', 'dirt', 'grass_side', 'grass_top', 'deepslate', 'coal_ore', 'iron_ore', 'gold_ore',
          'diamond_ore', 'redstone_ore', 'emerald_ore', 'lapis_ore', 'deep_diamond', 'mossy', 'stone_bricks',
          'planks', 'ice', 'slime', 'gold', 'netherrack', 'nether_bricks', 'iron_bars', 'end_rod', 'trapdoor',
          'piston_side', 'piston_head', 'tnt_side', 'tnt_top', 'bedrock', 'lava', 'dark_oak', 'white']
L = {n: i for i, n in enumerate(LAYERS)}


def textures(seed=21):
    """[RGBA layer per name in LAYERS] and a dict of the RGB images by name (for voxel parts)."""
    rng = _rng(seed)
    bt = make_block_textures(3)
    stone = bt['stone']
    ds = deepslate(rng)
    img = {'stone': stone, 'stone2': bt['stone2'], 'dirt': bt['dirt'], 'grass_side': bt['grass_side'],
           'grass_top': bt['grass_top'], 'deepslate': ds}
    glow = {}
    for name, col, hi, base in (('coal_ore', (40, 40, 42), (70, 70, 72), stone),
                                ('iron_ore', (216, 176, 150), (236, 204, 180), stone),
                                ('gold_ore', (250, 214, 60), (255, 240, 140), stone),
                                ('diamond_ore', (90, 230, 220), (200, 255, 250), stone),
                                ('redstone_ore', (220, 30, 30), (255, 90, 80), stone),
                                ('emerald_ore', (40, 200, 90), (140, 255, 170), stone),
                                ('lapis_ore', (40, 70, 200), (90, 120, 240), stone),
                                ('deep_diamond', (90, 230, 220), (200, 255, 250), ds)):
        img[name], glow[name] = ore(base, rng, np.array(col, float), np.array(hi, float), n=4,
                                    glow=name in ('diamond_ore', 'redstone_ore', 'emerald_ore', 'deep_diamond'))
    img['mossy'] = mossy(stone, rng)
    img['stone_bricks'] = stone_bricks(rng)
    img['planks'] = planks(rng)
    img['ice'] = packed_ice(rng)
    img['slime'] = slime(rng)
    img['gold'] = gold(rng)
    img['netherrack'] = netherrack(rng)
    img['nether_bricks'] = nether_bricks(rng)
    img['iron_bars'] = iron_bars(rng)
    img['end_rod'], glow['end_rod'] = end_rod(rng)
    img['trapdoor'] = trapdoor(rng)
    img['piston_side'] = piston_side(rng)
    img['piston_head'] = piston_head(rng)
    img['tnt_side'] = TN.side(rng)
    img['tnt_top'] = TN.top(rng)
    img['bedrock'] = bedrock(rng)
    img['lava'], glow['lava'] = lava(rng)
    img['dark_oak'] = planks(rng, (72, 50, 30))
    img['white'] = np.full((N, N, 3), 255.0)
    glow['white'] = np.full((N, N), 0.6)
    layers = [_rgba(img[n], glow.get(n)) for n in LAYERS]
    return layers, img
