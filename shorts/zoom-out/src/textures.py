"""Procedural pixel-art: terrain block textures and HUD icons (hearts, X, check).

Everything here is drawn from scratch in code (no external image assets).
"""
import numpy as np

RNG = np.random.default_rng


def _shade(img, rng, amount=0.05):
    """Subtle per-pixel brightness jitter, like hand-shaded skins."""
    j = 1.0 + (rng.random(img.shape[:2]) - 0.5) * 2 * amount
    return np.clip(img * j[..., None], 0, 255)


# ---------------------------------------------------------------------------------------------
# Terrain block textures (16x16)
# ---------------------------------------------------------------------------------------------
def _speckle(base, var, rng, n=16, coarse=0.5):
    """Base colour with pixel noise + slightly coherent variation."""
    fine = rng.random((n, n)) - 0.5
    blob = np.kron(rng.random((n // 4, n // 4)) - 0.5, np.ones((4, 4)))
    k = fine * (1 - coarse) + blob * coarse
    img = np.array(base, float)[None, None, :] * (1 + var * 2 * k[..., None])
    return np.clip(img, 0, 255)


def make_block_textures(seed=3):
    rng = RNG(seed)
    tex = {}
    # grass top: bright lime like a sunny plains biome
    g = _speckle((104, 176, 58), 0.12, rng, coarse=0.35)
    dots = rng.random((16, 16))
    g[dots < 0.08] *= 0.84
    g[dots > 0.94] *= 1.10
    tex['grass_top'] = np.clip(g, 0, 255)
    # dirt
    d = _speckle((136, 96, 64), 0.12, rng, coarse=0.3)
    dots = rng.random((16, 16))
    d[dots < 0.10] = (108, 76, 50)
    d[dots > 0.93] = (158, 116, 80)
    tex['dirt'] = d
    # grass side: green lip with ragged edge over dirt
    gs = tex['dirt'].copy()
    lip = 3 + (rng.random(16) < 0.45).astype(int) + (rng.random(16) < 0.2).astype(int)
    gtop = _speckle((98, 168, 54), 0.10, rng, coarse=0.3)
    for c in range(16):
        gs[:lip[c], c] = gtop[:lip[c], c]
    tex['grass_side'] = gs
    # stone
    s = _speckle((128, 128, 130), 0.10, rng, coarse=0.5)
    dots = rng.random((16, 16))
    s[dots < 0.07] = (100, 100, 102)
    s[dots > 0.95] = (150, 150, 152)
    tex['stone'] = s
    # darker, slightly warmer stone band for cliff strata
    tex['stone2'] = np.clip(_speckle((112, 110, 112), 0.10, rng, coarse=0.5), 0, 255)
    # log side: vertical bark stripes
    lg = np.zeros((16, 16, 3))
    cols = rng.random(16)
    for c in range(16):
        base = np.array((106, 82, 50)) if cols[c] > 0.35 else np.array((82, 62, 36))
        lg[:, c] = base
    lg = np.clip(lg * (1 + (rng.random((16, 16, 1)) - 0.5) * 0.18), 0, 255)
    tex['log'] = lg
    # log top: rings
    yy, xx = np.mgrid[0:16, 0:16]
    rr = np.maximum(abs(xx - 7.5), abs(yy - 7.5))
    lt = np.where((rr.astype(int) % 3 == 0)[..., None], (150, 118, 72), (178, 144, 92)).astype(float)
    lt[rr >= 7] = (96, 74, 44)
    tex['log_top'] = _shade(lt, rng, 0.05)
    # leaves (opaque variant with dark gaps)
    lv = _speckle((64, 140, 44), 0.20, rng, coarse=0.25)
    dots = rng.random((16, 16))
    lv[dots < 0.16] = (30, 82, 22)
    lv[(dots > 0.16) & (dots < 0.22)] = (44, 104, 30)
    lv[dots > 0.90] = (96, 178, 66)
    tex['leaves'] = np.clip(lv, 0, 255)
    # sculk (Deep Dark): near-black teal, darker mottling, veins and a few glowing cyan specks (alpha = glow)
    sk = _speckle((14, 34, 42), 0.22, rng, coarse=0.55)
    dots = rng.random((16, 16))
    sk[dots < 0.16] = (9, 22, 28)
    sk[(dots > 0.80) & (dots < 0.90)] = (22, 58, 68)
    glow = np.zeros((16, 16))
    for _ in range(4):
        r, c = rng.integers(1, 15, 2)
        sk[r, c] = (48, 224, 236)
        glow[r, c] = 255
        dr, dc = [(0, 1), (1, 0), (0, -1), (-1, 0)][rng.integers(4)]
        if rng.random() < 0.6:
            sk[r + dr, c + dc] = (36, 170, 184)
            glow[r + dr, c + dc] = 255
    tex['sculk'] = np.concatenate([np.clip(sk, 0, 255), glow[..., None]], -1)
    return tex


# ---------------------------------------------------------------------------------------------
# HUD pixel icons
# ---------------------------------------------------------------------------------------------
HEART_ROWS = [
    ".OO...OO.",
    "ORRO.ORRO",
    "ORWRORRRO",
    "ORRRRRRRO",
    "ORRRRRRDO",
    ".ORRRRDO.",
    "..ORRDO..",
    "...ODO...",
    "....O....",
]


def heart_pixels(state='full', flash=False):
    """Return (9,9,4) uint8 RGBA for a heart icon. state: full | half | empty."""
    if flash:
        pal = {'O': (255, 255, 255), 'R': (255, 118, 118), 'W': (255, 238, 238), 'D': (236, 86, 86),
               'E': (120, 120, 120), 'e': (150, 150, 150), 'k': (96, 96, 96)}
    else:
        pal = {'O': (12, 6, 6), 'R': (236, 22, 22), 'W': (255, 214, 214), 'D': (170, 0, 0),
               'E': (40, 38, 38), 'e': (66, 62, 62), 'k': (30, 28, 28)}
    img = np.zeros((9, 9, 4), np.uint8)
    for r, row in enumerate(HEART_ROWS):
        for c, ch in enumerate(row):
            if ch == '.':
                continue
            filled = state == 'full' or (state == 'half' and c <= 4)
            if ch == 'O':
                col = pal['O']
            elif filled:
                col = pal[ch]
            else:
                col = pal['e'] if ch == 'W' else (pal['k'] if ch == 'D' else pal['E'])
            img[r, c, :3] = col
            img[r, c, 3] = 255
    return img


def _outline(mask, rings):
    """Grow a pixel mask ring by ring (8-connected). Returns list of masks, one per ring."""
    out = []
    cur = mask.copy()
    for _ in range(rings):
        grown = cur.copy()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                grown |= np.roll(np.roll(cur, dr, 0), dc, 1)
        out.append(grown & ~cur)
        cur = grown
    return out


def x_pixels():
    """Red pixel-art X with white rim and dark outline."""
    n = 17
    yy, xx = np.mgrid[0:n, 0:n]
    c0, c1 = 2, n - 3
    core = ((abs(yy - xx) <= 1) | (abs(yy + xx - (n - 1)) <= 1)) & (yy >= c0) & (yy <= c1) & (xx >= c0) & (xx <= c1)
    white, dark = _outline(core, 2)
    img = np.zeros((n, n, 4), np.uint8)
    img[dark] = (24, 10, 10, 255)
    img[white] = (255, 255, 255, 255)
    img[core] = (226, 30, 34, 255)
    # darker lower-right shading on the red for a little depth
    shade = core & ((yy + xx) > (n - 1)) & (abs(yy - xx) == 1)
    img[shade] = (178, 16, 20, 255)
    return img


def check_pixels():
    """Green pixel-art check mark with white rim and dark outline."""
    n_r, n_c = 15, 19
    core = np.zeros((n_r, n_c), bool)
    # short stroke going down-right, then long stroke going up-right
    for t in range(0, 5):
        r, c = 7 + t, 3 + t
        core[r - 1:r + 2, c:c + 1] = True
        core[r - 1:r + 2, c + 1:c + 2] = True
    for t in range(0, 10):
        r, c = 11 - t, 7 + t
        if 1 <= r < n_r - 1 and c < n_c - 2:
            core[r - 1:r + 2, c:c + 2] = True
    core[:2, :] = False
    core[-2:, :] = False
    core[:, :2] = False
    core[:, -2:] = False
    white, dark = _outline(core, 2)
    img = np.zeros((n_r, n_c, 4), np.uint8)
    img[dark] = (8, 26, 10, 255)
    img[white] = (255, 255, 255, 255)
    img[core] = (46, 204, 64, 255)
    yy, xx = np.mgrid[0:n_r, 0:n_c]
    img[core & (yy >= 10)] = (30, 164, 48, 255)
    return img


def upscale(img, k):
    return np.kron(img, np.ones((k, k, 1), img.dtype)) if img.ndim == 3 else np.kron(img, np.ones((k, k), img.dtype))
