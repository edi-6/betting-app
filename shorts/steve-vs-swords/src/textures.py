"""Procedural pixel-art: blocky character skin, sword sprites, terrain block textures, HUD icons.

Everything here is drawn from scratch in code (no external image assets).
"""
import numpy as np
from noise import fbm2d

RNG = np.random.default_rng


def _hex(c):
    return np.array(c, dtype=np.float64)


def _shade(img, rng, amount=0.05):
    """Subtle per-pixel brightness jitter, like hand-shaded skins."""
    j = 1.0 + (rng.random(img.shape[:2]) - 0.5) * 2 * amount
    return np.clip(img * j[..., None], 0, 255)


def _paint(rows, palette, rng, jitter=0.045):
    h, w = len(rows), len(rows[0])
    img = np.zeros((h, w, 3))
    for r, row in enumerate(rows):
        assert len(row) == w, (row, w)
        for c, ch in enumerate(row):
            opts = palette[ch]
            if isinstance(opts, list):
                img[r, c] = opts[rng.integers(len(opts))]
            else:
                img[r, c] = opts
    return _shade(img, rng, jitter)


# ---------------------------------------------------------------------------------------------
# Character skin
# ---------------------------------------------------------------------------------------------
HAIR = [_hex((64, 42, 26)), _hex((72, 48, 30)), _hex((56, 36, 22)), _hex((80, 54, 33))]
SKIN = [_hex((200, 146, 104)), _hex((194, 140, 99)), _hex((206, 152, 110)), _hex((189, 136, 96))]
SKIN_DK = _hex((168, 116, 80))
NOSE = _hex((150, 96, 62))
MOUTH = _hex((96, 56, 38))
BEARD = [_hex((122, 76, 50)), _hex((114, 70, 46))]
EYE_W = _hex((246, 246, 246))
IRIS = _hex((84, 62, 172))
SHIRT = [_hex((22, 178, 182)), _hex((16, 168, 173)), _hex((30, 188, 190)), _hex((12, 158, 165))]
SHIRT_DK = [_hex((10, 140, 148)), _hex((6, 132, 140))]
PANTS = [_hex((58, 54, 176)), _hex((52, 48, 164)), _hex((64, 60, 186)), _hex((48, 44, 156))]
SHOE = [_hex((108, 108, 112)), _hex((96, 96, 100)), _hex((118, 118, 122))]

SKIN_PAL = {
    'H': HAIR, 'S': SKIN, 's': SKIN_DK, 'N': NOSE, 'M': MOUTH, 'B': BEARD, 'W': EYE_W, 'I': IRIS,
    'T': SHIRT, 't': SHIRT_DK, 'P': PANTS, 'G': SHOE,
}


def make_skin(seed=7):
    """Return {part: {face: HxWx3 float array}} for a Steve-like blocky character (own design).

    Face image convention (as seen from outside the part):
      ny = front (-Y), py = back (+Y), nx = -X side, px = +X side, pz = top, nz = bottom.
      Row 0 is the top edge (for pz: the +Y edge), column 0 is the viewer's left.
    """
    rng = RNG(seed)
    P = lambda rows, j=0.045: _paint(rows, SKIN_PAL, rng, j)
    head = {
        'ny': P(["HHHHHHHH",
                 "HHHHHHHH",
                 "HSSSSSSH",
                 "SWISSIWS",
                 "SSsNNsSS",
                 "SBMMMMBS",
                 "SSSSSSSS",
                 "SSSSSSSS"]),
        # viewer at -X: image left = back (+Y), right = front (-Y)
        'nx': P(["HHHHHHHH",
                 "HHHHHHHH",
                 "HHHHSSSS",
                 "HHHSSSSS",
                 "HHSSSSSS",
                 "HSSSSSSS",
                 "HSSSSSSS",
                 "SSSSSSSS"]),
        'px': P(["HHHHHHHH",
                 "HHHHHHHH",
                 "SSSSHHHH",
                 "SSSSSHHH",
                 "SSSSSSHH",
                 "SSSSSSSH",
                 "SSSSSSSH",
                 "SSSSSSSS"]),
        'py': P(["HHHHHHHH"] * 7 + ["HHHHHHHH"]),
        'pz': P(["HHHHHHHH"] * 8),
        'nz': P(["ssssssss"] * 8, 0.03),
    }
    shirt_rows = ["TTTTTTTT",
                  "TTTTTTTT",
                  "TTTTTTTT",
                  "TTTTTTTT",
                  "TTTTTTTT",
                  "TTTTTTTT",
                  "TTTTTTTT",
                  "tTTTTTTt",
                  "TtTTTtTT",
                  "PPPPPPPP",
                  "PPPPPPPP",
                  "PPPPPPPP"]
    body = {
        'ny': P(shirt_rows),
        'py': P(shirt_rows),
        'nx': P(["TTTT"] * 8 + ["tTTt"] + ["PPPP"] * 3),
        'px': P(["TTTT"] * 8 + ["tTTt"] + ["PPPP"] * 3),
        'pz': P(["TTTTTTTT"] * 4),
        'nz': P(["PPPPPPPP"] * 4),
    }
    arm_rows = ["TTTT", "TTTT", "TTTT", "tTTt"] + ["SSSS"] * 8
    arm = lambda: {
        'ny': P(arm_rows), 'py': P(arm_rows), 'nx': P(arm_rows), 'px': P(arm_rows),
        'pz': P(["TTTT"] * 4), 'nz': P(["SSSS"] * 4),
    }
    leg_rows = ["PPPP"] * 10 + ["GGGG"] * 2
    leg = lambda: {
        'ny': P(leg_rows), 'py': P(leg_rows), 'nx': P(leg_rows), 'px': P(leg_rows),
        'pz': P(["PPPP"] * 4), 'nz': P(["GGGG"] * 4),
    }
    # vertical shading gradient on shirt / pants so large flat areas don't look plastic
    for part in (body,):
        for f in ('ny', 'py', 'nx', 'px'):
            h = part[f].shape[0]
            g = np.linspace(1.03, 0.95, h)[:, None, None]
            part[f] = np.clip(part[f] * g, 0, 255)
    return {'head': head, 'body': body, 'arm_r': arm(), 'arm_l': arm(), 'leg_r': leg(), 'leg_l': leg()}


# ---------------------------------------------------------------------------------------------
# Swords (16x16 sprites, tip at top-right, pommel at bottom-left)
# ---------------------------------------------------------------------------------------------
SWORD_MATERIALS = {
    #            outline         dark            mid             light           guard           guard dark
    'wood':      ((64, 44, 24),  (118, 84, 46),  (150, 110, 64), (184, 142, 86), (104, 74, 40), (74, 52, 28)),
    'stone':     ((52, 52, 54),  (104, 104, 106), (134, 134, 136), (170, 170, 172), (96, 96, 98), (66, 66, 68)),
    'iron':      ((70, 70, 74),  (176, 176, 180), (212, 212, 216), (244, 244, 248), (132, 132, 136), (92, 92, 96)),
    'gold':      ((112, 70, 8),  (222, 164, 22), (250, 214, 62), (255, 246, 150), (206, 150, 20), (140, 96, 10)),
    'diamond':   ((14, 70, 70),  (36, 176, 170), (86, 228, 218), (196, 255, 250), (28, 128, 124), (16, 84, 82)),
    'netherite': ((24, 20, 24),  (58, 50, 56),   (80, 70, 78),   (118, 106, 116), (58, 50, 56), (36, 30, 34)),
}
HANDLE = ((72, 48, 22), (104, 74, 38), (138, 102, 56))  # outline, dark, light


def sword_mask():
    """Return integer label image (16x16): 0 empty, 1 outline, 2 blade dark, 3 blade mid, 4 blade light,
    5 guard, 6 guard dark, 7 handle outline, 8 handle dark, 9 handle light."""
    m = np.zeros((16, 16), np.int32)
    rr, cc = np.mgrid[0:16, 0:16]
    d = cc - rr          # position along the blade axis (tip at the top-right corner)
    s = cc + rr          # position across the blade axis
    blade = (d >= -3) & (d <= 13)
    m[blade & (s == 14)] = 4
    m[blade & (s == 15)] = 3
    m[blade & (s == 16)] = 2
    guard = (d >= -5) & (d <= -4) & (s >= 11) & (s <= 19)
    m[guard & (d == -4)] = 5
    m[guard & (d == -5)] = 6
    handle = (d >= -11) & (d <= -6) & (s >= 14) & (s <= 16)
    m[handle & (s <= 15)] = 9
    m[handle & (s == 16)] = 8
    pommel = (d >= -13) & (d <= -12) & (s >= 13) & (s <= 17)
    m[pommel] = 6
    filled = m > 0
    grown = filled.copy()
    grown[1:, :] |= filled[:-1, :]
    grown[:-1, :] |= filled[1:, :]
    grown[:, 1:] |= filled[:, :-1]
    grown[:, :-1] |= filled[:, 1:]
    ring = grown & ~filled
    near_handle = np.zeros_like(filled)
    hmask = (m == 8) | (m == 9)
    near_handle[1:, :] |= hmask[:-1, :]
    near_handle[:-1, :] |= hmask[1:, :]
    near_handle[:, 1:] |= hmask[:, :-1]
    near_handle[:, :-1] |= hmask[:, 1:]
    m[ring] = 1
    m[ring & near_handle] = 7
    return m


def make_sword_sprites():
    """Return dict material -> (16,16,4) uint8 RGBA sprite."""
    m = sword_mask()
    out = {}
    for name, (o, dk, mid, lt, gd, gdd) in SWORD_MATERIALS.items():
        img = np.zeros((16, 16, 4), np.uint8)
        lut = {1: o, 2: dk, 3: mid, 4: lt, 5: gd, 6: gdd, 7: HANDLE[0], 8: HANDLE[1], 9: HANDLE[2]}
        for lab, col in lut.items():
            sel = m == lab
            img[sel, :3] = col
            img[sel, 3] = 255
        out[name] = img
    return out


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
    return tex


def block_atlas(tex, order):
    """Pack 16x16 textures into a horizontal strip atlas. Returns (uint8 image, index map)."""
    n = len(order)
    atlas = np.zeros((16, 16 * n, 3), np.uint8)
    idx = {}
    for i, name in enumerate(order):
        atlas[:, i * 16:(i + 1) * 16] = np.clip(tex[name], 0, 255).astype(np.uint8)
        idx[name] = i
    return atlas, idx


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
