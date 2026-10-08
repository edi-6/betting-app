"""Block textures (16x16, drawn here, no game assets) for the coast and the village.

The first 13 layers keep the order the renderer's terrain shader expects (grass top 0, grass side 2, leaves 7 and 12,
water 9); the rest are the village's blocks.
"""
import numpy as np

from textures import make_block_textures

LAYERS = ['grass_top', 'dirt', 'grass_side', 'stone', 'stone2', 'log', 'log_top', 'leaves', 'sand', 'water',
          'planks', 'birch_log', 'birch_leaves', 'gravel', 'cobble', 'mossy_cobble', 'stone_bricks', 'glass',
          'spruce_planks', 'path_top', 'path_side', 'hay_side', 'hay_top', 'farmland', 'wheat', 'wool_white',
          'wool_red', 'gold', 'glowstone', 'sponge', 'wet_sponge', 'terracotta', 'andesite', 'sandstone',
          'sandstone_top', 'clay', 'spruce_log', 'mossy_stone_bricks', 'bookshelf', 'iron_block', 'snow',
          'cobble_floor']
L = {n: i for i, n in enumerate(LAYERS)}
GLASS_LAYER = L['glass']
WET_SPONGE_LAYER = L['wet_sponge']
RNG = np.random.default_rng


def _speckle(base, var, rng, n=16):
    img = np.ones((n, n, 3)) * np.array(base, float)
    img *= 1.0 + (rng.random((n, n, 1)) - 0.5) * 2 * var
    return img


def _cobble(rng, base=(122, 122, 124), mortar=(70, 70, 72)):
    """Rounded stones of different greys separated by dark mortar (a Voronoi pattern that tiles)."""
    pts = rng.random((11, 2)) * 16
    yy, xx = np.mgrid[0:16, 0:16] + 0.5
    d = []
    for p in pts:
        for ox in (-16, 0, 16):
            for oy in (-16, 0, 16):
                d.append(np.hypot(xx - p[0] - ox, yy - p[1] - oy))
    d = np.sort(np.stack(d), 0)
    edge = (d[1] - d[0]) < 1.25
    owner = np.argmin(np.stack([np.min(np.stack([np.hypot(xx - p[0] - ox, yy - p[1] - oy)
                                                 for ox in (-16, 0, 16) for oy in (-16, 0, 16)]), 0)
                                for p in pts]), 0)
    shade = 0.82 + 0.3 * rng.random(len(pts))
    img = np.array(base, float)[None, None] * shade[owner][..., None]
    img *= 1.0 + (rng.random((16, 16, 1)) - 0.5) * 0.12
    # a highlight on the upper-left of each stone, shadow lower-right
    hl = (d[1] - d[0] > 2.6) & (rng.random((16, 16)) < 0.2)
    img[hl] *= 1.15
    img[edge] = np.array(mortar, float) * (0.9 + 0.2 * rng.random((edge.sum(), 1)))
    return img


def _bricks(rng, base=(124, 124, 126), line=(86, 86, 88)):
    img = _speckle(base, 0.06, rng)
    for r in (3, 7, 11, 15):
        img[r] = np.array(line) * (0.95 + 0.1 * rng.random((16, 1)))
    for k, r0 in enumerate((0, 4, 8, 12)):
        off = 0 if k % 2 == 0 else 8
        for c in ((off + 7) % 16, (off + 15) % 16):
            img[r0:r0 + 3, c] = line
        img[r0, :] *= 1.06                      # lit top edge of each course
    return img


def _planks(rng, base, line_mul=0.7):
    p = np.ones((16, 16, 3)) * np.array(base, float)
    p *= 1.0 + (rng.random((16, 16, 1)) - 0.5) * 0.10
    for r in (3, 7, 11, 15):
        p[r] = np.array(base) * line_mul
    for r0 in (0, 4, 8, 12):
        c = int(rng.integers(3, 13))
        p[r0:r0 + 3, c] *= 0.82
        p[r0 + 1, (c + 5) % 16] *= 0.86
        for _ in range(2):                       # grain streaks
            rr = r0 + int(rng.integers(0, 3))
            c0 = int(rng.integers(0, 12))
            p[rr, c0:c0 + int(rng.integers(3, 6))] *= 0.92
    return p


def block_textures(seed=4):
    rng = RNG(seed)
    t = make_block_textures()
    tex = {k: t[k] for k in ('grass_top', 'dirt', 'grass_side', 'stone', 'stone2', 'log', 'log_top', 'leaves')}
    for k in ('grass_top', 'grass_side', 'leaves'):
        g = tex[k].astype(float)
        grey = g.mean(-1, keepdims=True)
        tex[k] = grey + (g - grey) * 0.8
    tex['grass_top'] = tex['grass_top'] * np.array([0.95, 0.92, 0.88])
    s = _speckle((219, 207, 160), 0.06, rng)
    d = rng.random((16, 16))
    s[d < 0.1] = (200, 188, 140)
    s[d > 0.94] = (232, 222, 180)
    tex['sand'] = s
    yy, xx = np.mgrid[0:16, 0:16]
    w = np.ones((16, 16, 3)) * np.array((44, 92, 196), float)
    w[(np.sin((xx + 2 * np.sin(yy * 0.9)) * 0.8) > 0.55)] = (70, 124, 222)
    tex['water'] = w
    tex['planks'] = _planks(rng, (162, 130, 78))
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
    tex['cobble'] = _cobble(rng)
    mc = _cobble(rng)
    blob = (np.sin(xx * 0.8 + 0.6) + np.cos(yy * 0.6 + 1.1) + rng.random((16, 16)) * 1.4) > 1.1
    mc[blob] = np.array((86, 116, 58)) * (0.8 + 0.35 * rng.random((blob.sum(), 1)))
    tex['mossy_cobble'] = mc
    tex['stone_bricks'] = _bricks(rng)
    msb = _bricks(rng)
    blob = (np.sin(xx * 0.7) + np.cos(yy * 0.9 + 2.0) + rng.random((16, 16)) * 1.5) > 1.3
    msb[blob] = np.array((90, 118, 62)) * (0.8 + 0.3 * rng.random((blob.sum(), 1)))
    tex['mossy_stone_bricks'] = msb
    # glass: a dark, cool pane with a pale frame and two streaks of reflection
    gl = np.ones((16, 16, 3)) * np.array((58, 84, 104), float)
    gl *= 1.0 + (yy / 15.0 - 0.5)[..., None] * -0.25
    streak = ((xx + yy) % 16 >= 9) & ((xx + yy) % 16 <= 10) | ((xx + yy) % 16 == 13)
    gl[streak & (xx > 1) & (xx < 14) & (yy > 1) & (yy < 14)] = (150, 178, 196)
    gl[0, :] = gl[15, :] = gl[:, 0] = gl[:, 15] = (214, 226, 232)
    gl[1, 1:15] = (176, 196, 206)
    tex['glass'] = gl
    tex['spruce_planks'] = _planks(rng, (112, 82, 50), 0.66)
    pt = _speckle((148, 120, 72), 0.08, rng)
    d = rng.random((16, 16))
    pt[d < 0.12] = (124, 98, 58)
    pt[d > 0.92] = (168, 140, 88)
    tex['path_top'] = pt
    ps = tex['dirt'].copy()
    ps[:2] = pt[:2]
    tex['path_side'] = ps
    hs = _speckle((196, 160, 40), 0.10, rng)
    for r in (2, 3, 12, 13):
        hs[r] = (140, 96, 30)
    for c in range(0, 16, 2):
        hs[:, c] *= 0.92
    tex['hay_side'] = hs
    ht = _speckle((206, 170, 52), 0.10, rng)
    rr = np.maximum(abs(xx - 7.5), abs(yy - 7.5))
    ht[(rr.astype(int) % 3 == 0)] *= 0.86
    tex['hay_top'] = ht
    fl = _speckle((92, 60, 36), 0.10, rng)
    for r in (2, 6, 10, 14):
        fl[r] *= 0.72
    tex['farmland'] = fl
    wh = np.ones((16, 16, 3)) * np.array((92, 60, 36), float)          # wheat over soil
    for c in range(1, 16, 2):
        h = int(rng.integers(9, 15))
        wh[16 - h:, c] = (186, 164, 64)
        wh[16 - h:16 - h + 3, c] = (214, 190, 86)
        wh[16 - h + 3:, c] *= 1.0 - 0.3 * (np.arange(h - 3)[:, None] / h)
        if c + 1 < 16:
            wh[16 - h + 4:, c + 1] = (120, 140, 52)
    tex['wheat'] = wh
    tex['wool_white'] = _speckle((232, 232, 228), 0.05, rng)
    tex['wool_red'] = _speckle((172, 44, 40), 0.07, rng)
    go = _speckle((246, 204, 64), 0.05, rng)
    go[0, :] = go[:, 0] = (255, 240, 140)
    go[15, :] = go[:, 15] = (200, 150, 30)
    tex['gold'] = go
    gs = _speckle((214, 158, 84), 0.15, rng)
    glow = np.zeros((16, 16))
    for _ in range(14):
        r, c = rng.integers(0, 15, 2)
        gs[r:r + 2, c:c + 2] = (255, 232, 150)
        glow[r:r + 2, c:c + 2] = 200
    gs4 = np.concatenate([np.clip(gs, 0, 255), glow[..., None]], -1)
    tex['glowstone'] = gs4
    sp = _speckle((198, 186, 72), 0.07, rng)
    for _ in range(16):
        r, c = rng.integers(0, 15, 2)
        sp[r:r + int(rng.integers(1, 3)), c:c + int(rng.integers(1, 3))] = (150, 136, 40)
    tex['sponge'] = sp
    ws = _speckle((172, 160, 58), 0.07, rng)
    for _ in range(16):
        r, c = rng.integers(0, 15, 2)
        ws[r:r + int(rng.integers(1, 3)), c:c + int(rng.integers(1, 3))] = (124, 114, 40)
    for c in (3, 9, 13):                         # drips
        ws[11:, c] = (150, 160, 110)
    tex['wet_sponge'] = ws
    tex['terracotta'] = _speckle((152, 92, 66), 0.05, rng)
    a = _speckle((132, 132, 134), 0.08, rng)
    d = rng.random((16, 16))
    a[d < 0.14] = (104, 104, 108)
    a[d > 0.88] = (162, 162, 164)
    tex['andesite'] = a
    ss = _speckle((218, 206, 160), 0.04, rng)
    ss[:3] = (230, 220, 178)
    ss[12:] = (200, 186, 140)
    tex['sandstone'] = ss
    tex['sandstone_top'] = _speckle((224, 214, 170), 0.04, rng)
    tex['clay'] = _speckle((160, 166, 178), 0.05, rng)
    sl = np.zeros((16, 16, 3))
    for c in range(16):
        sl[:, c] = (74, 52, 30) if rng.random() > 0.35 else (58, 40, 22)
    tex['spruce_log'] = sl * (1 + (rng.random((16, 16, 1)) - 0.5) * 0.16)
    bk = _planks(rng, (162, 130, 78))
    cols = [(130, 40, 36), (44, 70, 130), (60, 110, 54), (140, 110, 50), (90, 50, 110), (40, 40, 44)]
    for r0 in (1, 9):
        c = 1
        while c < 15:
            w_ = int(rng.integers(1, 3))
            h_ = int(rng.integers(5, 7))
            col = np.array(cols[int(rng.integers(len(cols)))], float)
            bk[r0 + 6 - h_:r0 + 6, c:c + w_] = col * (0.85 + 0.3 * rng.random())
            c += w_
    tex['bookshelf'] = bk
    ib = _speckle((216, 216, 216), 0.03, rng)
    ib[0, :] = ib[:, 0] = (240, 240, 240)
    ib[15, :] = ib[:, 15] = (170, 170, 170)
    tex['iron_block'] = ib
    tex['snow'] = _speckle((238, 244, 250), 0.03, rng)
    cf = _cobble(rng, base=(132, 128, 124), mortar=(84, 80, 76))
    tex['cobble_floor'] = cf
    out = []
    for n in LAYERS:
        x = np.clip(tex[n], 0, 255).astype(np.uint8)
        if x.shape[-1] == 3:
            x = np.concatenate([x, np.zeros(x.shape[:2] + (1,), np.uint8)], -1)
        out.append(x)
    return out


# block kinds of the village: texture layer per face (top, sides, bottom), strength, density (relative to water)
#   strength: the load the water has to put on it to break it loose (see village.py)
KINDS = {
    'planks': ('planks', 'planks', 'planks', 1.0, 0.6),
    'spruce': ('spruce_planks', 'spruce_planks', 'spruce_planks', 1.0, 0.6),
    'log': ('log_top', 'log', 'log_top', 1.6, 0.7),
    'spruce_log': ('log_top', 'spruce_log', 'log_top', 1.6, 0.7),
    'cobble': ('cobble', 'cobble', 'cobble', 3.0, 2.4),
    'mossy_cobble': ('mossy_cobble', 'mossy_cobble', 'mossy_cobble', 3.0, 2.4),
    'bricks': ('stone_bricks', 'stone_bricks', 'stone_bricks', 4.0, 2.4),
    'mossy_bricks': ('mossy_stone_bricks', 'mossy_stone_bricks', 'mossy_stone_bricks', 4.0, 2.4),
    'glass': ('glass', 'glass', 'glass', 0.25, 2.2),
    'path': ('path_top', 'path_side', 'dirt', 9.0, 1.6),
    'hay': ('hay_top', 'hay_side', 'hay_top', 0.5, 0.4),
    'farmland': ('farmland', 'dirt', 'dirt', 9.0, 1.6),
    'wheat': ('wheat', 'wheat', 'farmland', 0.3, 0.4),
    'wool_white': ('wool_white', 'wool_white', 'wool_white', 0.5, 0.5),
    'wool_red': ('wool_red', 'wool_red', 'wool_red', 0.5, 0.5),
    'gold': ('gold', 'gold', 'gold', 2.0, 3.0),
    'glowstone': ('glowstone', 'glowstone', 'glowstone', 1.0, 1.5),
    'terracotta': ('terracotta', 'terracotta', 'terracotta', 2.5, 2.0),
    'leaves': ('leaves', 'leaves', 'leaves', 0.35, 0.3),
    'bookshelf': ('planks', 'bookshelf', 'planks', 0.8, 0.6),
    'sand': ('sand', 'sand', 'sand', 9.0, 1.6),
    'sponge': ('sponge', 'sponge', 'sponge', 99.0, 0.4),
    'wet_sponge': ('wet_sponge', 'wet_sponge', 'wet_sponge', 99.0, 0.9),
    'dirt': ('dirt', 'dirt', 'dirt', 9.0, 1.6),
    'grass': ('grass_top', 'grass_side', 'dirt', 9.0, 1.6),
    'stone': ('stone', 'stone', 'stone', 4.0, 2.5),
}
KIND_NAMES = list(KINDS)
KIND_ID = {k: i + 1 for i, k in enumerate(KIND_NAMES)}          # 0 = air
FACE_LAYERS = np.zeros((len(KIND_NAMES) + 1, 3), np.int32)      # [kind, (top, side, bottom)]
STRENGTH = np.zeros(len(KIND_NAMES) + 1)
DENSITY = np.ones(len(KIND_NAMES) + 1)
for _k, (_top, _side, _bot, _s, _d) in KINDS.items():
    FACE_LAYERS[KIND_ID[_k]] = (L[_top], L[_side], L[_bot])
    STRENGTH[KIND_ID[_k]] = _s
    DENSITY[KIND_ID[_k]] = _d


def debris_layers():
    """Texture layers for the loose blocks: three per kind (top, side, bottom), in KIND_NAMES order."""
    tex = block_textures()
    out = []
    for k in KIND_NAMES:
        for li in FACE_LAYERS[KIND_ID[k]]:
            img = tex[li][..., :3]
            out.append(np.concatenate([img, np.full((16, 16, 1), 255, np.uint8)], -1))
    return out


def debris_variant(kind_ids):
    """The prop variant (first texture layer) of each kind id (1-based)."""
    return 3.0 * (np.asarray(kind_ids) - 1)


def debris_mesh():
    """A unit block centred on the origin: triangles with pos3 normal3 uv2 layer1; layer 100 + (0 top, 1 side,
    2 bottom) so the instance's variant picks the kind."""
    h = 0.5
    v = []
    faces = [
        ((1, 0, 0), [(h, -h, -h), (h, h, -h), (h, h, h), (h, -h, h)], 1),
        ((-1, 0, 0), [(-h, h, -h), (-h, -h, -h), (-h, -h, h), (-h, h, h)], 1),
        ((0, 1, 0), [(h, h, -h), (-h, h, -h), (-h, h, h), (h, h, h)], 1),
        ((0, -1, 0), [(-h, -h, -h), (h, -h, -h), (h, -h, h), (-h, -h, h)], 1),
        ((0, 0, 1), [(-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h)], 0),
        ((0, 0, -1), [(-h, h, -h), (h, h, -h), (h, -h, -h), (-h, -h, -h)], 2),
    ]
    uv = [(0, 1), (1, 1), (1, 0), (0, 0)]
    for nrm, quad, layer in faces:
        for a, b, c in ((0, 1, 2), (0, 2, 3)):
            for k in (a, b, c):
                v.append((*quad[k], *nrm, *uv[k], 100.0 + layer))
    return np.array(v, np.float32)
