"""Arrow: 16x8 pixel-art sprite drawn in code, rendered Minecraft-style as two crossed quads.

Thin alpha-tested sprites vanish when mip-mapped normally (a 1-pixel shaft averages to ~25% alpha), so the
texture carries its own hand-built mip chain laid out side by side ("mip atlas") where every level keeps
alpha = max of the 2x2 block. The shader picks the level from screen-space derivatives and uses texelFetch.
"""
import numpy as np

TEX_W, TEX_H = 16, 8
ATLAS_W = 32                     # 16 + 8 + 4 + 2 + 1 (+1 padding)
LEVEL_X = (0, 16, 24, 28, 30)    # x offset of each mip level in the atlas
PX = 0.2                         # world units per sprite pixel
LENGTH = TEX_W * PX              # 3.2 units nock to tip
SHAFT_ROW = 3                    # sprite row of the shaft (the arrow axis)
TIP = LENGTH / 2                 # local x of the tip; the nock is at -TIP

# rows top to bottom, tip on the right
#   F/f = feather light/shade, o = feather outline, W/w = shaft light/dark, N = nock,
#   H/h/k = head light/mid/dark
ROWS = [
    "oo..............",
    "Ffo.........k...",
    "fFfo........hk..",
    "NWwWwWwWwWwWHhHk",
    "fFfo........hk..",
    "Ffo.........k...",
    "oo..............",
    "................",
]

SHAFT = {'W': (140, 102, 60), 'w': (112, 80, 44), 'N': (70, 50, 30)}
HEAD = {'H': (196, 196, 200), 'h': (150, 150, 156), 'k': (92, 92, 98)}
FEATHERS = [   # (light, shade, outline) per variant
    ((240, 240, 240), (206, 206, 210), (150, 150, 156)),
    ((226, 226, 228), (190, 190, 196), (140, 140, 146)),
    ((238, 232, 218), (204, 196, 180), (150, 144, 130)),
]


def sprite(variant=0, seed=0):
    rng = np.random.default_rng(seed + 17 * variant)
    light, shade, outline = FEATHERS[variant]
    lut = dict(SHAFT)
    lut.update(HEAD)
    lut.update({'F': light, 'f': shade, 'o': outline})
    img = np.zeros((TEX_H, TEX_W, 4), np.float64)
    for r, row in enumerate(ROWS):
        assert len(row) == TEX_W
        for c, ch in enumerate(row):
            if ch == '.':
                continue
            img[r, c, :3] = lut[ch]
            img[r, c, 3] = 255
    j = 1.0 + (rng.random((TEX_H, TEX_W, 1)) - 0.5) * 0.08
    img[..., :3] = np.clip(img[..., :3] * j, 0, 255)
    return img


def mip_atlas(img):
    """(TEX_H, ATLAS_W, 4) uint8: level 0 plus 4 coverage-preserving levels side by side."""
    out = np.zeros((TEX_H, ATLAS_W, 4), np.float64)
    lvl = img.astype(np.float64)
    for n, x0 in enumerate(LEVEL_X):
        h, w = lvl.shape[:2]
        out[:h, x0:x0 + w] = lvl
        if n == len(LEVEL_X) - 1:
            break
        a = lvl[..., 3:4] / 255.0
        blocks = lambda arr: arr.reshape(max(h // 2, 1), 2 if h > 1 else 1, w // 2, 2, arr.shape[-1])
        if h == 1:
            pa = (lvl[..., :3] * a).reshape(1, w // 2, 2, 3).sum(2)
            wa = a.reshape(1, w // 2, 2, 1).sum(2)
            amax = lvl[..., 3:4].reshape(1, w // 2, 2, 1).max(2)
        else:
            pa = blocks(lvl[..., :3] * a).sum((1, 3))
            wa = blocks(a).sum((1, 3))
            amax = blocks(lvl[..., 3:4]).max((1, 3))
        col = np.where(wa > 0, pa / np.maximum(wa, 1e-9), 0)
        lvl = np.concatenate([col, amax], -1)
    return np.clip(out, 0, 255).astype(np.uint8)


def texture_layers():
    return [mip_atlas(sprite(v)) for v in range(len(FEATHERS))]


def build_mesh():
    """Two crossed quads along local +X (tip at +TIP). Vertex: pos3 normal3 uv2."""
    x0, x1 = -TIP, TIP
    up = SHAFT_ROW + 0.5                 # texels above the axis
    dn = TEX_H - up                      # texels below the axis
    t, b = up * PX, -dn * PX
    v = []

    def quad(p, n):
        (a, bb, c, d) = p
        v.extend([a, bb, c, a, c, d])

    n1 = (0.0, 1.0, 0.0)                 # quad in the local XZ plane
    quad([(x0, 0, t, *n1, 0, 0), (x1, 0, t, *n1, 1, 0), (x1, 0, b, *n1, 1, 1), (x0, 0, b, *n1, 0, 1)], n1)
    n2 = (0.0, 0.0, 1.0)                 # quad in the local XY plane
    quad([(x0, t, 0, *n2, 0, 0), (x1, t, 0, *n2, 1, 0), (x1, b, 0, *n2, 1, 1), (x0, b, 0, *n2, 0, 1)], n2)
    return np.array(v, np.float32)


if __name__ == '__main__':
    import sys
    from PIL import Image
    atl = texture_layers()
    big = [np.kron(a, np.ones((24, 24, 1), np.uint8)) for a in atl]
    sheet = np.concatenate([np.pad(b, ((8, 8), (8, 8), (0, 0))) for b in big], 0)
    bg = np.zeros(sheet.shape[:2] + (3,), np.float64)
    bg[:] = (90, 150, 210)
    a = sheet[..., 3:4] / 255.0
    Image.fromarray((sheet[..., :3] * a + bg * (1 - a)).astype(np.uint8)).save(sys.argv[1])
    print('mesh verts', len(build_mesh()), 'length', LENGTH)
