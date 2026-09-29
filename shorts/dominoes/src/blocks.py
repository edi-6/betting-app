"""Block textures (16x16, drawn here) for the world's layers, and the dominoes' meshes and textures.

Dominoes are instanced props. A face whose layer is >= 200 is painted with the domino's own colour, packed into the
instance's variant as r * 65536 + g * 256 + b, over a white texture with a little grain and a darker bevel at the
edges; other faces use a fixed texture. The run's dominoes are coloured all over; in the field only the south face
(the one that ends up on top) carries the picture and the rest is plain off-white, so the picture stays hidden until
it falls.
"""
import numpy as np

import layout as LY
import world as WD
from textures import make_block_textures

RNG = np.random.default_rng


def _speckle(base, var, rng, n=16):
    img = np.ones((n, n, 3)) * np.array(base, float)
    img *= 1.0 + (rng.random((n, n, 1)) - 0.5) * 2 * var
    return img


def world_textures(seed=4):
    rng = RNG(seed)
    t = make_block_textures()
    tex = {k: t[k] for k in ('grass_top', 'dirt', 'grass_side', 'stone', 'stone2', 'log', 'log_top', 'leaves')}
    # plains grass, a little less acid than the default
    for k in ('grass_top', 'grass_side', 'leaves'):
        g = tex[k].astype(float)
        grey = g.mean(-1, keepdims=True)
        tex[k] = grey + (g - grey) * 0.78
    tex['grass_top'] = tex['grass_top'] * np.array([0.95, 0.9, 0.9])
    # sand
    s = _speckle((219, 207, 160), 0.06, rng)
    d = rng.random((16, 16))
    s[d < 0.1] = (200, 188, 140)
    s[d > 0.94] = (232, 222, 180)
    tex['sand'] = s
    # water: blue with lighter ripple streaks (the renderer adds reflections and movement)
    yy, xx = np.mgrid[0:16, 0:16]
    w = np.ones((16, 16, 3)) * np.array((44, 92, 196), float)
    ripple = (np.sin((xx + 2 * np.sin(yy * 0.9)) * 0.8) > 0.55)
    w[ripple] = (70, 124, 222)
    w *= 1.0 + (rng.random((16, 16, 1)) - 0.5) * 0.08
    tex['water'] = w
    # oak planks: four boards, grain lines, nail dots
    p = np.ones((16, 16, 3)) * np.array((162, 130, 78), float)
    p *= 1.0 + (rng.random((16, 16, 1)) - 0.5) * 0.10
    for r in (3, 7, 11, 15):
        p[r] = (116, 90, 52)
    for r0 in (0, 4, 8, 12):
        c = int(rng.integers(3, 13))
        p[r0:r0 + 3, c] *= 0.82
        p[r0 + 1, (c + 5) % 16] *= 0.86
    tex['planks'] = p
    # birch log: white bark with black marks
    b = _speckle((216, 214, 204), 0.05, rng)
    for _ in range(9):
        r, c = int(rng.integers(0, 16)), int(rng.integers(0, 14))
        b[r, c:c + int(rng.integers(2, 4))] = (40, 40, 38)
    tex['birch_log'] = b
    # birch leaves: lighter, yellower green
    bl = _speckle((104, 150, 64), 0.18, rng)
    d = rng.random((16, 16))
    bl[d < 0.15] = (58, 96, 36)
    bl[d > 0.9] = (140, 186, 92)
    tex['birch_leaves'] = bl
    # gravel (the riverbed)
    g = _speckle((128, 122, 118), 0.16, rng)
    d = rng.random((16, 16))
    g[d < 0.18] = (96, 90, 86)
    g[d > 0.85] = (160, 152, 146)
    tex['gravel'] = g
    # snow
    tex['snow'] = _speckle((238, 244, 250), 0.03, rng)
    ss = tex['dirt'].copy()
    lip = 4 + (rng.random(16) < 0.5).astype(int)
    for c in range(16):
        ss[:lip[c], c] = tex['snow'][:lip[c], c]
    tex['snow_side'] = ss
    # the crater: scorched dirt
    cr = tex['dirt'] * 0.55
    d = rng.random((16, 16))
    cr[d < 0.2] = (40, 30, 24)
    tex['crater'] = cr
    return [np.clip(tex[n], 0, 255).astype(np.uint8) for n in WD.LAYERS]


# ---------------------------------------------------------------------------------------------
# dominoes
# ---------------------------------------------------------------------------------------------
TINTED = 200                      # face layer >= TINTED: painted with the instance's colour
L_PAINT, L_BACKING = 0, 1         # texture layers: white grain (tinted), plain off-white (not tinted)


def domino_textures(seed=6):
    rng = RNG(seed)
    yy, xx = np.mgrid[0:16, 0:16]
    edge = np.minimum.reduce([xx, 15 - xx, yy, 15 - yy])
    bevel = np.where(edge == 0, 0.8, np.where(edge == 1, 0.93, 1.0))
    paint = 255.0 * (1.0 + (rng.random((16, 16)) - 0.5) * 0.05) * bevel
    back = np.array((236, 232, 222), float) * (1.0 + (rng.random((16, 16, 1)) - 0.5) * 0.05) * bevel[..., None]
    out = []
    for img in (np.repeat(paint[..., None], 3, -1), back):
        a = np.full((16, 16, 1), 255.0)
        out.append(np.clip(np.concatenate([img, a], -1), 0, 255).astype(np.uint8))
    return out


def domino_mesh(picture_face_only):
    """A T x W x H box centred on the origin (local x = forwards, z = up); triangles pos3 nrm3 uv2 layer1.
    picture_face_only: only the -x face (the back, which ends up on top) is tinted."""
    hx, hy, hz = LY.T / 2, LY.W / 2, LY.H / 2
    tris = []
    faces = {
        'px': ((1, 0, 0), [(hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz), (hx, -hy, hz)]),
        'nx': ((-1, 0, 0), [(-hx, hy, -hz), (-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz)]),
        'py': ((0, 1, 0), [(hx, hy, -hz), (-hx, hy, -hz), (-hx, hy, hz), (hx, hy, hz)]),
        'ny': ((0, -1, 0), [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)]),
        'pz': ((0, 0, 1), [(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]),
        'nz': ((0, 0, -1), [(-hx, hy, -hz), (hx, hy, -hz), (hx, -hy, -hz), (-hx, -hy, -hz)]),
    }
    uv = [(0, 1), (1, 1), (1, 0), (0, 0)]
    for name, (n, q) in faces.items():
        if picture_face_only:
            layer = TINTED + L_PAINT if name == 'nx' else L_BACKING
        else:
            layer = TINTED + L_PAINT
        for k in (0, 1, 2, 0, 2, 3):
            tris.append((*q[k], *n, *uv[k], layer))
    return np.array(tris, np.float32)


def pack_colour(rgb):
    rgb = np.asarray(rgb, np.int64)
    return (rgb[..., 0] * 65536 + rgb[..., 1] * 256 + rgb[..., 2]).astype(np.float32)
