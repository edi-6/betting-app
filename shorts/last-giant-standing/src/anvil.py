"""Anvil: the classic block-game anvil built from four boxes (base, foot, neck, working face) with textures drawn
in code: dark iron sides and a lighter working face in three states (intact, chipped, damaged), the way
anvils wear out in the game.

Model units: 16 pixels per anvil cell; the mesh is scaled to CELL world units. Vertex: pos3 normal3 uv2 layer1;
faces that show the working surface use layer 100 + n so the instance variant (0-2) picks the wear state.
"""
import numpy as np

CELL = 1.4                         # world size of one anvil (the grid it stacks on)

# boxes in model pixels (x0, y0, z0, x1, y1, z1), z up, long axis along y
BOXES = [
    ('base', (2, 2, 0, 14, 14, 4)),
    ('foot', (4, 3, 4, 12, 13, 5)),
    ('neck', (6, 4, 5, 10, 12, 10)),
    ('face', (3, 0, 10, 13, 16, 16)),
]
L_SIDE, L_BASE, L_TOP = 0, 1, 2     # layers: iron side, dark base, working face (+ variant 0..2)


def _px(base, var, rng, n=16):
    img = np.array(base, float)[None, None, :] * (1 + (rng.random((n, n, 1)) - 0.5) * 2 * var)
    return img


def textures(seed=5):
    """List of 16x16 RGBA layers: side, base, top (intact), top (chipped), top (damaged)."""
    rng = np.random.default_rng(seed)
    out = []
    # side: dark iron with a bright bevel on the upper edge and a darker lower lip
    s = _px((72, 72, 78), 0.06, rng)
    s[0:1] = (122, 122, 130)
    s[1:2] = (98, 98, 106)
    s[-2:] = (50, 50, 56)
    streak = rng.random(16) < 0.25
    s[2:-2, streak] *= 0.9
    out.append(s)
    # base block: darker, heavier iron
    b = _px((58, 58, 64), 0.07, rng)
    b[0:1] = (92, 92, 100)
    b[-1:] = (40, 40, 46)
    out.append(b)
    # working face: lighter, worn smooth, bright rim
    top = _px((104, 104, 112), 0.05, rng)
    top[0, :] = top[-1, :] = (134, 134, 142)
    top[:, 0] = top[:, -1] = (134, 134, 142)
    top[4:12, 5:11] *= 1.06
    crack_col = (44, 44, 50)
    for level in range(3):
        t = top.copy()
        if level >= 1:
            for (r, c) in ((3, 4), (4, 4), (4, 5), (5, 6), (6, 6), (10, 11), (11, 11), (11, 12)):
                t[r, c] = crack_col
        if level >= 2:
            for (r, c) in ((7, 2), (8, 3), (8, 4), (9, 4), (10, 5), (2, 10), (3, 10), (3, 11), (4, 12), (12, 8),
                           (13, 8), (13, 9), (6, 9), (7, 10)):
                t[r, c] = crack_col
            t *= 0.94
        out.append(t)
    return [np.concatenate([np.clip(t, 0, 255), np.full((16, 16, 1), 255.0)], -1).astype(np.uint8) for t in out]


def build_mesh():
    """Triangles of the anvil model, centred on x/y with its base at z = 0, scaled to CELL."""
    v = []
    s = CELL / 16.0
    for name, (x0, y0, z0, x1, y1, z1) in BOXES:
        side_layer = L_BASE if name == 'base' else L_SIDE
        faces = [
            ((1, 0, 0), [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)], (y0, z0, y1, z1), side_layer),
            ((-1, 0, 0), [(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)], (y0, z0, y1, z1), side_layer),
            ((0, 1, 0), [(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)], (x0, z0, x1, z1), side_layer),
            ((0, -1, 0), [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], (x0, z0, x1, z1), side_layer),
            ((0, 0, 1), [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], (x0, y0, x1, y1),
             100 + L_TOP if name == 'face' else side_layer),
            ((0, 0, -1), [(x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (x0, y0, z0)], (x0, y0, x1, y1), L_BASE),
        ]
        for nrm, quad, (u0, w0, u1, w1), layer in faces:
            # texture: 1 texel per model pixel; the v axis runs down the face (row 0 at the top edge)
            uv = [(u0 / 16, 1 - w0 / 16), (u1 / 16, 1 - w0 / 16), (u1 / 16, 1 - w1 / 16), (u0 / 16, 1 - w1 / 16)]
            if nrm[2] != 0:
                uv = [(u0 / 16, w0 / 16), (u1 / 16, w0 / 16), (u1 / 16, w1 / 16), (u0 / 16, w1 / 16)]
            pts = [((x - 8) * s, (y - 8) * s, z * s) for (x, y, z) in quad]
            for a, b, c in ((0, 1, 2), (0, 2, 3)):
                for k in (a, b, c):
                    v.append((*pts[k], *nrm, *uv[k], float(layer)))
    return np.array(v, np.float32)


HEIGHT = 16 * CELL / 16.0          # model height in world units


if __name__ == '__main__':
    import sys
    from PIL import Image
    layers = textures()
    big = [np.kron(a[..., :3], np.ones((16, 16, 1), np.uint8)) for a in layers]
    Image.fromarray(np.concatenate([np.pad(b, ((6, 6), (6, 6), (0, 0)), constant_values=255) for b in big], 1)).save(sys.argv[1])
    m = build_mesh()
    print('triangles', len(m) // 3, 'height', HEIGHT)
