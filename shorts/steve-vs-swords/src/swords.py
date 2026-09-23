"""Sword geometry: extruded pixel sprite (full detail) and a two-quad LOD, plus the texture array."""
import numpy as np
from textures import sword_mask, make_sword_sprites

PX = 0.11                     # world units per sprite pixel
MATERIALS = ['wood', 'stone', 'iron', 'gold', 'diamond', 'netherite']
SHINY = {'iron', 'gold', 'diamond', 'netherite'}
S2 = np.sqrt(0.5)
TIP_X = 9.9 * PX              # local x of the blade tip (just past the last blade pixel)
HILT_X = -9.9 * PX


def _loc(c, r):
    """Sprite corner coords (c right, r down, 0..16) -> local (X along blade towards tip, Z across)."""
    a = (c - 8.0) * PX
    b = (8.0 - r) * PX
    return (a + b) * S2, (b - a) * S2


def _v(c, r, y, n, uv):
    X, Z = _loc(c, r)
    return (X, y, Z, n[0], n[1], n[2], uv[0], uv[1])


def build_meshes():
    m = sword_mask() > 0
    t = PX * 0.5
    full = []
    lod = []

    def quad(dst, a, b, c, d):
        dst.extend([a, b, c, a, c, d])

    for dst in (full, lod):
        # front (-Y) and back (+Y) faces cover the whole sprite; transparent texels are discarded
        n = (0.0, -1.0, 0.0)
        quad(dst, _v(0, 0, -t, n, (0, 0)), _v(16, 0, -t, n, (1, 0)), _v(16, 16, -t, n, (1, 1)), _v(0, 16, -t, n, (0, 1)))
        n = (0.0, 1.0, 0.0)
        quad(dst, _v(0, 0, t, n, (0, 0)), _v(0, 16, t, n, (0, 1)), _v(16, 16, t, n, (1, 1)), _v(16, 0, t, n, (1, 0)))
    # side walls along every opaque/transparent pixel boundary (texture sampled at that pixel's centre)
    dirs = {
        'right': ((1, 0), (S2, 0.0, -S2)),
        'left': ((-1, 0), (-S2, 0.0, S2)),
        'up': ((0, -1), (S2, 0.0, S2)),
        'down': ((0, 1), (-S2, 0.0, -S2)),
    }
    for r in range(16):
        for c in range(16):
            if not m[r, c]:
                continue
            uv = ((c + 0.5) / 16.0, (r + 0.5) / 16.0)
            for name, ((dc, dr), n) in dirs.items():
                cc, rr = c + dc, r + dr
                if 0 <= cc < 16 and 0 <= rr < 16 and m[rr, cc]:
                    continue
                if name == 'right':
                    e0, e1 = (c + 1, r), (c + 1, r + 1)
                elif name == 'left':
                    e0, e1 = (c, r + 1), (c, r)
                elif name == 'up':
                    e0, e1 = (c, r), (c + 1, r)
                else:
                    e0, e1 = (c + 1, r + 1), (c, r + 1)
                quad(full, _v(*e0, -t, n, uv), _v(*e1, -t, n, uv), _v(*e1, t, n, uv), _v(*e0, t, n, uv))
    return np.array(full, np.float32), np.array(lod, np.float32)


def texture_layers():
    """RGBA sprites; alpha 255 = shiny metal/gem texel, 170 = rough (wood, stone, handle), 0 = empty."""
    sprites = make_sword_sprites()
    lab = sword_mask()
    out = []
    for name in MATERIALS:
        s = sprites[name].copy()
        opaque = s[..., 3] > 0
        handle = (lab >= 7)
        if name in SHINY:
            s[..., 3] = np.where(opaque, np.where(handle, 170, 255), 0)
        else:
            s[..., 3] = np.where(opaque, 170, 0)
        out.append(s)
    return out


if __name__ == '__main__':
    f, l = build_meshes()
    print('full verts', len(f), 'tris', len(f) // 3, 'lod tris', len(l) // 3)
    print('x range', f[:, 0].min(), f[:, 0].max(), 'z range', f[:, 2].min(), f[:, 2].max())
