"""Whole Minecraft blocks torn out of the world (grass, dirt, stone, log, leaves), drawn as an instanced prop:
a unit cube whose faces pick their texture layer from the block kind (instance variant = 3 x kind)."""
import numpy as np

from textures import make_block_textures

KINDS = ('grass', 'dirt', 'stone', 'log', 'leaves')
K_GRASS, K_DIRT, K_STONE, K_LOG, K_LEAVES = range(5)
_FACES = {'grass': ('grass_top', 'grass_side', 'dirt'), 'dirt': ('dirt', 'dirt', 'dirt'),
          'stone': ('stone', 'stone', 'stone'), 'log': ('log_top', 'log', 'log_top'),
          'leaves': ('leaves', 'leaves', 'leaves')}


def texture_layers():
    """RGBA uint8 16x16 layers, three per kind (top, side, bottom)."""
    tex = make_block_textures()
    out = []
    for k in KINDS:
        for name in _FACES[k]:
            img = np.clip(np.asarray(tex[name], np.float64)[..., :3], 0, 255)
            out.append(np.concatenate([img, np.full((16, 16, 1), 255.0)], -1).astype(np.uint8))
    return out


def build_mesh():
    """A unit block centred on the origin: triangles with pos3 normal3 uv2 layer1; layer 100 + (0 top, 1 side,
    2 bottom) so the instance variant selects the kind."""
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
