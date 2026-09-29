"""Flat textured quads stuck to blocks: the text on signs, maps and items in item frames, painted words on walls,
labels and posters. Four instanced kinds, each with its own texture array:

  sign   128 x 64   black text, up to 4 lines (the game's sign)
  map    128 x 128  map images (maps.py)
  item   16 x 16    item icons (textures.item_icons)
  wall   512 x 512  painted text / labels / posters, stretched to the decal's size
"""
import numpy as np

import gfx
import maps as MP
import pixelfont as PF
import textures as TX

QUAD = None


def quad_mesh():
    """Unit quad in the x-z plane facing -y; uv row 0 at the top; layer -1 (from the instance)."""
    P = [(-0.5, 0, -0.5), (0.5, 0, -0.5), (0.5, 0, 0.5), (-0.5, 0, 0.5)]
    UV = [(0, 1), (1, 1), (1, 0), (0, 0)]
    out = []
    for k in (0, 1, 2, 0, 2, 3):
        out.append((*P[k], 0, -1, 0, *UV[k], -1))
    return np.array(out, np.float32)


def facing_quat(f):
    a = np.radians(90.0 * f) / 2
    return np.array([0.0, 0.0, np.sin(a), np.cos(a)])


def text_image(lines, size=(128, 64), color=(16, 12, 8), scale=1, line_h=10, shadow=False, font_px=1):
    W, H = size
    img = np.zeros((H, W, 4), np.uint8)
    masks = [PF.text_mask(ln) for ln in lines]
    total = len(lines) * line_h * font_px
    y = (H - total) // 2 + 1
    for m in masks:
        m = np.kron(m, np.ones((font_px, font_px), bool))
        h, w = m.shape
        x = (W - w) // 2
        if x < 0:
            m = m[:, -x:W - x]
            w = m.shape[1]
            x = 0
        sub = img[y:y + h, x:x + w]
        sub[m[:sub.shape[0], :sub.shape[1]]] = (*color, 255)
        y += line_h * font_px
    return img


def painted_text(lines, aspect, color=(88, 14, 12), res=512, rng=None):
    """Big uneven painted letters (smeared, dripping) for the wall, stretched into a square layer."""
    rng = rng or np.random.default_rng(2)
    W = 512
    H = max(32, int(round(W / aspect)))
    img = np.zeros((H, W, 4), np.uint8)
    masks = [PF.text_mask(ln) for ln in lines]
    wmax = max(m.shape[1] for m in masks)
    px = max(1, int(min((W - 16) / wmax, (H - 8) / (len(lines) * 9.5))))
    y = (H - len(lines) * 9 * px) // 2
    for m in masks:
        big = np.kron(m, np.ones((px, px), bool))
        h, w = big.shape
        x = (W - w) // 2
        # jitter the strokes a little, like a brush
        jit = big & (rng.random(big.shape) < 0.93)
        col = np.array(color, float) * rng.uniform(0.8, 1.1, (h, w))[..., None]
        sub = img[y:y + h, x:x + w]
        sub[jit, :3] = np.clip(col[jit], 0, 255)
        sub[jit, 3] = 255
        # drips
        for c in range(0, w, max(1, px)):
            col_on = np.nonzero(big[:, c])[0]
            if len(col_on) and rng.random() < 0.18:
                y1 = col_on[-1]
                L = int(rng.uniform(1.0, 3.0) * px)
                sub[y1:min(h, y1 + L), c:c + max(1, px // 3), :3] = color
                sub[y1:min(h, y1 + L), c:c + max(1, px // 3), 3] = 255
        y += 9 * px
    from PIL import Image
    return np.asarray(Image.fromarray(img).resize((res, res), Image.NEAREST))


def poster(kind, res=512, rng=None):
    rng = rng or np.random.default_rng(3)
    W, H = 100, 130
    img = np.zeros((H, W, 4), np.uint8)
    img[..., 3] = 255
    if kind == 'poster_space':
        img[..., :3] = (10, 14, 40)
        stars = rng.random((H, W)) < 0.02
        img[stars, :3] = 220
        yy, xx = np.mgrid[0:H, 0:W]
        d = np.hypot(xx - 50, yy - 62)
        img[d < 26, :3] = (190, 110, 60)
        img[(d < 26) & ((yy - 62 + 0.3 * (xx - 50)) % 9 < 3), :3] = (160, 80, 44)
        ring = (np.abs(((xx - 50) * 0.35 + (yy - 62)) ) < 3) & (d > 22) & (d < 44)
        img[ring, :3] = (220, 200, 160)
        m = PF.text_mask('OUT THERE')
        img[112:120, (W - m.shape[1]) // 2:(W - m.shape[1]) // 2 + m.shape[1]][m] = (230, 230, 230, 255)
    else:
        img[..., :3] = (18, 18, 20)
        m = np.kron(PF.text_mask('404'), np.ones((4, 4), bool))
        x = (W - m.shape[1]) // 2
        img[30:30 + m.shape[0], x:x + m.shape[1]][m] = (200, 30, 30, 255)
        m2 = PF.text_mask('NOT FOUND')
        x = (W - m2.shape[1]) // 2
        img[80:88, x:x + m2.shape[1]][m2] = (220, 220, 220, 255)
        m3 = PF.text_mask('WORLD TOUR')
        x = (W - m3.shape[1]) // 2
        img[100:108, x:x + m3.shape[1]][m3] = (150, 150, 150, 255)
    img[:2] = img[-2:] = (240, 240, 236, 255)
    img[:, :2] = img[:, -2:] = (240, 240, 236, 255)
    from PIL import Image
    return np.asarray(Image.fromarray(img).resize((res, res), Image.NEAREST))


WALL_TEXT = {
    'wall_sleep': (['YOU WILL SLEEP', 'HERE TONIGHT.'], (70, 10, 10)),
    'label_BEFORE': (['BEFORE'], (30, 22, 16)),
    'label_ABANDONED': (['ABANDONED'], (30, 22, 16)),
    'label_BURNED': (['BURNED'], (30, 22, 16)),
    'label_TODAY': (['TODAY'], (120, 20, 16)),
}


class Decals:
    """Builds the decal textures for a world and hands out their instances."""

    def __init__(self, r, w, maps=None, prefix=''):
        self.r = r
        self.w = w
        self.p = prefix
        mesh = quad_mesh()
        # signs
        sign_layers = []
        self.sign_idx = {}
        for s in w.signs:
            key = s.get('key') or '|'.join(s['lines'])
            if key not in self.sign_idx:
                self.sign_idx[key] = len(sign_layers)
                sign_layers.append(text_image(s['lines']))
        if not sign_layers:
            sign_layers.append(text_image(['']))
        r.add_kind(self.p + 'dec_sign', mesh, sign_layers)
        # maps
        mp = maps if maps is not None else MP.make_maps(w)
        self.map_names = sorted(mp)
        r.add_kind(self.p + 'dec_map', mesh, [mp[k] for k in self.map_names])
        self.maps = mp
        # the map in his hands: layer 0 is TODAY, layer 1 is redrawn per frame (with the red marks)
        if 'held_map' not in r.kinds:
            r.add_kind('held_map', mesh, [mp['TODAY'], mp['TODAY'].copy()])
        # items
        icons = TX.item_icons()
        self.item_names = sorted(icons)
        r.add_kind(self.p + 'dec_item', mesh, [icons[k] for k in self.item_names])
        # wall pieces
        wall_layers = []
        self.wall_idx = {}
        for d in w.decals:
            k = d['key']
            if k in self.wall_idx:
                continue
            self.wall_idx[k] = len(wall_layers)
            if k in WALL_TEXT:
                lines, col = WALL_TEXT[k]
                aspect = d['size'][0] / d['size'][1]
                wall_layers.append(painted_text(lines, aspect, col))
            else:
                wall_layers.append(poster(k))
        if not wall_layers:
            wall_layers.append(np.zeros((512, 512, 4), np.uint8))
        r.add_kind(self.p + 'dec_wall', mesh, wall_layers)
        self.static = self._build()

    def _inst(self, pos, facing, size, layer, emit=0.0):
        q = facing_quat(facing)
        return [*pos, *q, size[0], 1.0, size[1], layer, 1.0, 1.0, 1.0, emit, gfx.MAT_ENTITY]

    def _block_point(self, x, y, z, facing, lx, ly, lz):
        """A point in a block's local 1/16 units (facing south) rotated by facing, in world coordinates."""
        p = np.array([lx, ly, lz], float) / 16.0 - np.array([0.5, 0.5, 0.0])
        a = np.radians(90.0 * facing)
        c, s = np.cos(a), np.sin(a)
        p = np.array([c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]])
        return np.array([x + 0.5, y + 0.5, z], float) + p

    def _build(self):
        """Rows per kind, each tagged (sign key, 'frame:<name>@x,y,z', decal key) so shots can leave some out."""
        out = {'dec_sign': [], 'dec_map': [], 'dec_item': [], 'dec_wall': []}
        for s in self.w.signs:
            key = s.get('key') or '|'.join(s['lines'])
            x, y, z = s['pos']
            f = s['facing']
            if s['wall']:
                p = self._block_point(x, y, z, f, 8, 14 - 0.08, 8)
            else:
                p = self._block_point(x, y, z, f, 8, 7 - 0.08, 13)
            out['dec_sign'].append((key, self._inst(p, f, (0.95, 0.475), self.sign_idx[key])))
        for fr in self.w.frames:
            x, y, z = fr['pos']
            f = fr['facing']
            kind, name = fr['content']
            p = self._block_point(x, y, z, f, 8, 15 - 0.08, 8)
            tag = f'frame:{name}@{x},{y},{z}'
            if kind == 'map':
                out['dec_map'].append((tag, self._inst(p, f, (0.86, 0.86), self.map_names.index(name))))
            else:
                out['dec_item'].append((tag, self._inst(p, f, (0.55, 0.55), self.item_names.index(name))))
        for d in self.w.decals:
            out['dec_wall'].append((d['key'], self._inst(d['pos'], d['facing'], d['size'], self.wall_idx[d['key']])))
        return out

    def instances(self, exclude=()):
        """dict kind -> rows, without the tagged rows in exclude (a tag, or a prefix ending in '*')."""
        res = {}
        pre = [e[:-1] for e in exclude if e.endswith('*')]
        exact = set(e for e in exclude if not e.endswith('*'))
        for k, rows in self.static.items():
            keep = [row for (tag, row) in rows if tag not in exact and not any(tag.startswith(p) for p in pre)]
            if keep:
                res[self.p + k] = np.array(keep, np.float32).reshape(-1, 16)
        return res

    def sign_row(self, key, pos, facing, standing=True):
        """A text decal for a moving sign (the one that falls): pos = the sign block's origin corner."""
        x, y, z = pos
        p = self._block_point(x, y, z, facing, 8, (7 if standing else 14) - 0.08, 13 if standing else 8)
        return (self.p + 'dec_sign', self._inst(p, facing, (0.95, 0.475), self.sign_idx[key]))
