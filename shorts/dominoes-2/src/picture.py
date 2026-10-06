"""The picture hidden in the field: LOOK / BEHIND / YOU in blood red on black, in the game's font with its drop
shadow, and the O's of LOOK are Herobrine's eyes: white, and at the end they glow.

It's anamorphic, like 3D street art: the words are laid out on the screen of the last shot (from the top of the
cliff, looking south over the field) and projected back down onto the field, so from up there they read straight,
the near word big and the far ones as tall as they look; from anywhere else they stretch away. The field falls
towards the cliff, so the words appear in reading order: LOOK, then BEHIND, then YOU.

picture(x, y) -> (N, 3) uint8: the colour of the field domino standing at (x, y); eyes(x, y) -> (N,) bool: the
dominoes that glow.
"""
import numpy as np

import layout as LY
import pixelfont as PF

SW, SH = 1080, 1920                         # the screen the words are laid out on
BG = np.array([17.0, 16.0, 22.0])
RED = np.array([196.0, 22.0, 20.0])
RED_SHADOW = np.array([70.0, 8.0, 10.0])
EYE = np.array([255.0, 255.0, 255.0])
# word, centre on screen (y px), width on screen (px)
WORDS = (('LOOK', 470.0, 800.0), ('BEHIND', 835.0, 880.0), ('YOU', 1265.0, 690.0))


def final_camera():
    """eye, forwards, right, up (unit vectors) and the vertical field of view of the last shot."""
    eye = np.array(LY.FINAL_EYE, float)
    p = np.radians(LY.FINAL_PITCH)
    f = np.array([0.0, -np.cos(p), np.sin(p)])
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return eye, f, r, u, LY.FINAL_FOV


def project(P, cam=None):
    """World points (N, 3) -> screen (N, 2) px on the SW x SH screen of the last shot."""
    eye, f, r, u, fov = cam or final_camera()
    d = np.asarray(P, float) - eye
    z = d @ f
    ty = np.tan(np.radians(fov) / 2)
    tx = ty * SW / SH
    sx = SW / 2 + (d @ r) / z / tx * SW / 2
    sy = SH / 2 - (d @ u) / z / ty * SH / 2
    return np.stack([sx, sy], -1)


def screen_layout():
    """The words on the screen: (colour (SH, SW, 3), kind (SH, SW) 0 background / 1 letter / 2 eye / 3 shadow)."""
    col = np.zeros((SH, SW, 3)) + BG
    kind = np.zeros((SH, SW), np.int8)
    yy, xx = np.mgrid[0:SH, 0:SW]
    # a little darker towards the edges (the field reads as a hole in the meadow)
    v = np.hypot((xx - SW / 2) / SW, (yy - SH * 0.55) / SH)
    col *= (1.0 - 0.35 * np.clip(v * 1.6, 0, 1))[..., None]
    for word, cy, width in WORDS:
        glyphs = [PF.text_mask(ch)[:7] for ch in word]
        units = sum(g.shape[1] for g in glyphs) + (len(glyphs) - 1)
        px = width / units
        x = SW / 2 - width / 2
        y0 = cy - 3.5 * px
        for ch, g in zip(word, glyphs):
            h, w = g.shape
            # the shadow (a font pixel down and right), then the letter
            for dx, dy, c, k in ((1, 1, RED_SHADOW, 3), (0, 0, RED, 1)):
                for (r_, c_) in zip(*np.nonzero(g)):
                    xa, ya = int(round(x + (c_ + dx) * px)), int(round(y0 + (r_ + dy) * px))
                    xb, yb = int(round(x + (c_ + dx + 1) * px)), int(round(y0 + (r_ + dy + 1) * px))
                    col[ya:yb, xa:xb] = c
                    kind[ya:yb, xa:xb] = k
            if word == 'LOOK' and ch == 'O':
                # the eyes: the inside of each O, white
                xa, ya = int(round(x + 1 * px)), int(round(y0 + 1 * px))
                xb, yb = int(round(x + (w - 1) * px)), int(round(y0 + (h - 1) * px))
                col[ya:yb, xa:xb] = EYE
                kind[ya:yb, xa:xb] = 2
            x += (w + 1) * px
    return col, kind


_CACHE = {}


def _lookup(x, y):
    if 'img' not in _CACHE:
        _CACHE['img'] = screen_layout()
    col, kind = _CACHE['img']
    # where each domino's picture face ends up once it's down: lying forwards (north) on the next one
    P = np.stack([np.asarray(x, float), np.asarray(y, float) + 0.5, np.full(np.shape(x), 0.22)], -1)
    s = project(P)
    sx = np.clip(np.rint(s[:, 0]).astype(int), 0, SW - 1)
    sy = np.clip(np.rint(s[:, 1]).astype(int), 0, SH - 1)
    off = (s[:, 0] < 0) | (s[:, 0] >= SW) | (s[:, 1] < 0) | (s[:, 1] >= SH)
    c = col[sy, sx]
    k = kind[sy, sx]
    c[off] = BG * 0.7
    k[off] = 0
    return c, k


def picture(x, y, seed=3):
    c, _ = _lookup(x, y)
    rng = np.random.default_rng(seed)
    # each domino a little different, like real painted dominoes
    c = c * (1.0 + (rng.random((len(c), 1)) - 0.5) * 0.08)
    return np.clip(c, 0, 255).astype(np.uint8)


def eyes(x, y):
    return _lookup(x, y)[1] == 2


def kinds(x, y):
    """Per field domino: 0 background, 1 a red letter (glows at the end), 2 an eye (glows), 3 a letter's shadow."""
    return _lookup(x, y)[1]


if __name__ == '__main__':
    import sys
    from PIL import Image
    L = LY.Layout(picture=picture)
    f = L.field
    img = np.zeros((L.nrow, L.ncol, 3), np.uint8)
    c = np.rint(f['x'] / LY.PITCH + (L.ncol - 1) / 2).astype(int)
    img[L.nrow - 1 - f['k'], c] = f['face']
    out = sys.argv[1] if len(sys.argv) > 1 else 'pic.png'
    Image.fromarray(img).resize((L.ncol * 3, L.nrow * 3), Image.NEAREST).save(out)
    col, kind = screen_layout()
    Image.fromarray(col.astype(np.uint8)).resize((540, 960)).save(out.replace('.png', '_screen.png'))
    s = project(np.array([[0, LY.FIELD_Y0, 0], [0, L.field_y1, 0], [-76.8, LY.FIELD_Y0, 0], [76.8, LY.FIELD_Y0, 0],
                          [0, 150, 0]]))
    print('field south edge, north edge, SW, SE corners, y=150 on screen:', np.round(s))
    print('eyes', int(eyes(f['x'], f['y']).sum()))
