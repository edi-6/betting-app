"""The interface over the ride, in the game's own style: the speed on the action bar (above where the hotbar would
be), and the advancement toasts that slide in at the top right when the cart enters a new dimension
("We Need to Go Deeper", "The End?"), as the game shows them.
"""
import numpy as np

import pixelfont as PF
import ride as RD
import voxel as VX

_CACHE = {}


def _sprite(text, px, color, outline=1):
    k = (text, px, color, outline)
    if k not in _CACHE:
        _CACHE[k] = PF.render(text, px=px, color=color, outline=outline, outline_col=(10, 8, 14))
    return _CACHE[k]


def blit(img, spr, x, y, alpha=1.0):
    h, w = spr.shape[:2]
    H, W = img.shape[:2]
    x, y = int(round(x)), int(round(y))
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    s = spr[y0 - y:y1 - y, x0 - x:x1 - x]
    a = s[..., 3:4].astype(np.float32) / 255.0 * alpha
    reg = img[y0:y1, x0:x1].astype(np.float32)
    img[y0:y1, x0:x1] = (reg * (1 - a) + s[..., :3] * a).astype(np.uint8)


TOASTS = [('B', 0.7, 'We Need to Go Deeper', 'obsidian'), ('C', 0.7, 'The End?', 'end_stone')]


def _toast_sprite(title, icon, scale):
    """The advancement toast: a dark panel with a lighter rim, the item, yellow 'Advancement Made!' and the name."""
    k = ('toast', title, icon, scale)
    if k in _CACHE:
        return _CACHE[k]
    w, h = 160 * scale, 32 * scale
    img = np.zeros((h, w, 4), np.uint8)
    img[..., :3] = (33, 33, 33)
    img[..., 3] = 235
    b = scale
    img[:b, :, :3] = img[-b:, :, :3] = (85, 85, 85)
    img[:, :b, :3] = img[:, -b:, :3] = (85, 85, 85)
    img[b:2 * b, b:-b, :3] = (60, 60, 60)
    for (yy, xx) in ((0, 0), (0, w - b), (h - b, 0), (h - b, w - b)):
        img[yy:yy + b, xx:xx + b, 3] = 0                         # the rounded corners
    at = VX.atlas()
    tex = at.rgba[at[icon]]
    ic = np.kron(tex, np.ones((scale, scale, 1), np.uint8))
    y0 = (h - ic.shape[0]) // 2
    img[y0:y0 + ic.shape[0], 8 * scale:8 * scale + ic.shape[1]] = ic
    t1 = PF.render('Advancement Made!', px=scale, color=(255, 255, 0), shadow=True)
    t2 = PF.render(title, px=scale, color=(255, 255, 255), shadow=True)
    tx = 30 * scale
    for spr, ty in ((t1, 7 * scale), (t2, 18 * scale)):
        hh, ww = spr.shape[:2]
        ww = min(ww, w - tx - scale)
        a = spr[:hh, :ww, 3:4].astype(np.float32) / 255.0
        reg = img[ty:ty + hh, tx:tx + ww, :3].astype(np.float32)
        img[ty:ty + hh, tx:tx + ww, :3] = (reg * (1 - a) + spr[:hh, :ww, :3] * a).astype(np.uint8)
    _CACHE[k] = img
    return img


def draw(img, ride, f, sc):
    img = img.copy()
    H, W = img.shape[:2]
    k = W / 1080.0
    # the speed, on the action bar
    kmh = sc['v'] * 3.6
    if sc['seg'].name != 'D' or kmh > 30:
        txt = f'{int(round(kmh))} km/h'
        spr = _sprite(txt, max(2, int(round(6 * k))), (255, 255, 255))
        blit(img, spr, (W - spr.shape[1]) / 2, H * 0.842)
    # the toasts
    tv = f / RD.FPS
    for (seg, at, title, icon) in TOASTS:
        i = [sg.name for sg in ride.segments].index(seg)
        t = tv - (ride.starts[i] + at)
        if not (0 <= t < 3.4):
            continue
        spr = _toast_sprite(title, icon, max(2, int(round(4 * k))))
        sw = spr.shape[1]
        slide = min(t / 0.35, 1.0, (3.4 - t) / 0.35)
        slide = slide * slide * (3 - 2 * slide)
        x = W - sw * slide - 20 * k * slide
        blit(img, spr, x, 120 * k)
    return img
