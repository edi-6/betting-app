"""2D overlay for the black hole: the BLOCKS EATEN counter with the giants' portraits, the SIZE banners and
the EATEN stamps. Sprites are built once with PIL and composited onto the rendered frames in numpy."""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from models import face_portrait, SPECS
from textures import upscale

FONT_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets', 'fonts')
GOLD = (255, 200, 61)
RED = (255, 62, 52)
INK = (16, 16, 22)

CROWN_ROWS = [
    "...O.....O.....O...",
    "..OLO...OLO...OLO..",
    "..OYO..OLYYO..OYO..",
    ".OLYYO.OLYYO.OLYYO.",
    ".OLYYYOLYYYYOLYYYO.",
    ".OLYYYYYYYYYYYYYYO.",
    ".OYYRYYYYBYYYYRYYO.",
    ".OYYYYYYYYYYYYYYYO.",
    ".ODDDDDDDDDDDDDDDO.",
    "..OOOOOOOOOOOOOOO..",
]
CROWN_PAL = {'O': (46, 28, 6), 'Y': (255, 204, 58), 'L': (255, 238, 160), 'D': (206, 138, 24),
             'R': (232, 40, 64), 'B': (70, 150, 255)}


def crown_pixels():
    h, w = len(CROWN_ROWS), len(CROWN_ROWS[0])
    img = np.zeros((h, w, 4), np.uint8)
    for r, row in enumerate(CROWN_ROWS):
        for c, ch in enumerate(row):
            if ch != '.':
                img[r, c, :3] = CROWN_PAL[ch]
                img[r, c, 3] = 255
    return img


def thin_x(n=16):
    """A slimmer red X (1-pixel strokes with a dark outline) to lay over a portrait."""
    yy, xx = np.mgrid[0:n, 0:n]
    core = ((yy == xx) | (yy + xx == n - 1)) & (yy >= 1) & (yy <= n - 2)
    core |= ((np.abs(yy - xx) == 1) | (np.abs(yy + xx - (n - 1)) == 1)) & (yy >= 2) & (yy <= n - 3) & (xx >= 2) & (xx <= n - 3)
    grown = core.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            grown |= np.roll(np.roll(core, dy, 0), dx, 1)
    img = np.zeros((n, n, 4), np.uint8)
    img[grown] = (26, 8, 8, 235)
    img[core] = (236, 36, 36, 255)
    return img


def font(weight, size):
    return ImageFont.truetype(os.path.join(FONT_DIR, f'Montserrat-{weight}.ttf'), size)


def _blur_alpha(a, radius):
    return np.array(Image.fromarray(a).filter(ImageFilter.GaussianBlur(radius)))


def with_shadow(sprite, blur=6, offset=(0, 5), strength=0.5, pad=18):
    """RGBA uint8 sprite with a soft drop shadow baked in (padded by `pad`)."""
    h, w = sprite.shape[:2]
    H, W = h + 2 * pad, w + 2 * pad
    sh = np.zeros((H, W), np.uint8)
    sh[pad + offset[1]:pad + offset[1] + h, pad + offset[0]:pad + offset[0] + w] = sprite[..., 3]
    sa = _blur_alpha(sh, blur).astype(np.float32) / 255.0 * strength
    sp = np.zeros((H, W, 4), np.float32)
    sp[pad:pad + h, pad:pad + w] = sprite
    a = sp[..., 3] / 255.0
    out_a = a + sa * (1 - a)
    out = np.zeros((H, W, 4), np.float32)
    out[..., :3] = sp[..., :3] * (a / np.maximum(out_a, 1e-6))[..., None]
    out[..., 3] = out_a * 255
    return np.clip(out, 0, 255).astype(np.uint8)


def text_sprite(text, fnt, fill=(255, 255, 255), stroke=0, stroke_fill=INK, tracking=0, shadow=True,
                gradient=None):
    """Text as an RGBA sprite, with letter spacing, outline, optional vertical colour gradient and shadow."""
    widths = [fnt.getlength(ch) for ch in text]
    total = int(sum(widths) + tracking * (len(text) - 1)) + 2 * stroke + 8
    asc, desc = fnt.getmetrics()
    hgt = asc + desc + 2 * stroke + 8
    img = Image.new('RGBA', (total, hgt), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    x = stroke + 4
    for ch, w in zip(text, widths):
        d.text((x, stroke + 4), ch, font=fnt, fill=fill + (255,), stroke_width=stroke,
               stroke_fill=stroke_fill + (255,))
        x += w + tracking
    arr = np.array(img)
    if gradient is not None:
        # recolour the fill (not the outline): vertical gradient top -> bottom
        m = Image.new('L', (total, hgt), 0)
        dm = ImageDraw.Draw(m)
        x = stroke + 4
        for ch, w in zip(text, widths):
            dm.text((x, stroke + 4), ch, font=fnt, fill=255)
            x += w + tracking
        m = np.array(m).astype(np.float32) / 255.0
        ys = np.nonzero(m.max(1) > 0)[0]
        y0, y1 = (ys.min(), ys.max()) if len(ys) else (0, hgt - 1)
        u = np.clip((np.arange(hgt) - y0) / max(1, y1 - y0), 0, 1)[:, None, None]
        top, bot = np.array(gradient[0], np.float32), np.array(gradient[1], np.float32)
        col = top * (1 - u) + bot * u
        arr = arr.astype(np.float32)
        arr[..., :3] = arr[..., :3] * (1 - m[..., None]) + col * m[..., None]
        arr = arr.astype(np.uint8)
    # trim
    ys, xs = np.nonzero(arr[..., 3] > 0)
    if len(ys):
        arr = arr[max(0, ys.min() - 2):ys.max() + 3, max(0, xs.min() - 2):xs.max() + 3]
    return with_shadow(arr, blur=6, offset=(0, 5), strength=0.55, pad=16) if shadow else arr


def rounded_panel(w, h, r, fill, border=None):
    img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=r, fill=fill, outline=border, width=2 if border else 0)
    return np.array(img)


def over(dst, src, x, y, opacity=1.0):
    """Alpha-composite RGBA sprite onto float RGB frame at integer position (x, y)."""
    if opacity <= 0.0:
        return
    h, w = src.shape[:2]
    H, W = dst.shape[:2]
    x, y = int(round(x)), int(round(y))
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    a = s[..., 3:4] / 255.0 * opacity
    dst[y0:y1, x0:x1] = dst[y0:y1, x0:x1] * (1 - a) + s[..., :3] * a


def add(dst, src, x, y, gain=1.0):
    """Additive glow sprite (RGBA, alpha used as weight)."""
    h, w = src.shape[:2]
    H, W = dst.shape[:2]
    x, y = int(round(x)), int(round(y))
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    dst[y0:y1, x0:x1] += s[..., :3] * (s[..., 3:4] / 255.0) * gain


def scaled(sprite, s, resample=Image.BICUBIC):
    if abs(s - 1.0) < 1e-3:
        return sprite
    img = Image.fromarray(sprite)
    return np.array(img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), resample))


def rotated(sprite, deg):
    if abs(deg) < 0.05:
        return sprite
    return np.array(Image.fromarray(sprite).rotate(deg, resample=Image.BICUBIC, expand=True))


def grey(sprite, k=1.0, dark=0.55):
    out = sprite.astype(np.float32)
    g = out[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32)
    out[..., :3] = out[..., :3] * (1 - k) + (g[..., None] * dark) * k
    return np.clip(out, 0, 255).astype(np.uint8)


def ease_out_back(t, s=1.9):
    t = np.clip(t, 0.0, 1.0) - 1.0
    return 1.0 + (s + 1.0) * t ** 3 + s * t ** 2


def smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def glow_sprite(w, h, col, radius):
    img = np.zeros((h, w), np.uint8)
    Image.fromarray(img)
    a = Image.new('L', (w, h), 0)
    m = int(min(radius * 2.2, 0.4 * min(w, h)))
    ImageDraw.Draw(a).ellipse((m, m, w - m, h - m), fill=255)
    a = np.array(a.filter(ImageFilter.GaussianBlur(radius * 0.8)))
    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = col
    out[..., 3] = a
    return out


PURPLE = (176, 112, 255)
LILAC = (214, 190, 255)


class HUD:
    """The black hole's overlay: BLOCKS EATEN counter with the four giants' portraits under it (struck out as
    they are eaten), the SIZE banners, the EATEN stamp, and full-frame flash / darkness / vignette."""
    ORDER = ('zombie', 'creeper', 'steve', 'warden')

    def __init__(self, W=1080, H=1920):
        self.W, self.H = W, H
        self.k = W / 1080.0
        k = self.k
        self._cache = {}
        # counter panel
        self.panel_w = int(round(620 * k))
        self.panel_h = int(round(262 * k))
        self.panel_y = int(round(150 * k))
        self.panel = with_shadow(rounded_panel(self.panel_w, self.panel_h, int(round(26 * k)), (10, 8, 22, 170),
                                               (176, 112, 255, 110)), blur=12, offset=(0, 7), strength=0.4,
                                 pad=int(round(22 * k)))
        self.label = text_sprite('BLOCKS EATEN', font(800, int(round(34 * k))), fill=LILAC, stroke=0,
                                 tracking=int(round(7 * k)), shadow=False)
        f_num = font(900, int(round(112 * k)))
        self.glyphs = {c: text_sprite(c, f_num, stroke=int(round(6 * k)), shadow=False,
                                      gradient=((255, 255, 255), (222, 196, 255))) for c in '0123456789,'}
        self.cell = max(g.shape[1] for c, g in self.glyphs.items() if c != ',') - int(round(14 * k))
        self.num_glow = glow_sprite(int(560 * k), int(220 * k), PURPLE, int(46 * k))
        # portraits
        self.port = int(round(58 * k))
        xs = thin_x()
        self.portraits, self.portraits_x = [], []
        for kind in self.ORDER:
            face = face_portrait(kind)
            p = np.array(Image.fromarray(face).resize((self.port, self.port), Image.NEAREST))
            framed = np.zeros((self.port + 8, self.port + 8, 4), np.uint8)
            framed[..., :3] = (20, 16, 30)
            framed[..., 3] = 255
            framed[2:-2, 2:-2, :3] = (226, 214, 250)
            framed[4:-4, 4:-4, :3] = p
            self.portraits.append(framed)
            xo = framed.copy()
            xo[4:-4, 4:-4] = grey(xo[4:-4, 4:-4], 0.9, 0.55)
            xx = np.array(Image.fromarray(xs).resize((self.port, self.port), Image.NEAREST))
            over_rgba(xo, xx, 4, 4)
            self.portraits_x.append(xo)
        # banners and stamps
        self.f_size = font(800, int(round(62 * k)))
        self.f_big = font(900, int(round(200 * k)))
        self.f_stamp = font(900, int(round(118 * k)))
        self.f_name = font(800, int(round(64 * k)))
        self.glow_purple = glow_sprite(int(900 * k), int(560 * k), PURPLE, int(80 * k))
        self.glow_red = glow_sprite(int(640 * k), int(420 * k), RED, int(60 * k))
        self.ring = self._ring_sprite(int(420 * k), int(10 * k))
        ys, xs_ = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.hypot((xs_ - W / 2) / (W / 2), (ys - H / 2) / (H / 2)) / np.sqrt(2)
        self.vig = np.clip((r - 0.35) / 0.65, 0, 1) ** 1.6

    @staticmethod
    def _ring_sprite(size, width):
        yy, xx = np.mgrid[0:size, 0:size] - size / 2
        r = np.hypot(xx, yy)
        a = np.exp(-((r - size / 2 + width * 1.5) / width) ** 2)
        out = np.zeros((size, size, 4), np.uint8)
        out[..., :3] = (226, 196, 255)
        out[..., 3] = (np.clip(a, 0, 1) * 255).astype(np.uint8)
        return out

    def _text(self, key, *a, **kw):
        if key not in self._cache:
            self._cache[key] = text_sprite(*a, **kw)
        return self._cache[key]

    def draw(self, frame, st):
        """frame: HxWx3 uint8. st: dict with
        counter: (value, pulse 0..1, vis 0..1) ; eaten: {giant index: seconds since eaten} ;
        size: (label, t) ; stamp: (giant index, t) ; flash ; dark ; vignette."""
        out = frame.astype(np.float32)
        if st.get('dark', 0.0) > 0:
            out *= 1.0 - st['dark']
        if st.get('vignette', 0.0) > 0:
            out *= (1.0 - st['vignette'] * self.vig)[..., None]
        if st.get('counter') is not None:
            self._counter(out, *st['counter'], st.get('eaten', {}))
        if st.get('size') is not None:
            self._size(out, *st['size'])
        if st.get('stamp') is not None:
            self._stamp(out, *st['stamp'])
        if st.get('flash', 0.0) > 0:
            out = out * (1 - st['flash']) + 255.0 * st['flash']
        return np.clip(out, 0, 255).astype(np.uint8)

    # -------------------------------------------------------------------------------------------
    def _number(self, value):
        txt = f'{int(value):,}'
        widths = [self.glyphs[c].shape[1] - int(round(26 * self.k)) if c == ',' else self.cell for c in txt]
        h = max(g.shape[0] for g in self.glyphs.values())
        out = np.zeros((h, sum(widths) + int(round(40 * self.k)), 4), np.uint8)
        x = int(round(10 * self.k))
        for c, w in zip(txt, widths):
            g = self.glyphs[c]
            gx = x + (w - g.shape[1]) // 2
            y = h - g.shape[0]
            if c == ',':
                gx = x - int(round(14 * self.k))
            over_rgba_clip(out, g, gx, y)
            x += w
        return out

    def _counter(self, out, value, pulse, vis, eaten):
        k = self.k
        if vis <= 0:
            return
        e = smooth(vis)
        pp = int(round(22 * k))
        x0 = (self.W - self.panel_w) // 2
        y0 = self.panel_y - int(30 * k * (1 - e))
        over(out, self.panel, x0 - pp, y0 - pp, e)
        lb = self.label
        over(out, lb, (self.W - lb.shape[1]) // 2, y0 + int(20 * k), e)
        num = self._number(value)
        s = 1.0 + 0.12 * pulse
        if pulse > 0.01:
            add(out, self.num_glow, (self.W - self.num_glow.shape[1]) // 2,
                y0 + int(112 * k) - self.num_glow.shape[0] // 2, gain=0.55 * pulse * e)
        n = scaled(num, s)
        over(out, n, (self.W - n.shape[1]) // 2, y0 + int(112 * k) - n.shape[0] // 2, e)
        # the giants, struck out once eaten
        P = self.port + 8
        gap = int(round(22 * k))
        rw = 4 * P + 3 * gap
        px = (self.W - rw) // 2
        py = y0 + self.panel_h - P - int(16 * k)
        for i in range(4):
            te = eaten.get(i)
            x = px + i * (P + gap)
            if te is None:
                over(out, self.portraits[i], x, py, e)
                continue
            if te < 0.28:
                # sucked into a point, spinning
                u = smooth(te / 0.28)
                spr = rotated(scaled(self.portraits[i], max(0.05, 1.0 - u)), 540 * u)
                over(out, spr, x + (P - spr.shape[1]) / 2, py + (P - spr.shape[0]) / 2, e)
            else:
                u = smooth((te - 0.28) / 0.2)
                s2 = 1.25 - 0.25 * u
                spr = scaled(self.portraits_x[i], s2, Image.NEAREST)
                over(out, spr, x + (P - spr.shape[1]) / 2, py + (P - spr.shape[0]) / 2, e * min(1.0, 0.3 + u))

    def _size(self, out, label, t):
        """SIZE banner: the number slams in over a purple glow and a ring; ~1.6 s."""
        k = self.k
        if t < 0 or t > 1.6:
            return
        small = self._text(('size',), 'SIZE', self.f_size, fill=GOLD, stroke=int(round(5 * k)),
                           tracking=int(round(12 * k)))
        big = self._text(('big', label), label, self.f_big, stroke=int(round(10 * k)), tracking=int(round(4 * k)),
                         gradient=((255, 255, 255), (206, 164, 255)))
        a = min(1.0, t / 0.05) * (1.0 - smooth((t - 1.32) / 0.28))
        s = 2.3 - 1.3 * smooth(t / 0.17) if t < 0.17 else 1.0 + 0.04 * np.exp(-(t - 0.17) / 0.12)
        cy = int(0.285 * self.H)
        rng = np.random.default_rng(int(t * 1000))
        shake = (rng.normal(0, 7 * k, 2) * max(0.0, 1.0 - (t - 0.17) / 0.3)) if t > 0.17 else np.zeros(2)
        add(out, self.glow_purple, (self.W - self.glow_purple.shape[1]) // 2,
            cy + int(40 * k) - self.glow_purple.shape[0] // 2, gain=0.55 * a * (0.6 + 0.4 * np.exp(-t / 0.3)))
        if 0.15 < t < 0.75:
            u = (t - 0.15) / 0.6
            rg = scaled(self.ring, 0.4 + 2.4 * u)
            add(out, rg, (self.W - rg.shape[1]) // 2, cy + int(50 * k) - rg.shape[0] // 2, gain=0.9 * (1 - u))
        b = scaled(big, s)
        over(out, b, (self.W - b.shape[1]) // 2 + shake[0], cy + int(58 * k) - b.shape[0] // 2 + shake[1], a)
        sm = scaled(small, 0.85 + 0.15 * smooth(t / 0.2))
        over(out, sm, (self.W - sm.shape[1]) // 2 + shake[0] * 0.5,
             cy - int(78 * k) - sm.shape[0] // 2 - int(40 * k * (1 - smooth(t / 0.2))), a)

    def _stamp(self, out, gi, t):
        """EATEN stamp: the portrait with a red X, the name, EATEN slammed in; ~1.5 s."""
        k = self.k
        if t < 0 or t > 1.5:
            return
        a = min(1.0, t / 0.05) * (1.0 - smooth((t - 1.25) / 0.25))
        s = 1.75 - 0.8 * smooth(t / 0.16) if t < 0.16 else 0.95 + 0.05 * smooth((t - 0.16) / 0.14)
        cy = int(0.70 * self.H)
        add(out, self.glow_red, (self.W - self.glow_red.shape[1]) // 2, cy - self.glow_red.shape[0] // 2,
            gain=0.32 * a)
        port = self._cache.get(('stamp_port', gi))
        if port is None:
            port = with_shadow(upscale(self.portraits_x[gi], 2), blur=8, offset=(0, 6), strength=0.5, pad=18)
            self._cache[('stamp_port', gi)] = port
        p = scaled(port, s, Image.NEAREST)
        over(out, p, (self.W - p.shape[1]) // 2, cy - int(150 * k) - p.shape[0] // 2, a)
        name = self._text(('name', gi), SPECS[self.ORDER[gi]]['label'], self.f_name, stroke=int(5 * k),
                          tracking=int(round(4 * k)))
        n = scaled(name, s)
        over(out, n, (self.W - n.shape[1]) // 2, cy + int(8 * k) - n.shape[0] // 2, a)
        stamp = self._text('stamp', 'EATEN', self.f_stamp, fill=RED, stroke=int(round(8 * k)),
                           tracking=int(round(6 * k)), gradient=((255, 110, 84), (212, 22, 40)))
        st = rotated(scaled(stamp, s), -4.0)
        over(out, st, (self.W - st.shape[1]) // 2, cy + int(108 * k) - st.shape[0] // 2, a)


def over_rgba_clip(dst, src, x, y):
    """over_rgba with clipping to the destination."""
    h, w = src.shape[:2]
    H, W = dst.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    over_rgba(dst[y0:y1, x0:x1], src[y0 - y:y1 - y, x0 - x:x1 - x], 0, 0)


_SPARK = {}


def sparkle(size):
    size = max(3, int(size))
    if size not in _SPARK:
        n = size * 4 + 1
        yy, xx = np.mgrid[0:n, 0:n] - n // 2
        r = np.hypot(xx, yy) + 1e-6
        star = np.exp(-np.abs(xx) / (0.18 * size) - np.abs(yy) / (1.4 * size)) + \
            np.exp(-np.abs(yy) / (0.18 * size) - np.abs(xx) / (1.4 * size)) + np.exp(-r / (0.5 * size))
        a = np.clip(star / star.max(), 0, 1)
        out = np.zeros((n, n, 4), np.uint8)
        out[..., :3] = (255, 236, 170)
        out[..., 3] = (a * 255).astype(np.uint8)
        _SPARK[size] = out
    return _SPARK[size]


def with_shadow_cached(hud, spr):
    key = ('shadow', id(spr))
    if key not in hud._cache:
        hud._cache[key] = with_shadow(spr, blur=8, offset=(0, 6), strength=0.5, pad=18)
    return hud._cache[key]


def over_rgba(dst, src, x, y):
    """Composite RGBA onto RGBA uint8 in place (for building sprites)."""
    h, w = src.shape[:2]
    d = dst[y:y + h, x:x + w].astype(np.float32)
    s = src.astype(np.float32)
    a = s[..., 3:4] / 255.0
    d[..., :3] = d[..., :3] * (1 - a) + s[..., :3] * a
    d[..., 3:4] = np.maximum(d[..., 3:4], s[..., 3:4])
    dst[y:y + h, x:x + w] = d.astype(np.uint8)
