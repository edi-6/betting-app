"""2D overlay for the battle royale: the scoreboard (portrait, name and hearts of every giant), round banners,
fighter introductions, elimination stamps and the winner card. Sprites are built once with PIL and
composited onto the rendered frames in numpy."""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from models import face_portrait, SPECS
from textures import heart_pixels, upscale

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


class HUD:
    ORDER = ('zombie', 'creeper', 'steve', 'warden')

    def __init__(self, W=1080, H=1920):
        self.W, self.H = W, H
        self.k = W / 1080.0
        k = self.k
        self.hs = max(1, int(round(4 * k)))                       # heart pixel size
        self.heart_w = 9 * self.hs
        self.heart_gap = int(round(3 * k))
        self.hearts = {(st, fl): upscale(heart_pixels(st, fl), self.hs) for st in ('full', 'half', 'empty')
                       for fl in (False, True)}
        self.port = int(round(64 * k))
        self.row_h = int(round(76 * k))
        self.pad = int(round(16 * k))
        self.x0 = int(round(28 * k))
        self.y0 = int(round(176 * k))
        self.row_gap = int(round(10 * k))
        hearts_w = 10 * self.heart_w + 9 * self.heart_gap
        self.panel_w = self.pad + self.port + 8 + int(round(16 * k)) + hearts_w + self.pad
        self.panel_h = 2 * self.pad + 4 * self.row_h + 3 * self.row_gap
        self.panel_pad = int(round(20 * k))
        self._panels = {}
        f_name = font(800, int(round(29 * k)))
        self.portraits, self.portraits_x, self.names, self.names_out = [], [], [], []
        xs = thin_x()
        for kind in self.ORDER:
            face = face_portrait(kind)
            p = np.array(Image.fromarray(face).resize((self.port, self.port), Image.NEAREST))
            framed = np.zeros((self.port + 8, self.port + 8, 4), np.uint8)
            framed[..., :3] = (20, 20, 26)
            framed[..., 3] = 255
            framed[2:-2, 2:-2, :3] = (235, 235, 240)
            framed[4:-4, 4:-4, :3] = p
            self.portraits.append(framed)
            xo = framed.copy()
            xo[4:-4, 4:-4] = grey(xo[4:-4, 4:-4], 0.85, 0.75)
            xx = np.array(Image.fromarray(xs).resize((self.port, self.port), Image.NEAREST))
            over_rgba(xo, xx, 4, 4)
            self.portraits_x.append(xo)
            label = SPECS[kind]['label']
            self.names.append(text_sprite(label, f_name, tracking=int(round(1.5 * k)), shadow=False, stroke=0))
            self.names_out.append(text_sprite(label, f_name, fill=(150, 150, 156), tracking=int(round(1.5 * k)),
                                              shadow=False, stroke=0))
        # banner, intro, stamp, winner sprites
        self.f_round = font(800, int(round(50 * k)))
        self.f_title = font(900, int(round(116 * k)))
        self.f_intro = font(900, int(round(150 * k)))
        self.f_stamp = font(900, int(round(104 * k)))
        self.f_stamp_name = font(800, int(round(64 * k)))
        self.f_win = font(900, int(round(136 * k)))
        self._cache = {}
        self.crown = crown_pixels()
        self.crown_small = upscale(self.crown, max(1, int(round(2 * k))))
        self.crown_big = upscale(self.crown, max(1, int(round(9 * k))))
        self.glow_gold = glow_sprite(int(560 * k), int(560 * k), GOLD, int(90 * k))
        self.glow_red = glow_sprite(int(640 * k), int(420 * k), RED, int(60 * k))

    # -------------------------------------------------------------------------------------------
    def _text(self, key, *a, **kw):
        if key not in self._cache:
            self._cache[key] = text_sprite(*a, **kw)
        return self._cache[key]

    def draw(self, frame, st):
        """frame: HxWx3 uint8. st: dict with
        board: list of 4 dicts (hp, vis 0..1, hurt 0..1, out 0..1, win 0..1) or None
        banner: (round_no, title, t) ; intro: (giant, t) ; elim: (giant, t) ; winner: (giant, t) ;
        flash: white overlay 0..1 ; seed: frame number (heart jiggle)."""
        out = frame.astype(np.float32)
        if st.get('board') is not None:
            self._board(out, st['board'], st.get('seed', 0))
        if st.get('intro') is not None:
            self._intro(out, *st['intro'])
        if st.get('banner') is not None:
            self._banner(out, *st['banner'])
        if st.get('elim') is not None:
            self._elim(out, *st['elim'])
        if st.get('winner') is not None:
            self._winner(out, *st['winner'])
        if st.get('flash', 0.0) > 0:
            out = out * (1 - st['flash']) + 255.0 * st['flash']
        return np.clip(out, 0, 255).astype(np.uint8)

    # -------------------------------------------------------------------------------------------
    def _board(self, out, rows, seed):
        k = self.k
        vis_any = max(r['vis'] for r in rows)
        if vis_any <= 0:
            return
        pp = self.panel_pad
        n_rows = sum(smooth(r['vis']) for r in rows)
        hgt = int(round(2 * self.pad + n_rows * self.row_h + max(0.0, n_rows - 1) * self.row_gap))
        hgt = max(hgt, 2 * self.pad + 8)
        if hgt not in self._panels:
            self._panels[hgt] = with_shadow(rounded_panel(self.panel_w, hgt, int(round(20 * k)), (12, 14, 22, 150),
                                                          (255, 255, 255, 26)), blur=10, offset=(0, 6),
                                            strength=0.35, pad=pp)
        over(out, self._panels[hgt], self.x0 - pp, self.y0 - pp, opacity=smooth(min(1.0, vis_any * 1.5)))
        rng = np.random.default_rng(seed)
        for i, r in enumerate(rows):
            if r['vis'] <= 0:
                continue
            e = smooth(r['vis'])
            dx = -40 * k * (1 - e)
            if r.get('hurt', 0) > 0:
                dx += np.sin(r['hurt'] * 40.0) * 4 * k * r['hurt']
            alpha = e * (1.0 - 0.45 * r.get('out', 0.0)) * (1.0 - 0.25 * r.get('dim', 0.0))
            y = self.y0 + self.pad + i * (self.row_h + self.row_gap)
            x = self.x0 + self.pad + dx
            port = self.portraits_x[i] if r.get('out', 0) > 0.5 else self.portraits[i]
            if r.get('win', 0) > 0:
                key = ('rowline',)
                if key not in self._cache:
                    self._cache[key] = rounded_panel(self.panel_w - int(12 * k), self.row_h + int(8 * k),
                                                     int(14 * k), (255, 200, 61, 34), (255, 200, 61, 235))
                ln = self._cache[key]
                over(out, ln, self.x0 + int(6 * k), y - int(4 * k), smooth(r['win'] * 1.5))
            if r.get('hurt', 0) > 0 and r.get('out', 0) < 0.5:
                tint = port.astype(np.float32)
                tint[..., :3] = tint[..., :3] * (1 - 0.55 * r['hurt']) + np.array([255, 40, 40]) * 0.55 * r['hurt']
                port = tint.astype(np.uint8)
            py = y + (self.row_h - port.shape[0]) // 2
            over(out, port, x, py, alpha)
            tx = x + port.shape[1] + 16 * k
            name = self.names_out[i] if r.get('out', 0) > 0.5 else self.names[i]
            over(out, name, tx, y + 2 * k, alpha)
            if r.get('win', 0) > 0:
                cs = self.crown_small
                over(out, cs, tx + name.shape[1] + 8 * k, y + 2 * k + (name.shape[0] - cs.shape[0]) // 2,
                     alpha * smooth(r['win'] * 2))
            if r.get('out', 0) > 0.5:
                # struck through in red
                ly = int(y + 2 * k + name.shape[0] * 0.52)
                x_a, x_b = int(tx), int(tx + name.shape[1] * min(1.0, (r['out'] - 0.5) * 3.0))
                if x_b > x_a:
                    out[ly:ly + max(2, int(3 * k)), x_a:x_b] = np.array(RED, np.float32)
            hp = int(r['hp'])
            flash = r.get('hurt', 0) > 0.3 and int(r['hurt'] * 10) % 2 == 0
            hy = y + self.row_h - self.heart_w - 4 * k
            for h in range(10):
                v = hp - 2 * h
                state = 'full' if v >= 2 else ('half' if v == 1 else 'empty')
                spr = self.hearts[(state, flash)]
                if r.get('out', 0) > 0.5:
                    spr = grey(spr, 1.0, 0.8)
                jy = 0
                if 0 < hp <= 4 and r.get('out', 0) < 0.5:
                    jy = int(rng.integers(-1, 2)) * self.hs
                over(out, spr, tx + h * (self.heart_w + self.heart_gap), hy + jy, alpha)

    # -------------------------------------------------------------------------------------------
    def _banner(self, out, round_no, title, t):
        """Round banner: small gold ROUND N between two rules, the round title below; ~1.5 s."""
        k = self.k
        if t < 0 or t > 1.55:
            return
        small = self._text(('round', round_no), f'ROUND {round_no}', self.f_round, fill=GOLD, stroke=int(4 * k),
                           tracking=int(round(8 * k)))
        big = self._text(('title', title), title, self.f_title, stroke=int(round(7 * k)), tracking=int(round(2 * k)))
        a = smooth(t / 0.12) * (1.0 - smooth((t - 1.3) / 0.25))
        s = ease_out_back(t / 0.28) * 0.35 + 0.65 if t < 0.28 else 1.0
        cy = int(0.365 * self.H) - int(14 * k * smooth((t - 1.3) / 0.25))
        b = scaled(big, s)
        over(out, b, (self.W - b.shape[1]) // 2, cy - b.shape[0] // 2 + int(38 * k), a)
        sm = scaled(small, 0.8 + 0.2 * smooth(t / 0.2))
        sy = cy - int(62 * k) - sm.shape[0] // 2
        over(out, sm, (self.W - sm.shape[1]) // 2, sy, a)
        # gold rules either side of ROUND N, drawn out from the centre
        rl = int(110 * k * smooth((t - 0.05) / 0.25))
        if rl > 2:
            ry = sy + sm.shape[0] // 2 - 2
            th = max(3, int(4 * k))
            for side in (-1, 1):
                if side < 0:
                    x_b = (self.W - sm.shape[1]) // 2 + int(6 * k)
                    x_a = x_b - rl
                else:
                    x_a = (self.W + sm.shape[1]) // 2 - int(6 * k)
                    x_b = x_a + rl
                seg = out[ry:ry + th, max(0, x_a):x_b]
                seg[:] = seg * (1 - a) + np.array(GOLD, np.float32) * a

    def _intro(self, out, gi, t):
        """Fighter introduction: the name slams in from the right in the lower third."""
        k = self.k
        label = SPECS[self.ORDER[gi]]['label']
        spr = self._text(('intro', gi), label, self.f_intro, stroke=int(round(8 * k)), tracking=int(round(3 * k)))
        e = smooth(t / 0.16)
        x = (self.W - spr.shape[1]) // 2 + int(420 * k * (1 - e) ** 2)
        s = 1.0 + 0.06 * (1 - smooth(t / 0.3))
        sp = scaled(spr, s)
        y = int(0.655 * self.H) - sp.shape[0] // 2
        over(out, sp, x - (sp.shape[1] - spr.shape[1]) // 2, y, min(1.0, t / 0.06))

    def _elim(self, out, gi, t):
        """Elimination stamp: portrait with a red X, the name, ELIMINATED slammed in; ~1.6 s."""
        k = self.k
        if t < 0 or t > 1.65:
            return
        a = min(1.0, t / 0.05) * (1.0 - smooth((t - 1.4) / 0.25))
        s = 1.75 - 0.8 * smooth(t / 0.16) if t < 0.16 else 0.95 + 0.05 * smooth((t - 0.16) / 0.14)
        cy = int(0.47 * self.H)
        add(out, self.glow_red, (self.W - self.glow_red.shape[1]) // 2, cy - self.glow_red.shape[0] // 2,
            gain=0.35 * a)
        port = self._cache.get(('elim_port', gi))
        if port is None:
            port = with_shadow(upscale(self.portraits_x[gi], 2), blur=8, offset=(0, 6), strength=0.5, pad=18)
            self._cache[('elim_port', gi)] = port
        p = scaled(port, s, Image.NEAREST)
        over(out, p, (self.W - p.shape[1]) // 2, cy - int(150 * k) - p.shape[0] // 2, a)
        name = self._text(('ename', gi), SPECS[self.ORDER[gi]]['label'], self.f_stamp_name, stroke=int(5 * k),
                          tracking=int(round(4 * k)))
        n = scaled(name, s)
        over(out, n, (self.W - n.shape[1]) // 2, cy + int(12 * k) - n.shape[0] // 2, a)
        stamp = self._text('stamp', 'ELIMINATED', self.f_stamp, fill=RED, stroke=int(round(7 * k)),
                           tracking=int(round(2 * k)), gradient=((255, 96, 80), (214, 24, 30)))
        st = rotated(scaled(stamp, s), -3.0)
        over(out, st, (self.W - st.shape[1]) // 2, cy + int(104 * k) - st.shape[0] // 2, a)

    def _winner(self, out, gi, t):
        """Winner card in the lower third (the winner himself is on screen): a crown drops onto WINNER, gold glow
        and sparkles; stays up."""
        k = self.k
        if t < 0:
            return
        a = min(1.0, t / 0.08)
        cy = int(0.655 * self.H)
        add(out, self.glow_gold, (self.W - self.glow_gold.shape[1]) // 2, cy - int(60 * k) - self.glow_gold.shape[0] // 2,
            gain=0.3 * a * (0.85 + 0.15 * np.sin(t * 5.0)))
        win = self._text('winner', 'WINNER', self.f_win, fill=GOLD, stroke=int(round(8 * k)),
                         tracking=int(round(4 * k)), gradient=((255, 236, 140), (240, 150, 20)))
        sw = ease_out_back(t / 0.3) * 0.45 + 0.55 if t < 0.3 else 1.0
        w_ = scaled(win, sw)
        over(out, w_, (self.W - w_.shape[1]) // 2, cy - w_.shape[0] // 2, a)
        # the crown drops onto the word
        tc = t - 0.2
        if tc > 0:
            cr = with_shadow_cached(self, self.crown_big)
            drop = (1.0 - smooth(tc / 0.28)) * 260 * k
            bounce = np.sin(min(1.0, max(0.0, (tc - 0.28) / 0.22)) * np.pi) * 12 * k
            cyy = cy - win.shape[0] // 2 - cr.shape[0] + int(40 * k) - drop - bounce
            over(out, cr, (self.W - cr.shape[1]) // 2, cyy, min(1.0, tc / 0.1))
        # twinkling sparkles around the card
        rng = np.random.default_rng(99)
        for i in range(14):
            ang = rng.uniform(0, 2 * np.pi)
            rad = rng.uniform(150, 250) * k
            ph = rng.uniform(0, 2 * np.pi)
            tw_ = np.sin(t * rng.uniform(4, 7) + ph)
            if tw_ <= 0.2 or t < 0.3:
                continue
            sx = self.W / 2 + np.cos(ang) * rad * 1.7
            sy = cy - 50 * k + np.sin(ang) * rad * 0.8
            size = int((6 + 8 * tw_) * k)
            sp = sparkle(size)
            add(out, sp, sx - sp.shape[1] / 2, sy - sp.shape[0] / 2, gain=0.9 * tw_ * a)


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
