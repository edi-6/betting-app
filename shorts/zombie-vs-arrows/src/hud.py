"""2D overlay: health bar, round label, X / check stamps. Composited onto rendered frames in numpy."""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from textures import heart_pixels, x_pixels, check_pixels, upscale

FONT_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets', 'fonts')


def _blur_alpha(rgba, radius):
    a = Image.fromarray(rgba[..., 3])
    return np.array(a.filter(ImageFilter.GaussianBlur(radius)))


def _over(dst, src, x, y, opacity=1.0):
    """Alpha-composite RGBA uint8/float sprite onto float RGB frame at integer position (x, y)."""
    h, w = src.shape[:2]
    H, W = dst.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    a = s[..., 3:4] / 255.0 * opacity
    dst[y0:y1, x0:x1] = dst[y0:y1, x0:x1] * (1 - a) + s[..., :3] * a


def _shadowed(sprite, blur=5, offset=(0, 4), strength=0.45, pad=16):
    """Return sprite with a soft drop shadow baked in (RGBA uint8, padded)."""
    h, w = sprite.shape[:2]
    canvas = np.zeros((h + 2 * pad, w + 2 * pad, 4), np.float32)
    sh = np.zeros_like(canvas)
    sh[pad + offset[1]:pad + offset[1] + h, pad + offset[0]:pad + offset[0] + w, 3] = sprite[..., 3]
    sa = _blur_alpha(sh.astype(np.uint8), blur).astype(np.float32) * strength
    canvas[..., 3] = sa
    # sprite over shadow
    sp = np.zeros_like(canvas)
    sp[pad:pad + h, pad:pad + w] = sprite
    a = sp[..., 3:4] / 255.0
    out_rgb = sp[..., :3] * a
    out_a = a[..., 0] + sa / 255.0 * (1 - a[..., 0])
    out = np.zeros_like(canvas)
    out[..., :3] = np.where(out_a[..., None] > 0, out_rgb / np.maximum(out_a[..., None], 1e-6), 0)
    out[..., 3] = out_a * 255
    return np.clip(out, 0, 255).astype(np.uint8), pad


class HUD:
    def __init__(self, W=1080, H=1920, heart_scale=10, gap=6, hearts_top=0.118):
        self.W, self.H = W, H
        self.hs = heart_scale
        self.gap = gap
        self.hearts_top = int(hearts_top * H)
        self.hearts = {}
        for st in ('full', 'half', 'empty'):
            for fl in (False, True):
                spr = upscale(heart_pixels(st, fl), heart_scale)
                self.hearts[(st, fl)] = _shadowed(spr, blur=4, offset=(0, 4), strength=0.32, pad=12)
        self.heart_w = 9 * heart_scale
        self.total_w = 10 * self.heart_w + 9 * gap
        self.font = ImageFont.truetype(os.path.join(FONT_DIR, 'Montserrat-800.ttf'), 94)
        self._labels = {}
        self.x_art = upscale(x_pixels(), 20)
        self.check_art = upscale(check_pixels(), 19)
        self.x_spr = _shadowed(self.x_art, blur=8, offset=(0, 8), strength=0.4, pad=24)
        self.check_spr = _shadowed(self.check_art, blur=8, offset=(0, 8), strength=0.4, pad=24)

    # -- label ---------------------------------------------------------------------------------
    def label_sprite(self, text):
        if text in self._labels:
            return self._labels[text]
        f = self.font
        bbox = f.getbbox(text, stroke_width=5)
        w, h = bbox[2] - bbox[0] + 40, bbox[3] - bbox[1] + 40
        img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.text((20 - bbox[0], 20 - bbox[1]), text, font=f, fill=(255, 255, 255, 255),
               stroke_width=5, stroke_fill=(18, 18, 22, 235))
        arr = np.array(img)
        spr = _shadowed(arr, blur=7, offset=(0, 5), strength=0.5, pad=18)
        self._labels[text] = spr
        return spr

    # -- drawing -------------------------------------------------------------------------------
    def draw(self, frame, st):
        """frame: HxWx3 uint8. st: dict(hp, flash, jiggle_seed, label, label_t, label_alpha, stamp, stamp_t)."""
        out = frame.astype(np.float32)
        self._draw_hearts(out, st)
        if st.get('label'):
            self._draw_label(out, st['label'], st.get('label_t', 1.0), st.get('label_alpha', 1.0))
        if st.get('stamp'):
            self._draw_stamp(out, st['stamp'], st.get('stamp_t', 1.0), st.get('stamp_alpha', 1.0))
        return np.clip(out, 0, 255).astype(np.uint8)

    def _draw_hearts(self, out, st):
        hp = int(st.get('hp', 20))
        flash = bool(st.get('flash', False))
        rng = np.random.default_rng(st.get('jiggle_seed', 0))
        x0 = (self.W - self.total_w) // 2
        for i in range(10):
            v = hp - 2 * i
            state = 'full' if v >= 2 else ('half' if v == 1 else 'empty')
            spr, pad = self.hearts[(state, flash)]
            dy = 0
            if hp <= 4 and st.get('jiggle', True):
                dy = int(rng.integers(-1, 2)) * (self.hs // 2)
            x = x0 + i * (self.heart_w + self.gap) - pad
            y = self.hearts_top + dy - pad
            _over(out, spr, x, y)

    def _draw_label(self, out, text, t, alpha):
        spr, pad = self.label_sprite(text)
        # pop: overshoot scale then settle
        if t < 1.0:
            s = _pop_curve(t)
        else:
            s = 1.0
        img = Image.fromarray(spr)
        if abs(s - 1.0) > 1e-3:
            img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.BICUBIC)
        arr = np.array(img)
        cy = int(0.225 * self.H)
        _over(out, arr, (self.W - arr.shape[1]) // 2, cy - arr.shape[0] // 2, opacity=alpha * min(1.0, t * 4 + 0.001))

    def _draw_stamp(self, out, kind, t, alpha):
        spr, pad = self.x_spr if kind == 'x' else self.check_spr
        s = _stamp_curve(t)
        img = Image.fromarray(spr)
        if abs(s - 1.0) > 1e-3:
            img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.BILINEAR)
        arr = np.array(img)
        cy = int(0.41 * self.H)
        _over(out, arr, (self.W - arr.shape[1]) // 2, cy - arr.shape[0] // 2, opacity=alpha * min(1.0, t * 6 + 0.001))


def _pop_curve(t):
    """0..1 -> scale: 0.55 -> 1.08 -> 1.0 (ease out back)."""
    t = np.clip(t, 0, 1)
    c1, c3 = 1.9, 2.9
    e = 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2
    return 0.55 + 0.45 * e


def _stamp_curve(t):
    """Stamp slams in: 1.65 -> 0.93 -> 1.0."""
    t = np.clip(t, 0, 1)
    if t < 0.45:
        u = t / 0.45
        return 1.65 - 0.72 * (u * u)
    u = (t - 0.45) / 0.55
    return 0.93 + 0.07 * (1 - (1 - u) ** 2)
