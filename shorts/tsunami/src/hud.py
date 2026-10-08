"""The overlay, in the game's style (the blocky pixel font with its drop shadow, the heart icons): the four rounds
across the top (1 / 10 / 100 / 1,000, the one being played lit up, a tick or a cross on the ones done), the village's
hearts under them, the big round title when a round starts and the verdict when it's over. And the cold open's
title."""
import numpy as np
from PIL import Image

import pixelfont as PF
from textures import check_pixels, heart_pixels, upscale, x_pixels

LABELS = ['1', '10', '100', '1,000']
TITLES = ['1 BLOCK', '10 BLOCKS', '100 BLOCKS', '1,000 BLOCKS']
TITLE_COL = [((255, 255, 255), (220, 230, 240)), ((255, 250, 160), (255, 205, 60)), ((255, 200, 120), (255, 120, 30)),
             ((255, 120, 110), (215, 20, 20))]
GREEN = ((180, 255, 130), (60, 200, 50))
RED = ((255, 130, 120), (225, 30, 30))


def over(dst, src, x, y, opacity=1.0):
    if opacity <= 0.0:
        return
    h, w = src.shape[:2]
    Hh, Ww = dst.shape[:2]
    x, y = int(round(x)), int(round(y))
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(Ww, x + w), min(Hh, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    a = s[..., 3:4] / 255.0 * opacity
    dst[y0:y1, x0:x1] = dst[y0:y1, x0:x1] * (1 - a) + s[..., :3] * a


def _scaled(spr, s):
    if abs(s - 1.0) < 1e-3:
        return spr
    img = Image.fromarray(spr)
    return np.array(img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.NEAREST))


def _pop(age, dur=0.45):
    """Scale of something popping in: overshoots and settles."""
    if age < 0:
        return 0.0
    u = age / dur
    return 1.0 + 0.35 * np.exp(-u * 5.0) * np.cos(u * 9.0) - 0.35 * np.exp(-u * 30.0)


class HUD:
    def __init__(self, W=1080, H=1920):
        self.W, self.H = W, H
        self.k = k = W / 1080.0
        self._cache = {}
        self.check = upscale(check_pixels(), max(1, int(round(3 * k))))
        self.cross = upscale(x_pixels(), max(1, int(round(3 * k))))
        self.big_check = upscale(check_pixels(), max(1, int(round(9 * k))))
        self.big_cross = upscale(x_pixels(), max(1, int(round(9 * k))))
        hs = max(1, int(round(6 * k)))
        self.hearts = {(st, fl): upscale(heart_pixels(st, fl), hs) for st in ('full', 'half', 'empty')
                       for fl in (False, True)}
        self.heart_w = 9 * hs
        self.heart_gap = int(round(6 * k))

    def _text(self, text, px, color=(255, 255, 255), grad=None, outline=1):
        key = (text, px, color, grad, outline)
        if key not in self._cache:
            self._cache[key] = PF.render(text, px=max(1, int(round(px * self.k))), color=color, grad=grad,
                                         outline=outline)
        return self._cache[key]

    def _center(self, out, spr, y, a=1.0, x=None):
        cx = self.W / 2 if x is None else x
        over(out, spr, cx - spr.shape[1] / 2, y - spr.shape[0] / 2, a)

    # -- the rounds across the top ---------------------------------------------------------------------------------
    def tracker(self, out, current, results, t, a=1.0):
        """current: 0..3 (the round being played) or None; results: {round: True (survived) / False}."""
        k = self.k
        bw, bh, gap = int(186 * k), int(84 * k), int(20 * k)
        x0 = (self.W - (4 * bw + 3 * gap)) / 2
        y0 = int(232 * k)
        lab = self._text('BLOCK TSUNAMI', 4)
        self._center(out, lab, y0 - 26 * k, a)
        for i in range(4):
            x = x0 + i * (bw + gap)
            active = current == i
            done = i in results
            # a dark panel with a border, like the game's buttons
            ys, xs = int(y0), int(x)
            sub = out[ys:ys + bh, xs:xs + bw]
            sub *= 1.0 - 0.55 * a
            b = max(2, int(4 * k))
            col = np.array((255, 255, 255) if active else ((90, 200, 70) if done and results[i] else
                                                           ((220, 50, 40) if done else (110, 110, 110))), float)
            for sl in (np.s_[ys:ys + b, xs:xs + bw], np.s_[ys + bh - b:ys + bh, xs:xs + bw],
                       np.s_[ys:ys + bh, xs:xs + b], np.s_[ys:ys + bh, xs + bw - b:xs + bw]):
                out[sl] = out[sl] * (1 - a) + col * a
            if active:
                txt = self._text(LABELS[i], 7, grad=((255, 250, 170), (255, 200, 40)))
                txt = _scaled(txt, 1.0 + 0.05 * np.sin(t * 6.0))
            else:
                txt = self._text(LABELS[i], 7, color=(200, 200, 200) if not done else (255, 255, 255))
            over(out, txt, x + bw / 2 - txt.shape[1] / 2, y0 + bh / 2 - txt.shape[0] / 2, a)
            if done:
                ic = self.check if results[i] else self.cross
                over(out, ic, x + bw - ic.shape[1] * 0.7, y0 - ic.shape[0] * 0.35, a)

    # -- the village's hearts ----------------------------------------------------------------------------------------
    def hearts_row(self, out, half, flash=False, a=1.0, shake=0.0, t=0.0):
        """half: 0..20 half hearts left."""
        k = self.k
        n = 10
        w = n * self.heart_w + (n - 1) * self.heart_gap
        x0 = (self.W - w) / 2
        y0 = int(348 * k)
        lab = self._text('VILLAGE', 3, color=(235, 235, 235))
        over(out, lab, x0, y0 - lab.shape[0] - 8 * k, a)
        for i in range(n):
            v = half - 2 * i
            st = 'full' if v >= 2 else ('half' if v == 1 else 'empty')
            dy = 0.0
            if shake > 0:
                dy = shake * 7 * k * np.sin(t * 40.0 + i * 1.7)
            over(out, self.hearts[(st, flash)], x0 + i * (self.heart_w + self.heart_gap), y0 + dy, a)

    # -- big text ------------------------------------------------------------------------------------------------------
    def title(self, out, n, age, hold=1.15):
        """The round's title popping in the middle of the screen."""
        if age < 0 or age > hold + 0.3:
            return
        a = 1.0 - np.clip((age - hold) / 0.3, 0, 1)
        s = _pop(age)
        big = _scaled(self._text(TITLES[n], 15, grad=TITLE_COL[n], outline=1), s)
        sub = _scaled(self._text('TSUNAMI', 9, color=(255, 255, 255)), s)
        y = 760 * self.k
        self._center(out, big, y, a)
        self._center(out, sub, y + big.shape[0] / 2 + sub.shape[0] / 2 + 10 * self.k, a)

    def verdict(self, out, ok, age, text=None):
        if age < 0:
            return
        s = _pop(age, 0.5)
        ic = _scaled(self.big_check if ok else self.big_cross, s)
        y = 1120 * self.k
        self._center(out, ic, y)
        txt = text or ('SURVIVED' if ok else 'DESTROYED')
        t = _scaled(self._text(txt, 10, grad=GREEN if ok else RED), s)
        self._center(out, t, y + ic.shape[0] / 2 + t.shape[0] / 2 + 14 * self.k)

    def caption(self, out, text, y, px=8, grad=None, color=(255, 255, 255), a=1.0, pop=None):
        spr = self._text(text, px, color=color, grad=grad)
        if pop is not None:
            spr = _scaled(spr, _pop(pop))
        self._center(out, spr, y * self.k, a)

    def cold_open(self, out, age, a=1.0):
        s = _pop(age, 0.4)
        l1 = _scaled(self._text('1 vs 10 vs 100 vs 1,000', 7, color=(255, 255, 255)), s)
        l2 = _scaled(self._text('BLOCK TSUNAMI', 13, grad=((255, 236, 120), (255, 150, 30))), s)
        y = 300 * self.k
        self._center(out, l1, y, a)
        self._center(out, l2, y + l1.shape[0] / 2 + l2.shape[0] / 2 + 12 * self.k, a)
