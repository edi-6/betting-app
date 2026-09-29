"""The on-screen counter, in the game's style: how many dominoes have fallen, in the blocky pixel font with the
game's drop shadow, over an XP-bar style progress bar and '/ 10,000 DOMINOES'. It pops and turns gold at 10,000, and
when Herobrine's eyes light up it turns red and glitches.
"""
import numpy as np

import pixelfont as PF

WHITE = (255, 255, 255)
GOLD_T, GOLD_B = (255, 240, 120), (255, 170, 20)
RED = (255, 60, 50)
XP = np.array([128, 255, 32], float)


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


def fmt(n):
    return f'{int(n):,}'


class HUD:
    def __init__(self, W=1080, H=1920, total=10000):
        self.W, self.H = W, H
        self.k = W / 1080.0
        self.total = total
        self._cache = {}
        self.label = PF.render(f'/ {fmt(total)} DOMINOES', px=max(1, int(round(4 * self.k))),
                               color=(225, 225, 225), outline=1)

    def _num(self, n, style):
        key = (n, style)
        if key not in self._cache:
            px = max(1, int(round(11 * self.k)))
            if style == 'gold':
                spr = PF.render(fmt(n), px=px, grad=(GOLD_T, GOLD_B), outline=1)
            elif style == 'red':
                spr = PF.render(fmt(n), px=px, color=RED, outline=1)
            else:
                spr = PF.render(fmt(n), px=px, color=WHITE, outline=1)
            if len(self._cache) > 400:
                self._cache.clear()
            self._cache[key] = spr
        return self._cache[key]

    def _bar(self, out, frac, y, style):
        k = self.k
        w, h = int(640 * k), int(22 * k)
        x = (self.W - w) // 2
        b = max(1, int(round(3 * k)))
        out[y - b:y + h + b, x - b:x + w + b] = out[y - b:y + h + b, x - b:x + w + b] * 0.25
        out[y:y + h, x:x + w] = out[y:y + h, x:x + w] * 0.35 + np.array([28, 28, 28]) * 0.65
        fw = int(round(w * np.clip(frac, 0, 1)))
        if fw > 0:
            col = XP if style == 'white' else (np.array([255, 196, 40]) if style == 'gold' else np.array([230, 40, 30]))
            out[y:y + h, x:x + fw] = col
            out[y:y + max(1, h // 4), x:x + fw] = np.minimum(col * 1.25 + 30, 255)
            out[y + h - max(1, h // 4):y + h, x:x + fw] = col * 0.7
        # segment ticks like the XP bar
        for s in range(1, 18):
            xs = x + int(w * s / 18)
            out[y:y + h, xs:xs + b] = out[y:y + h, xs:xs + b] * 0.3

    def draw(self, frame, n, t_done=None, glitch=0.0, t=0.0, alpha=1.0):
        """n: dominoes down; t_done: seconds since the counter reached the total (None before); glitch: 0..1."""
        out = frame.astype(np.float32)
        k = self.k
        style = 'white'
        if t_done is not None:
            style = 'gold'
        if glitch > 0.5:
            style = 'red'
        spr = self._num(int(n), style)
        s = 1.0
        if t_done is not None and t_done < 0.5:
            s = 1.0 + 0.25 * np.sin(np.pi * t_done / 0.5) * np.exp(-t_done * 3)
        if s != 1.0:
            from PIL import Image
            img = Image.fromarray(spr)
            spr = np.array(img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.NEAREST))
        y0 = int(150 * k)
        x = (self.W - spr.shape[1]) / 2
        yy = y0 + (self._num(0, 'white').shape[0] - spr.shape[0]) / 2
        if glitch > 0.0:
            # slices of the number jump sideways, the colour splits
            rng = np.random.default_rng(int(t * 30))
            tmp = np.zeros_like(out)
            over(tmp, spr, x, yy)
            for _ in range(int(2 + 6 * glitch)):
                ya = int(yy + rng.integers(0, spr.shape[0]))
                hh = int(rng.integers(4, 18) * k)
                dx = int(rng.integers(-40, 40) * glitch * k)
                tmp[ya:ya + hh] = np.roll(tmp[ya:ya + hh], dx, axis=1)
            mask = tmp.sum(-1, keepdims=True) > 0
            out = np.where(mask, out * (1 - alpha) + tmp * alpha, out)
        else:
            over(out, spr, x, yy, alpha)
        bar_y = int(y0 + self._num(0, 'white').shape[0] + 14 * k)
        self._bar(out, n / self.total, bar_y, style)
        over(out, self.label, (self.W - self.label.shape[1]) / 2, bar_y + int(34 * k), alpha)
        return np.clip(out, 0, 255).astype(np.uint8)
