"""The screen over the picture, in the game's style: the title, the Y level (like the F3 screen), Steve's hearts,
the diamonds he's carrying, the captions that pop up as things happen, and the death screen ("You died!", the
score, the Respawn button)."""
import cv2
import numpy as np

import art as A
import pixelfont as PF

W, H = 1080, 1920
COLS = {'yellow': ((255, 240, 120), (255, 190, 30)), 'red': ((255, 120, 110), (230, 30, 30)),
        'green': ((170, 255, 140), (60, 200, 60)), 'white': ((255, 255, 255), (225, 225, 230)),
        'purple': ((235, 200, 255), (170, 110, 240)), 'cyan': ((200, 255, 255), (40, 210, 220))}
HEART = [".OO...OO.", "ORRO.ORRO", "ORWRORRRO", "ORRRRRRRO", "ORRRRRRDO", ".ORRRRDO.", "..ORRDO..", "...ODO...",
         "....O...."]


def over(dst, src, x, y, opacity=1.0):
    h, w = src.shape[:2]
    x, y = int(round(x)), int(round(y))
    xa, ya, xb, yb = max(0, x), max(0, y), min(dst.shape[1], x + w), min(dst.shape[0], y + h)
    if xb <= xa or yb <= ya or opacity <= 0:
        return
    s = src[ya - y:yb - y, xa - x:xb - x].astype(np.float32)
    a = s[..., 3:4] / 255.0 * opacity
    dst[ya:yb, xa:xb] = dst[ya:yb, xa:xb] * (1 - a) + s[..., :3] * a


def heart(state, flash=False):
    pal = {'O': (12, 6, 6), 'R': (236, 22, 22), 'W': (255, 214, 214), 'D': (170, 0, 0)}
    if flash:
        pal = {'O': (255, 255, 255), 'R': (255, 120, 120), 'W': (255, 240, 240), 'D': (236, 90, 90)}
    img = np.zeros((9, 9, 4), np.float32)
    for r, row in enumerate(HEART):
        for c, ch in enumerate(row):
            if ch == '.':
                continue
            filled = state == 'full' or (state == 'half' and c <= 4)
            col = pal['O'] if ch == 'O' else (pal[ch] if filled else (52, 46, 46))
            img[r, c, :3] = col
            img[r, c, 3] = 255
    return img


def up(img, k):
    return cv2.resize(img, (img.shape[1] * k, img.shape[0] * k), interpolation=cv2.INTER_NEAREST)


class Hud:
    def __init__(self):
        self._c = {}
        self.hearts = {(s, f): up(heart(s, f), 5) for s in ('full', 'half', 'empty') for f in (False, True)}
        self.diamond = up(A.sprite('diamond'), 5)

    def text(self, s, px, color=(255, 255, 255), grad=None, outline=1):
        k = (s, px, color, grad, outline)
        if k not in self._c:
            self._c[k] = PF.render(s, px=px, color=color, grad=grad, outline=outline).astype(np.float32)
        return self._c[k]

    def _centre(self, out, spr, cy, op=1.0, scale=1.0, cx=W / 2):
        if scale != 1.0:
            spr = cv2.resize(spr, (max(1, int(spr.shape[1] * scale)), max(1, int(spr.shape[0] * scale))),
                             interpolation=cv2.INTER_NEAREST)
        over(out, spr, cx - spr.shape[1] / 2, cy - spr.shape[0] / 2, op)
        return spr.shape

    def panel(self, out, x0, y0, x1, y1, a=0.45):
        out[y0:y1, x0:x1] *= 1 - a

    def draw(self, img, story, t):
        out = img.astype(np.float32)
        ev = story.events
        dead = t >= ev['dead']
        if not dead and t < 1.0:
            self.intro(out, story, t)
        if not dead:
            # title
            self._centre(out, self.text('I DUG STRAIGHT DOWN', 8), 118)
            self._centre(out, self.text('(X-RAY VIEW)', 5, color=(255, 230, 110)), 186)
            hud_a = min(1.0, max(0.0, (t - 0.9) / 0.3))
            if hud_a <= 0:
                return np.clip(out, 0, 255).astype(np.uint8)
            # Y level, like the F3 screen
            y = story.y_display(t)
            sp = self.text(f'Y: {y}', 7)
            self.panel(out, 28, 238, 28 + sp.shape[1] + 24, 238 + sp.shape[0] + 18)
            over(out, sp, 40, 247)
            # diamonds
            n = story.diamonds_at(t)
            ds = self.text(f'x{n}', 7, grad=COLS['cyan'] if n else None)
            x1 = W - 28
            x0 = x1 - (ds.shape[1] + self.diamond.shape[1] + 24)
            self.panel(out, x0, 238, x1, 238 + 84)
            pop = 1.0
            for (td, k) in story.dings:
                if td <= t < td + 0.25:
                    pop = 1.0 + 0.35 * (1 - (t - td) / 0.25)
            dd = cv2.resize(self.diamond, None, fx=pop, fy=pop, interpolation=cv2.INTER_NEAREST)
            over(out, dd, x0 + 6 + (self.diamond.shape[1] - dd.shape[1]) / 2, 238 + 2 + (80 - dd.shape[0]) / 2)
            over(out, ds, x0 + self.diamond.shape[1] + 14, 238 + (84 - ds.shape[0]) / 2)
            # hearts
            hp = story.hp_at(t)
            flash = story.hurt_at(t) > 0.3 and int(t * 16) % 2 == 0
            hw = 9 * 5 + 6
            x = W / 2 - hw * 5 + 3
            yh = 360
            for k in range(10):
                v = hp - 2 * k
                st = 'full' if v >= 2 else ('half' if v == 1 else 'empty')
                jig = int(np.sin(t * 40 + k) * 3) if hp <= 6 and hp > 0 else 0
                over(out, self.hearts[(st, flash)], x + k * hw, yh + jig)
            # captions
            for (t0, t1, s, style) in story.captions:
                if t0 <= t < t1:
                    u = t - t0
                    pop = 1.0 + 0.4 * np.exp(-u / 0.06) * (u < 0.3)
                    op = min(1.0, (t1 - t) / 0.12)
                    spr = self.text(s, 10 if len(s) <= 14 else 8, grad=COLS[style])
                    dx = int(np.sin(t * 50) * 5) if style == 'red' else 0
                    self._centre(out, spr, 1330, op, pop, cx=W / 2 + dx)
        else:
            out = self.death(out, story, t - ev['dead'])
        return np.clip(out, 0, 255).astype(np.uint8)

    def intro(self, out, story, t):
        """Markers on the opening's view of the whole slice: where he is, where the diamonds are."""
        import world as WD
        cx, cy, zoom, _ = story.camera(t)
        s = 96 * zoom
        op = min(1.0, (1.0 - t) / 0.25)
        to = lambda x, y: (W / 2 + (x - cx) * s, H / 2 - (y - cy) * s)
        for (x, y, label, col, side) in ((0.5, WD.SURFACE + 1.0, 'YOU ARE HERE', 'white', 1),
                                         (0.5, WD.DIAMONDS[0][1] + 0.5, 'DIAMONDS', 'cyan', 1)):
            px, py = to(x, y)
            spr = self.text(label, 6, grad=COLS[col])
            ax = px + 26 * side
            y0, y1 = int(py) - 3, int(py) + 3
            out[y0:y1, int(px) + 18:int(ax) + 60] = out[y0:y1, int(px) + 18:int(ax) + 60] * (1 - op) + 255 * op
            over(out, spr, ax + 70, py - spr.shape[0] / 2, op)
            layer = out.copy()
            cv2.circle(layer, (int(px), int(py)), 18, (255, 255, 255), 4)
            out[:] = out * (1 - op) + layer * op

    def button(self, out, cx, cy, label, hover=False, pressed=False):
        bw, bh = 640, 92
        x0, y0 = int(cx - bw / 2), int(cy - bh / 2)
        base = np.array([111, 111, 111]) if not hover else np.array([126, 136, 190])
        if pressed:
            base = np.array([90, 100, 160])
        out[y0:y0 + bh, x0:x0 + bw] = base
        out[y0:y0 + 5, x0:x0 + bw] = base * 1.35
        out[y0:y0 + bh, x0:x0 + 5] = base * 1.25
        out[y0 + bh - 8:y0 + bh, x0:x0 + bw] = base * 0.55
        out[y0:y0 + bh, x0 + bw - 5:x0 + bw] = base * 0.6
        out[y0 - 4:y0, x0:x0 + bw] = 0
        out[y0 + bh:y0 + bh + 4, x0:x0 + bw] = 0
        out[y0:y0 + bh, x0 - 4:x0] = 0
        out[y0:y0 + bh, x0 + bw:x0 + bw + 4] = 0
        col = (255, 255, 160) if hover else (230, 230, 230)
        self._centre(out, self.text(label, 6, color=col), cy - 2)

    def death(self, out, story, u):
        """The game's death screen; Respawn is hovered, then clicked."""
        k = min(1.0, u / 0.35)
        yy = np.linspace(0, 1, H)[:, None, None]
        red = np.array([120, 10, 10]) * (1 - 0.4 * yy) + np.array([60, 0, 0]) * 0.4 * yy
        out = out * (1 - 0.72 * k) + red * 0.72 * k
        op = min(1.0, max(0.0, (u - 0.15) / 0.3))
        self._centre(out, self.text('You died!', 14), 560, op)
        n = 5
        sc = self.text(f'Score: {n}', 7)
        self._centre(out, sc, 700, op)
        op2 = min(1.0, max(0.0, (u - 0.6) / 0.3))
        self._centre(out, self.text('NEVER DIG STRAIGHT DOWN', 7, grad=COLS['yellow']), 850, op2)
        if u > 0.9:
            hover = u > 1.6
            pressed = u > 2.35
            self.button(out, W / 2, 1010, 'Respawn', hover=hover, pressed=pressed)
            self.button(out, W / 2, 1130, 'Title Screen')
            if hover:
                # the cursor
                cx, cy = int(W / 2 + 150 - 120 * min(1.0, (u - 1.6) / 0.4)), int(1030 - 20 * min(1, (u - 1.6) / 0.4))
                cur = np.zeros((14, 10), bool)
                for r in range(14):
                    cur[r, :max(1, min(10, r + 1 - max(0, r - 9) * 2))] = True
                cur = np.repeat(np.repeat(cur, 4, 0), 4, 1)
                out[cy:cy + cur.shape[0], cx:cx + cur.shape[1]][cur] = 255
        return out
