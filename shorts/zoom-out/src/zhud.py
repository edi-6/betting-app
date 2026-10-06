"""What's drawn over the picture: the title, the counter (how wide the view is, in Steve's pixels, then blocks, then
the giant's pixels), how long it takes to walk across, and the milestones as the view widens: one pixel, one block,
a chunk (the game's chunk borders), a village, the render distance, the Earth to scale, where the Far Lands were,
the world border, and the punchline. The outlines are drawn on the world (before the motion blur), the text on top.
"""
import cv2
import numpy as np

import pixelfont as PF
import ztimeline as TL
from hud import over

WHITE = (255, 255, 255)
GOLD = ((255, 240, 120), (255, 170, 20))
YELLOW = (255, 226, 92)
TITLE_Y = 112
LABEL_Y = 404
COUNT_Y = 1196

# milestones: (fade in from, full at, full until, gone at) as view widths, lines of text
R_BLOCK = (3.4, 4.4, 6.0, 8.0)
R_CHUNK = (9.0, 15.0, 42.0, 72.0)
R_VILLAGE = (72.0, 110.0, 240.0, 340.0)
R_RENDER = (340.0, 520.0, 1300.0, 2300.0)
R_EARTH = (9.0e6, 1.35e7, 2.0e7, 2.4e7)
R_FAR = (2.4e7, 2.8e7, 3.4e7, 3.85e7)
R_BORDER = (3.85e7, 4.5e7, 5.5e7, 6.0e7)
R_PIXEL = (6.0e7, 6.6e7, 1.45e8, 1.95e8)
LABELS = [
    (R_BLOCK, ['16 PIXELS = 1 BLOCK']),
    (R_CHUNK, ['1 CHUNK', '16 x 16 BLOCKS']),
    (R_VILLAGE, ['A VILLAGE']),
    (R_RENDER, ['RENDER DISTANCE', 'ALL YOUR GAME LOADS']),
    (R_EARTH, ['EARTH', '(TO SCALE)']),
    (R_FAR, ['THE FAR LANDS', '(OLD JAVA EDITION)']),
    (R_BORDER, ['THE WORLD BORDER', '7x BIGGER THAN EARTH']),
    (R_PIXEL, ['THE WHOLE WORLD...', "IS ONE PIXEL", "OF STEVE'S EYE"]),
]
EARTH = (0.0, 7.2e6, 6.371e6)                 # where the Earth is drawn, to scale (centre x, y, radius in m)
EARTH_FADE = (9.0e6, 1.35e7, 2.1e7, 2.6e7)
FAR_LANDS = 12_550_821.0
RENDER_DIST = (-192.0, 208.0)                 # 12 chunks round the player's chunk


def border_alpha(L):
    """The world border's line: shown as it comes into view, gone by the loop's seam."""
    return TL.ramp(L, 3.0e7, 3.8e7, 1.1e8, 2.2e8)


def earth_alpha(L):
    return TL.ramp(L, *EARTH_FADE)


def cloud_alpha(L):
    return 0.8 * TL.ramp(L, 400.0, 900.0, 1500.0, 2600.0)


class Hud:
    def __init__(self, W=1080, H=1920):
        self.W, self.H = W, H
        self._c = {}
        self.title = [self.sprite('HOW BIG IS A', 8), self.sprite('MINECRAFT WORLD?', 8)]

    def sprite(self, text, px, color=WHITE, grad=None):
        key = (text, px, color, grad)
        if key not in self._c:
            if len(self._c) > 600:
                self._c.clear()
            self._c[key] = PF.render(text, px=px, color=color, grad=grad, outline=1)
        return self._c[key]

    def _centred(self, out, spr, y, opacity=1.0, scale=1.0, cx=None):
        if scale != 1.0:
            h, w = spr.shape[:2]
            spr = cv2.resize(spr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_NEAREST)
        cx = self.W / 2 if cx is None else cx
        over(out, spr, cx - spr.shape[1] / 2, y - spr.shape[0] / 2, opacity)
        return spr.shape[0]

    # -- on the world ------------------------------------------------------------------------------------------------
    def world(self, img, L, phase, proj):
        """Outlines on the world (img: float32 HxWx3, drawn in place). proj(x, y, z) -> screen x, y."""
        W, H = self.W, self.H
        out = img
        L3 = L / TL.RATIO if phase == 'eye' else L

        def lines(segs, col, width, alpha):
            if alpha <= 0.0:
                return
            layer = out.copy()
            for (a, b) in segs:
                cv2.line(layer, (int(round(a[0] * 4)), int(round(a[1] * 4))), (int(round(b[0] * 4)), int(round(b[1] * 4))),
                         col, int(width), cv2.LINE_AA, shift=2)
            out[:] = out * (1 - alpha) + layer * alpha

        def rect(x0, y0, x1, y1, z, col, width, alpha):
            p = [proj(x0, y0, z), proj(x1, y0, z), proj(x1, y1, z), proj(x0, y1, z)]
            lines([(p[k], p[(k + 1) % 4]) for k in range(4)], col, width, alpha)
            return p

        # one pixel: Steve's pupil
        if phase in ('3d', 'fade', 'eye'):
            a = TL.ramp(L3, 0.13, 0.22, 0.85, 1.4)
            if a > 0:
                h = TL.PX / 2
                from zscene import PUPIL_Z
                p = rect(-h, -h, h, h, PUPIL_Z, (255, 255, 255), max(2, W // 360), a)
                y = max(q[1] for q in p)
                self._centred(out, self.sprite('1 PIXEL', 5), y + 36, a)
        # one block of grass, by his head
        if phase in ('3d', 'fade'):
            a = TL.ramp(L, *R_BLOCK)
            if a > 0:
                p = rect(1.0, -1.0, 2.0, 0.0, 0.0, (255, 255, 255), max(3, W // 300), a)
                top = min(q[1] for q in p)
                self._centred(out, self.sprite('1 BLOCK', 5), top - 30, a, cx=(p[0][0] + p[1][0]) / 2)
        # chunk borders, the game's way: yellow lines every 16 blocks, the chunk with his house picked out
        a = TL.ramp(L, *R_CHUNK) if phase in ('3d', 'fade', 'map') else 0.0
        if a > 0:
            half = L * 1.2
            k0, k1 = int(np.floor(-half / 16)), int(np.ceil(half / 16))
            segs = []
            for k in range(int(np.floor(-half * 1.8 / 16)), int(np.ceil(half * 1.8 / 16)) + 1):
                if k0 <= k <= k1:
                    segs.append((proj(16 * k, -half * 1.8, 0.0), proj(16 * k, half * 1.8, 0.0)))
                segs.append((proj(-half, 16 * k, 0.0), proj(half, 16 * k, 0.0)))
            shadow = [((p[0] + 2, p[1] + 2), (q[0] + 2, q[1] + 2)) for (p, q) in segs]
            lines(shadow, (20, 20, 10), max(3, W // 360), 0.5 * a)
            lines(segs, (255, 232, 70), max(3, W // 360), 0.9 * a)
            # the chunk with his house in it
            c = [proj(0.0, -16.0, 0.0), proj(16.0, 0.0, 0.0)]
            x0, y0 = int(round(min(c[0][0], c[1][0]))), int(round(min(c[0][1], c[1][1])))
            x1, y1 = int(round(max(c[0][0], c[1][0]))), int(round(max(c[0][1], c[1][1])))
            sl = out[max(y0, 0):max(min(y1, H), 0), max(x0, 0):max(min(x1, W), 0)]
            sl[:] = sl * (1 - 0.22 * a) + np.array([255, 226, 60], np.float32) * 0.22 * a
            rect(0.0, -16.0, 16.0, 0.0, 0.0, (255, 232, 70), max(5, W // 200), a)
        # the village
        a = TL.ramp(L, *R_VILLAGE) if phase in ('3d', 'fade', 'map') else 0.0
        if a > 0:
            c = proj(-12.0, -88.0, 0.0)
            e = proj(-12.0 + 66.0, -88.0, 0.0)
            r = abs(e[0] - c[0])
            layer = out.copy()
            cv2.circle(layer, (int(c[0] * 4), int(c[1] * 4)), int(r * 4), (255, 236, 90), max(3, W // 300),
                       cv2.LINE_AA, shift=2)
            out[:] = out * (1 - a) + layer * a
        # the render distance: the world outside it dims, as if it wasn't loaded
        a = TL.ramp(L, *R_RENDER) if phase in ('map', 'fade', '3d') else 0.0
        if a > 0:
            r0, r1 = RENDER_DIST
            p0, p1 = proj(r0, r1, 0.0), proj(r1, r0, 0.0)
            x0, y0, x1, y1 = [int(round(v)) for v in (p0[0], p0[1], p1[0], p1[1])]
            dim = np.ones((H, W, 1), np.float32) * (1.0 - 0.42 * a)
            dim[max(y0, 0):max(min(y1, H), 0), max(x0, 0):max(min(x1, W), 0)] = 1.0
            out *= dim
            rect(r0, r0, r1, r1, 0.0, (255, 255, 255), max(3, W // 300), a)
        # where the Far Lands were
        a = TL.ramp(L, *R_FAR) if phase == 'map' else 0.0
        if a > 0:
            f = FAR_LANDS
            rect(-f, -f, f, f, 0.0, (255, 120, 80), max(3, W // 300), a)
        # the Earth's outline
        a = earth_alpha(L) if phase == 'map' else 0.0
        if a > 0:
            ex, ey, er = EARTH
            c = proj(ex, ey, 0.0)
            r = abs(proj(ex + er, ey, 0.0)[0] - c[0])
            layer = out.copy()
            cv2.circle(layer, (int(c[0] * 4), int(c[1] * 4)), int(r * 4), (255, 255, 255), max(2, W // 400),
                       cv2.LINE_AA, shift=2)
            out[:] = out * (1 - 0.6 * a) + layer * 0.6 * a
        return out

    # -- on the screen ----------------------------------------------------------------------------------------------
    def draw(self, img, L):
        out = img.astype(np.float32)
        y = TITLE_Y
        for spr in self.title:
            y += self._centred(out, spr, y + spr.shape[0] / 2) + 8
        # milestones
        for (rng, lines) in LABELS:
            a = TL.ramp(L, *rng)
            if a <= 0.004:
                continue
            pop = 0.86 + 0.14 * min(1.0, a * 1.6)
            y = LABEL_Y
            for k, txt in enumerate(lines):
                big = k == 0
                spr = self.sprite(txt, 7 if big else 5, grad=GOLD if big else None, color=WHITE)
                y += self._centred(out, spr, y + spr.shape[0] * pop / 2, a, pop) + 10
        # the counter
        num, unit = TL.counter(L)
        g = TL.ramp(L, 5.0e7, 5.9e7, 1.1e8, 1.9e8)
        pop = 1.0 + 0.12 * TL.ramp(L, 5.6e7, 6.0e7, 6.0e7, 7.0e7)
        spr = self.sprite(num, 13, grad=GOLD) if g > 0.5 else self.sprite(num, 13)
        h = self._centred(out, spr, COUNT_Y + spr.shape[0] / 2, 1.0, pop)
        su = self.sprite(unit, 6, color=YELLOW)
        self._centred(out, su, COUNT_Y + h + 22 + su.shape[0] / 2)
        a = TL.ramp(L, 1.3, 2.2, 4.5e7, 6.0e7)
        if a > 0:
            sw = self.sprite('WALK ACROSS: ' + TL.walk_text(L), 4, color=(225, 225, 225))
            self._centred(out, sw, COUNT_Y + h + 22 + su.shape[0] + 26 + sw.shape[0] / 2, a)
        return np.clip(out, 0, 255).astype(np.uint8)
