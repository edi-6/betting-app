"""The 50 flags, drawn in code from their official construction sheets (proportions, positions, colours), then
rasterised with 4x4 supersampling. Every flag is drawn in flag units: y runs from 0 (top) to 1 (bottom) and x from
0 (hoist) to the flag's aspect ratio. Emblems too fiddly to reproduce exactly (Mexico's eagle, Spain's and
Portugal's arms, Egypt's eagle) are simplified but keep their layout and colours.

python flags.py OUT.png      # a review sheet: every flag next to its emoji
"""
import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, '..', 'cache', 'flags')
FONT_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def hexc(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# ----------------------------------------------------------------------------------------------------------------
# geometry

def circle_pts(cx, cy, r, n=None, a0=0.0, a1=2 * math.pi):
    n = n or 360
    a = np.linspace(a0, a1, n, endpoint=(a1 - a0) < 2 * math.pi - 1e-9)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)


def ellipse_pts(cx, cy, rx, ry, rot=0.0, n=360):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    x, y = rx * np.cos(a), ry * np.sin(a)
    c, s = math.cos(rot), math.sin(rot)
    return np.stack([cx + c * x - s * y, cy + s * x + c * y], 1)


def star_pts(cx, cy, r, n=5, ratio=None, rot=-math.pi / 2):
    """A regular star with n points, the first pointing at angle rot (y down, so -pi/2 points up)."""
    if ratio is None:
        ratio = math.cos(2 * math.pi / n) / math.cos(math.pi / n) if n == 5 else 0.5
    a = rot + np.arange(2 * n) * math.pi / n
    rr = np.where(np.arange(2 * n) % 2 == 0, r, r * ratio)
    return np.stack([cx + rr * np.cos(a), cy + rr * np.sin(a)], 1)


def band_pts(p0, p1, half):
    """A straight band of half-width `half` along p0 -> p1."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    n = np.array([-d[1], d[0]]) * half
    return np.array([p0 + n, p1 + n, p1 - n, p0 - n])


def offset_poly(pts, dist):
    """Moves every edge of a polygon outward by dist (inward if negative) and re-intersects neighbouring edges
    (mitred corners). Works for any simple polygon without degenerate edges."""
    p = np.asarray(pts, float)
    area = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    sgn = 1.0 if area > 0 else -1.0     # y is down: positive area means clockwise on screen
    n = len(p)
    lines = []
    for i in range(n):
        a, b = p[i], p[(i + 1) % n]
        d = (b - a) / np.linalg.norm(b - a)
        nrm = np.array([d[1], -d[0]]) * sgn       # outward normal
        lines.append((a + nrm * dist, d))
    out = []
    for i in range(n):
        (a0, d0), (a1, d1) = lines[i - 1], lines[i]
        m = np.array([d0, -d1]).T
        t = np.linalg.solve(m, a1 - a0)
        out.append(a0 + d0 * t[0])
    return np.array(out)


def arc_band(cx, cy, r0, r1, a0, a1, n=200):
    a = np.linspace(a0, a1, n)
    outer = np.stack([cx + r1 * np.cos(a), cy + r1 * np.sin(a)], 1)
    inner = np.stack([cx + r0 * np.cos(a[::-1]), cy + r0 * np.sin(a[::-1])], 1)
    return np.concatenate([outer, inner])


def svg_arc(p0, p1, rx, ry, phi, large, sweep, n=24):
    """Points along an SVG elliptical arc from p0 to p1 (endpoint parameterisation, per the SVG spec)."""
    x1, y1 = p0
    x2, y2 = p1
    c, s = math.cos(phi), math.sin(phi)
    dx, dy = (x1 - x2) / 2, (y1 - y2) / 2
    x1p, y1p = c * dx + s * dy, -s * dx + c * dy
    lam = x1p ** 2 / rx ** 2 + y1p ** 2 / ry ** 2
    if lam > 1:
        rx, ry = rx * math.sqrt(lam), ry * math.sqrt(lam)
    num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    co = math.sqrt(max(0.0, num / den)) * (-1 if large == sweep else 1)
    cxp, cyp = co * rx * y1p / ry, -co * ry * x1p / rx
    cx, cy = c * cxp - s * cyp + (x1 + x2) / 2, s * cxp + c * cyp + (y1 + y2) / 2

    def ang(ux, uy, vx, vy):
        a = math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)
        return a

    t1 = ang(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dt = ang((x1p - cxp) / rx, (y1p - cyp) / ry, (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if not sweep and dt > 0:
        dt -= 2 * math.pi
    elif sweep and dt < 0:
        dt += 2 * math.pi
    t = t1 + dt * np.linspace(0, 1, n)[1:]
    return np.stack([cx + rx * np.cos(t) * c - ry * np.sin(t) * s, cy + rx * np.cos(t) * s + ry * np.sin(t) * c], 1)


# ----------------------------------------------------------------------------------------------------------------
# canvas

class Canvas:
    """A supersampled RGB image addressed in flag units (W x H units, `ppu` final pixels per unit)."""

    def __init__(self, W, H, ppu, ss=4, bg=WHITE):
        self.W, self.H, self.ss = W, H, ss
        self.ppu = ppu * ss
        self.wpx = int(round(W * ppu)) * ss
        self.hpx = int(round(H * ppu)) * ss
        self.img = np.empty((self.hpx, self.wpx, 3), np.uint8)
        self.img[:] = bg

    def px(self, pts):
        return np.round(np.asarray(pts, np.float64) * self.ppu * 16).astype(np.int32)

    def fill(self, polys, color, dst=None):
        """Fills one polygon or a list of them (even-odd rule, so a list can cut holes)."""
        if isinstance(polys, np.ndarray) and polys.ndim == 2:
            polys = [polys]
        dst = self.img if dst is None else dst
        cv2.fillPoly(dst, [self.px(p) for p in polys], color, lineType=cv2.LINE_8, shift=4)

    def mask(self):
        return np.zeros((self.hpx, self.wpx), np.uint8)

    def paint(self, mask, color):
        self.img[mask > 0] = color

    def rect(self, x0, y0, x1, y1, color, dst=None):
        self.fill(np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], float), color, dst)

    def circle(self, cx, cy, r, color, dst=None):
        self.fill(circle_pts(cx, cy, r, n=max(90, int(r * self.ppu))), color, dst)

    def ring(self, cx, cy, r0, r1, color, dst=None):
        n = max(90, int(r1 * self.ppu))
        self.fill([circle_pts(cx, cy, r1, n), circle_pts(cx, cy, r0, n)], color, dst)

    def ellipse(self, cx, cy, rx, ry, color, rot=0.0, dst=None):
        self.fill(ellipse_pts(cx, cy, rx, ry, rot, n=max(90, int(max(rx, ry) * self.ppu))), color, dst)

    def ellipse_ring(self, cx, cy, rx, ry, w, color, rot=0.0, dst=None):
        n = max(90, int(max(rx, ry) * self.ppu))
        self.fill([ellipse_pts(cx, cy, rx + w / 2, ry + w / 2, rot, n),
                   ellipse_pts(cx, cy, rx - w / 2, ry - w / 2, rot, n)], color, dst)

    def star(self, cx, cy, r, color, n=5, ratio=None, rot=-math.pi / 2, dst=None):
        self.fill(star_pts(cx, cy, r, n, ratio, rot), color, dst)

    def line(self, p0, p1, w, color, dst=None):
        self.fill(band_pts(p0, p1, w / 2), color, dst)

    def polyline(self, pts, w, color, round_joints=True):
        pts = np.asarray(pts, float)
        for a, b in zip(pts[:-1], pts[1:]):
            self.line(a, b, w, color)
            if round_joints:
                self.circle(b[0], b[1], w / 2, color)
        if round_joints:
            self.circle(pts[0][0], pts[0][1], w / 2, color)

    def crescent(self, c_out, r_out, c_in, r_in, color):
        m = self.mask()
        self.circle(c_out[0], c_out[1], r_out, 255, dst=m)
        self.circle(c_in[0], c_in[1], r_in, 0, dst=m)
        self.paint(m, color)

    def hstripes(self, colors, weights=None):
        weights = weights or [1] * len(colors)
        tot, y = float(sum(weights)), 0.0
        for col, w in zip(colors, weights):
            self.rect(0, y, self.W, y + self.H * w / tot + 1e-3, col)
            y += self.H * w / tot

    def vstripes(self, colors, weights=None):
        weights = weights or [1] * len(colors)
        tot, x = float(sum(weights)), 0.0
        for col, w in zip(colors, weights):
            self.rect(x, 0, x + self.W * w / tot + 1e-3, self.H, col)
            x += self.W * w / tot

    def text_arc(self, text, cx, cy, radius, a_mid, size, color, track=0.08):
        """Upright letters (tops pointing away from the centre) set along an arc of `radius` centred on a_mid."""
        font = ImageFont.truetype(FONT_BOLD, max(8, int(size * self.ppu)))
        widths = [font.getlength(ch) / self.ppu for ch in text]
        gap = track * size
        total = sum(widths) + gap * (len(text) - 1)
        a = a_mid - (total / 2) / radius
        for ch, w in zip(text, widths):
            am = a + (w / 2) / radius
            a += (w + gap) / radius
            if ch == ' ':
                continue
            box = font.getbbox('H')
            gb = font.getbbox(ch)
            im = Image.new('L', (gb[2] - gb[0] + 8, box[3] - box[1] + 8), 0)
            ImageDraw.Draw(im).text((4 - gb[0], 4 - box[1]), ch, font=font, fill=255)
            # rotate so the glyph's up points along the radius (outward)
            deg = -math.degrees(am + math.pi / 2)
            im = im.rotate(deg, resample=Image.BICUBIC, expand=True)
            g = np.asarray(im)
            px = (cx + radius * math.cos(am)) * self.ppu
            py = (cy + radius * math.sin(am)) * self.ppu
            x0, y0 = int(px - g.shape[1] / 2), int(py - g.shape[0] / 2)
            sub = self.img[y0:y0 + g.shape[0], x0:x0 + g.shape[1]]
            al = (g[:sub.shape[0], :sub.shape[1]] / 255.0)[..., None]
            sub[:] = (sub * (1 - al) + np.array(color) * al).astype(np.uint8)

    def finish(self):
        h, w = self.hpx // self.ss, self.wpx // self.ss
        return cv2.resize(self.img, (w, h), interpolation=cv2.INTER_AREA)

    def sub(self, x0, y0, W, H):
        """A child canvas covering the rectangle (x0, y0)-(x0 + W', y0 + H') of this one, in its own units W x H
        (call paste() when done)."""
        return _Sub(self, x0, y0, W, H)


class _Sub(Canvas):
    def __init__(self, parent, x0, y0, W, H):
        self.parent = parent
        self.W, self.H, self.ss = W, H, parent.ss
        self.px0 = int(round(x0 * parent.ppu))
        self.py0 = int(round(y0 * parent.ppu))
        self.ppu = None
        self.img = None

    def area(self, wu, hu):
        """Set the size (in parent units) this child covers."""
        p = self.parent
        self.wpx = int(round(wu * p.ppu))
        self.hpx = int(round(hu * p.ppu))
        self.ppu = self.hpx / self.H
        self.img = p.img[self.py0:self.py0 + self.hpx, self.px0:self.px0 + self.wpx].copy()
        return self

    def paste(self):
        self.parent.img[self.py0:self.py0 + self.hpx, self.px0:self.px0 + self.wpx] = self.img


# ----------------------------------------------------------------------------------------------------------------
# shared elements

def union_jack(c):
    """The Union Jack in its own 60 x 30 units: St Andrew's white saltire, St Patrick's red saltire counterchanged
    (broad white uppermost at the hoist, the arms turning like a pinwheel), St George's cross fimbriated in white."""
    blue, red = hexc('#012169'), hexc('#C8102E')
    c.rect(0, 0, 60, 30, blue)
    for p0, p1 in (((0, 0), (60, 30)), ((0, 30), (60, 0))):
        p0, p1 = np.array(p0, float), np.array(p1, float)
        d = (p1 - p0) / np.linalg.norm(p1 - p0)
        c.fill(band_pts(p0 - d * 10, p1 + d * 10, 3), WHITE)
    ctr = np.array([30.0, 15.0])
    for corner in ((0, 0), (60, 0), (60, 30), (0, 30)):
        corner = np.array(corner, float)
        u = (corner - ctr) / np.linalg.norm(corner - ctr)
        side = np.array([u[1], -u[0]])     # top-left arm: red below the diagonal; the others follow by rotation
        a = corner + u * 5
        c.fill(np.array([a, ctr, ctr + side * 2, a + side * 2]), red)
    c.rect(25, 0, 35, 30, WHITE)
    c.rect(0, 10, 60, 20, WHITE)
    c.rect(27, 0, 33, 30, red)
    c.rect(0, 12, 60, 18, red)


# ----------------------------------------------------------------------------------------------------------------
# the flags (alphabetical by code within regions does not matter; COUNTRIES below sets the order)

def f_india(c):
    c.hstripes([hexc('#FF671F'), WHITE, hexc('#046A38')])
    navy = hexc('#06038D')
    cx, cy, r = c.W / 2, 0.5, 0.125 * 0.92
    c.ring(cx, cy, r * 0.88, r, navy)
    c.circle(cx, cy, r * 0.17, navy)
    for k in range(24):
        a = 2 * math.pi * k / 24
        tip = (cx + r * 0.9 * math.cos(a), cy + r * 0.9 * math.sin(a))
        base = np.array([cx + r * 0.15 * math.cos(a), cy + r * 0.15 * math.sin(a)])
        mid = r * 0.45
        n = np.array([-math.sin(a), math.cos(a)])
        m = np.array([cx + mid * math.cos(a), cy + mid * math.sin(a)])
        c.fill(np.array([base, m + n * r * 0.045, tip, m - n * r * 0.045]), navy)
        a2 = a + math.pi / 24
        c.circle(cx + r * 0.88 * math.cos(a2), cy + r * 0.88 * math.sin(a2), r * 0.035, navy)


def f_indonesia(c):
    c.hstripes([hexc('#E70011'), WHITE])


def f_pakistan(c):
    green = hexc('#01411C')
    c.rect(0, 0, c.W, 1, WHITE)
    c.rect(0.375, 0, c.W, 1, green)
    A = np.array([0.375 + 0.5625, 0.5])
    d = np.array([1.5, 0.0]) - np.array([0.375, 1.0])
    d /= np.linalg.norm(d)
    c.crescent(A, 0.3, A + d * 0.0625, 0.275, WHITE)
    s = A + d * (math.hypot(0.5625, 0.5) - 13 / 24)
    c.star(s[0], s[1], 0.1, WHITE, rot=math.atan2(d[1], d[0]))


def f_bangladesh(c):
    c.rect(0, 0, c.W, 1, hexc('#006A4E'))
    c.circle(0.75, 0.5, 1 / 3, hexc('#F42A41'))


def f_philippines(c):
    blue, red, gold = hexc('#0038A8'), hexc('#CE1126'), hexc('#FCD116')
    c.hstripes([blue, red])
    h = math.sqrt(3) / 2
    c.fill(np.array([(0, 0), (h, 0.5), (0, 1)]), WHITE)
    cx, cy = h / 3, 0.5
    c.circle(cx, cy, 0.09, gold)
    for k in range(8):
        a = -math.pi / 2 + k * math.pi / 4
        for off, ln, wd in ((0.0, 0.205, 0.022), (-0.13, 0.17, 0.014), (0.13, 0.17, 0.014)):
            aa = a + off
            d = np.array([math.cos(aa), math.sin(aa)])
            n = np.array([-d[1], d[0]])
            b = np.array([cx, cy]) + d * 0.085
            c.fill(np.array([b + n * wd, np.array([cx, cy]) + d * ln, b - n * wd]), gold)
    for p in ((0.085, 0.09), (0.085, 0.91), (h - 0.105, 0.5)):
        rot = math.atan2(p[1] - cy, p[0] - cx)
        c.star(p[0], p[1], 0.05, gold, rot=rot)


def f_vietnam(c):
    c.rect(0, 0, c.W, 1, hexc('#DA251D'))
    c.star(c.W / 2, 0.5, 0.3, hexc('#FFFF00'))


def f_japan(c):
    c.rect(0, 0, c.W, 1, WHITE)
    c.circle(c.W / 2, 0.5, 0.3, hexc('#BC002D'))


def f_korea(c):
    red, blue = hexc('#CD2E3A'), hexc('#0047A0')
    c.rect(0, 0, c.W, 1, WHITE)
    cx, cy, r = c.W / 2, 0.5, 0.25
    a = np.array([3.0, 2.0]) / math.sqrt(13)          # axis: upper-left to lower-right
    n = np.array([a[1], -a[0]])                      # towards the upper right
    C = np.array([cx, cy])
    m = c.mask()
    c.circle(cx, cy, r, 255, dst=m)
    top = c.mask()
    big = 3.0
    c.fill(np.array([C - a * big, C + a * big, C + a * big + n * big, C - a * big + n * big]), 255, dst=top)
    m_red = (m > 0) & (top > 0)
    s1, s2 = c.mask(), c.mask()
    p1, p2 = C - a * r / 2, C + a * r / 2
    c.circle(p1[0], p1[1], r / 2, 255, dst=s1)
    c.circle(p2[0], p2[1], r / 2, 255, dst=s2)
    m_red = (m_red | (s1 > 0)) & ~(s2 > 0)
    img = c.img
    img[(m > 0)] = blue
    img[m_red] = red
    # trigrams: bars 1/2 r*2 long... in H=1 units: length 0.25, thickness 1/24, gap 1/48
    L, T, G = 0.25, 1 / 24, 1 / 48
    d_in = r + 0.125

    def trigram(direction, pattern):
        u = direction / np.linalg.norm(direction)
        v = np.array([-u[1], u[0]])
        for i, broken in enumerate(pattern):
            r0 = d_in + i * (T + G)
            p = C + u * (r0 + T / 2)
            if not broken:
                c.fill(band_pts(p - v * L / 2, p + v * L / 2, T / 2), BLACK)
            else:
                c.fill(band_pts(p - v * L / 2, p - v * G / 2, T / 2), BLACK)
                c.fill(band_pts(p + v * G / 2, p + v * L / 2, T / 2), BLACK)

    trigram(np.array([-3.0, -2.0]), (0, 0, 0))      # geon, upper left
    trigram(np.array([3.0, 2.0]), (1, 1, 1))        # gon, lower right
    trigram(np.array([3.0, -2.0]), (1, 0, 1))       # gam, upper right
    trigram(np.array([-3.0, 2.0]), (0, 1, 0))       # ri, lower left


def f_china(c):
    c.rect(0, 0, c.W, 1, hexc('#EE1C25'))
    y = hexc('#FFFF00')
    u = 1 / 20
    c.star(5 * u, 5 * u, 3 * u, y)
    for sx, sy in ((10, 2), (12, 4), (12, 7), (10, 9)):
        rot = math.atan2(5 - sy, 5 - sx)
        c.star(sx * u, sy * u, u, y, rot=rot)


def f_thailand(c):
    c.hstripes([hexc('#A51931'), hexc('#F4F5F8'), hexc('#2D2A4A'), hexc('#F4F5F8'), hexc('#A51931')],
               [1, 1, 2, 1, 1])


def f_malaysia(c):
    red, blue, yel = hexc('#CC0001'), hexc('#010066'), hexc('#FFCC00')
    for i in range(14):
        c.rect(0, i / 14, c.W, (i + 1) / 14 + 1e-3, red if i % 2 == 0 else WHITE)
    ch = 8 / 14
    c.rect(0, 0, 1.0, ch, blue)
    cy = ch / 2
    c.crescent((0.355, cy), 0.235, (0.42, cy), 0.205, yel)
    c.star(0.69, cy, 0.215, yel, n=14, ratio=0.48)


def f_turkey(c):
    c.rect(0, 0, c.W, 1, hexc('#E30A17'))
    c.crescent((0.5, 0.5), 0.25, (0.5625, 0.5), 0.2, WHITE)
    c.star(0.5625 - 0.2 + 1 / 3 + 0.125, 0.5, 0.125, WHITE, rot=math.pi)


def f_uzbekistan(c):
    blue, green, red = hexc('#0099B5'), hexc('#1EB53A'), hexc('#CE1126')
    c.hstripes([blue, red, WHITE, red, green], [10, 0.6, 9.4 - 0.6, 0.6, 10])
    c.crescent((0.235, 0.165), 0.12, (0.275, 0.165), 0.105, WHITE)
    rows = ((0.075, 3), (0.165, 4), (0.255, 5))
    for y, k in rows:
        for j in range(k):
            x = 0.81 - j * 0.09
            c.star(x, y, 0.032, WHITE)


def f_uae(c):
    c.hstripes([hexc('#00732F'), WHITE, BLACK])
    c.rect(0, 0, 0.5, 1, hexc('#FF0000'))


def eagle_saladin(c, cx, cy, s):
    """A simplified Eagle of Saladin (Egypt), height s, facing the hoist."""
    gold, dark = hexc('#C09300'), hexc('#8a6a00')
    P = lambda pts: np.array([(cx + x * s, cy + y * s) for x, y in pts])
    for sgn in (-1, 1):
        wing = [(0.05, -0.22), (0.22, -0.3), (0.36, -0.22), (0.42, -0.02), (0.38, 0.16), (0.3, 0.3),
                (0.22, 0.24), (0.2, 0.33), (0.12, 0.26), (0.1, 0.33), (0.05, 0.22)]
        c.fill(P([(sgn * x, y) for x, y in wing]), gold)
        for k in range(4):
            yy = -0.12 + k * 0.1
            c.line(P([(sgn * 0.12, yy)])[0], P([(sgn * 0.36, yy + 0.06)])[0], 0.012 * s, dark)
    c.fill(P([(-0.1, -0.28), (0.1, -0.28), (0.12, 0.3), (-0.12, 0.3)]), gold)
    c.circle(cx - 0.03 * s, cy - 0.36 * s, 0.075 * s, gold)
    c.fill(P([(-0.09, -0.38), (-0.19, -0.35), (-0.09, -0.32)]), gold)
    c.circle(cx - 0.055 * s, cy - 0.38 * s, 0.012 * s, dark)
    sh = P([(-0.085, -0.17), (0.085, -0.17), (0.085, 0.08), (0.0, 0.18), (-0.085, 0.08)])
    c.fill(sh, dark)
    inner = offset_poly(sh, -0.012 * s)
    m = c.mask()
    c.fill(inner, 255, dst=m)
    xs = [inner[:, 0].min(), inner[:, 0].max()]
    third = (xs[1] - xs[0]) / 3
    for k, col in enumerate((hexc('#CE1126'), WHITE, BLACK)):
        mm = c.mask()
        c.rect(xs[0] + k * third, cy - s, xs[0] + (k + 1) * third + 1e-3, cy + s, 255, dst=mm)
        c.img[(m > 0) & (mm > 0)] = col
    c.fill(P([(-0.2, 0.36), (0.2, 0.36), (0.24, 0.44), (-0.24, 0.44)]), gold)
    c.fill(P([(-0.24, 0.44), (-0.3, 0.4), (-0.24, 0.36)]), gold)
    c.fill(P([(0.24, 0.44), (0.3, 0.4), (0.24, 0.36)]), gold)


def f_egypt(c):
    c.hstripes([hexc('#CE1126'), WHITE, BLACK])
    eagle_saladin(c, c.W / 2, 0.5, 0.3)


def stroked_pentagram(c, cx, cy, r, w, color, rot=-math.pi / 2):
    """A pentagram drawn as five lines of width w (Morocco, Ethiopia)."""
    outer = star_pts(cx, cy, r, 5, rot=rot)
    m = c.mask()
    c.fill(offset_poly(outer, w / 2), 255, dst=m)
    holes = []
    pent = outer[1::2]
    holes.append(offset_poly(pent, -w / 2))
    for i in range(5):
        tri = np.array([outer[2 * i], outer[(2 * i + 1) % 10], outer[(2 * i - 1) % 10]])
        holes.append(offset_poly(tri, -w / 2))
    for h in holes:
        c.fill(h, 0, dst=m)
    c.paint(m, color)


def f_morocco(c):
    c.rect(0, 0, c.W, 1, hexc('#C1272D'))
    stroked_pentagram(c, c.W / 2, 0.52, 0.22, 0.035, hexc('#006233'))


def f_algeria(c):
    c.vstripes([hexc('#006233'), WHITE])
    red = hexc('#D21034')
    c.crescent((0.75, 0.5), 0.25, (0.821, 0.5), 0.216, red)
    c.star(0.845, 0.5, 0.1, red, rot=0.0)


def f_nigeria(c):
    c.vstripes([hexc('#008751'), WHITE, hexc('#008751')])


def f_kenya(c):
    red, green = hexc('#BB0000'), hexc('#006600')
    c.hstripes([BLACK, WHITE, red, WHITE, green], [6, 1, 4, 1, 6])
    cx, cy = c.W / 2, 0.5
    for sgn in (-1, 1):
        a = math.radians(22) * sgn
        d = np.array([math.sin(a), -math.cos(a)])
        p0 = np.array([cx, cy]) - d * 0.45
        p1 = np.array([cx, cy]) + d * 0.38
        c.line(p0, p1, 0.016, WHITE)
        n = np.array([-d[1], d[0]])
        tip = p1 + d * 0.11
        c.fill(np.array([p1 - d * 0.01, p1 + d * 0.04 + n * 0.022, tip, p1 + d * 0.04 - n * 0.022]), WHITE)
        c.fill(np.array([p0 + n * 0.012, p0 - n * 0.012, p0 - d * 0.04]), WHITE)
    # the Maasai shield: a pointed oval, black at the sides, red in the middle, white details
    H2, W2 = 0.37, 0.135
    t = np.linspace(-1, 1, 200)
    right = np.stack([cx + W2 * (1 - t ** 2) ** 0.7, cy + H2 * t], 1)
    left = right[::-1].copy()
    left[:, 0] = 2 * cx - left[:, 0]
    shield = np.concatenate([right, left])
    c.fill(shield, BLACK)
    m = c.mask()
    inner = shield.copy()
    inner[:, 0] = cx + (inner[:, 0] - cx) * 0.62
    inner[:, 1] = cy + (inner[:, 1] - cy) * 0.97
    c.fill(inner, 255, dst=m)
    c.paint(m, red)
    for sgn in (-1, 1):
        lune = np.stack([cx + sgn * W2 * 0.8 * (1 - t ** 2) ** 0.7, cy + H2 * 0.75 * t], 1)
        c.polyline(lune[::8], 0.006, WHITE)
    c.line((cx, cy - H2 * 0.82), (cx, cy + H2 * 0.82), 0.012, WHITE)
    c.ellipse(cx, cy, 0.022, 0.05, WHITE)
    c.ellipse(cx, cy, 0.011, 0.03, red)
    for yy in (-0.22, 0.22):
        c.ellipse(cx, cy + yy, 0.02, 0.035, WHITE)
        c.ellipse(cx, cy + yy, 0.008, 0.018, red)


def f_south_africa(c):
    red, blue, green, gold = hexc('#E03C31'), hexc('#001489'), hexc('#007749'), hexc('#FFB81C')
    c.rect(0, 0, c.W, 0.5, red)
    c.rect(0, 0.5, c.W, 1, blue)
    s2 = math.sqrt(2)

    def y_shape(half, color):
        # the two arms run from the hoist corners at 45 degrees and meet the horizontal arm a third of the way along
        c.fill(band_pts((-0.3, -0.3), (0.5, 0.5), half), color)
        c.fill(band_pts((-0.3, 1.3), (0.5, 0.5), half), color)
        c.rect(0.5, 0.5 - half, c.W, 0.5 + half, color)
        c.circle(0.5, 0.5, half, color)

    y_shape(0.1 + 1 / 15, WHITE)                    # green 1/5 of the height, white borders 1/15
    k = 0.1 * s2                                     # inside the green arms: a gold border 1/15 wide, then black
    c.fill(np.array([(0, k), (0.5 - k, 0.5), (0, 1 - k)]), gold)
    k2 = (0.1 + 1 / 15) * s2
    c.fill(np.array([(0, k2), (0.5 - k2, 0.5), (0, 1 - k2)]), BLACK)
    y_shape(0.1, green)


def f_ghana(c):
    c.hstripes([hexc('#CE1126'), hexc('#FCD116'), hexc('#006B3F')])
    r = (1 / 3) / (1 + math.cos(math.pi / 5)) * 0.98
    c.star(c.W / 2, 1 / 3 + r * 1.0 + 0.003, r, BLACK)


def f_ethiopia(c):
    c.hstripes([hexc('#078930'), hexc('#FCDD09'), hexc('#DA121A')])
    cx, cy = c.W / 2, 0.5
    yel = hexc('#FCDD09')
    c.circle(cx, cy, 0.29, hexc('#0F47AF'))
    stroked_pentagram(c, cx, cy + 0.01, 0.215, 0.024, yel)
    for k in range(5):
        a = math.pi / 2 + 2 * math.pi * k / 5
        d = np.array([math.cos(a), math.sin(a)])
        p0 = np.array([cx, cy + 0.01]) + d * 0.1
        p1 = np.array([cx, cy + 0.01]) + d * 0.225
        c.line(p0, p1, 0.016, yel)


def f_germany(c):
    c.hstripes([BLACK, hexc('#DD0000'), hexc('#FFCE00')])


def f_uk(c):
    s = c.sub(0, 0, 60, 30).area(c.W, 1.0)
    union_jack(s)
    s.paste()


def f_france(c):
    c.vstripes([hexc('#0055A4'), WHITE, hexc('#EF4135')])


def f_italy(c):
    c.vstripes([hexc('#009246'), hexc('#F1F2F1'), hexc('#CE2B37')])


def arms_spain(c, cx, cy, s):
    """Spain's coat of arms, simplified: the quartered shield (Castile, Leon, Aragon, Navarre, Granada and the
    Bourbon oval) under the royal crown, between the Pillars of Hercules. s = total height."""
    red, gold, purple, blue = hexc('#C60B1E'), hexc('#E8B400'), hexc('#7b2a6b'), hexc('#1a3a9a')
    dark = hexc('#5a3200')
    silver = hexc('#d8d8d8')
    k = s / 0.86
    P = lambda pts: np.array([(cx + x * k, cy + y * k) for x, y in pts])
    # pillars
    for sgn in (-1, 1):
        x = sgn * 0.34
        c.fill(P([(x - 0.045, -0.12), (x + 0.045, -0.12), (x + 0.045, 0.34), (x - 0.045, 0.34)]), silver)
        c.fill(P([(x - 0.06, -0.15), (x + 0.06, -0.15), (x + 0.06, -0.11), (x - 0.06, -0.11)]), gold)
        c.fill(P([(x - 0.065, 0.33), (x + 0.065, 0.33), (x + 0.065, 0.39), (x - 0.065, 0.39)]), gold)
        c.fill(P([(x - 0.05, -0.15), (x - 0.05, -0.22), (x - 0.02, -0.19), (x, -0.25), (x + 0.02, -0.19),
                  (x + 0.05, -0.22), (x + 0.05, -0.15)]), gold)
        for yy in (0.02, 0.17):
            c.fill(P([(x - 0.075, yy), (x + 0.075, yy - 0.05), (x + 0.075, yy - 0.01), (x - 0.075, yy + 0.04)]), red)
    # shield outline
    t = np.linspace(0, math.pi, 60)
    bottom = [(0.21 * math.cos(a), 0.17 + 0.21 * math.sin(a)) for a in t]
    shield = [(-0.21, -0.2), (0.21, -0.2)] + [(0.21, 0.17)] + bottom[1:-1] + [(-0.21, 0.17)]
    c.fill(P(shield), dark)
    inner = offset_poly(P(shield), -0.008 * k)
    m = c.mask()
    c.fill(inner, 255, dst=m)

    def quarter(x0, y0, x1, y1, col):
        mm = c.mask()
        c.fill(P([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]), 255, dst=mm)
        c.img[(m > 0) & (mm > 0)] = col
        return mm

    quarter(-0.25, -0.25, 0.0, 0.07, red)
    quarter(0.0, -0.25, 0.25, 0.07, WHITE)
    quarter(-0.25, 0.07, 0.0, 0.45, gold)
    quarter(0.0, 0.07, 0.25, 0.45, red)
    for i in range(4):
        x0 = -0.2 + i * 0.05 + 0.012
        mm = c.mask()
        c.fill(P([(x0, 0.07), (x0 + 0.022, 0.07), (x0 + 0.022, 0.45), (x0, 0.45)]), 255, dst=mm)
        c.img[(m > 0) & (mm > 0)] = red
    # castle
    c.fill(P([(-0.16, 0.02), (-0.05, 0.02), (-0.05, -0.06), (-0.07, -0.06), (-0.07, -0.13), (-0.09, -0.13),
              (-0.09, -0.08), (-0.12, -0.08), (-0.12, -0.13), (-0.14, -0.13), (-0.14, -0.06), (-0.16, -0.06)]),
           gold)
    # lion
    c.ellipse(cx + 0.105 * k, cy - 0.05 * k, 0.05 * k, 0.07 * k, purple, rot=0.4)
    c.circle(cx + 0.075 * k, cy - 0.12 * k, 0.03 * k, purple)
    # chains
    for gx in np.linspace(0.04, 0.2, 4):
        for gy in np.linspace(0.12, 0.3, 4):
            c.ring(cx + gx * k, cy + gy * k, 0.009 * k, 0.017 * k, gold)
    # Granada
    mm = quarter(-0.06, 0.32, 0.06, 0.45, WHITE)
    c.circle(cx, cy + 0.36 * k, 0.025 * k, hexc('#d0423a'))
    # Bourbon oval
    c.ellipse(cx, cy - 0.06 * k, 0.065 * k, 0.08 * k, red)
    c.ellipse(cx, cy - 0.06 * k, 0.05 * k, 0.065 * k, blue)
    for dx, dy in ((-0.02, -0.09), (0.02, -0.09), (0.0, -0.04)):
        c.circle(cx + dx * k, cy + dy * k, 0.01 * k, gold)
    # crown
    c.fill(P([(-0.19, -0.21), (0.19, -0.21), (0.17, -0.26), (-0.17, -0.26)]), gold)
    c.fill(P([(-0.16, -0.26), (-0.17, -0.34), (-0.08, -0.42), (0.0, -0.44), (0.08, -0.42), (0.17, -0.34),
              (0.16, -0.26)]), gold)
    c.fill(P([(-0.12, -0.27), (-0.12, -0.33), (-0.04, -0.39), (0.04, -0.39), (0.12, -0.33), (0.12, -0.27)]), red)
    c.fill(P([(-0.015, -0.43), (0.015, -0.43), (0.015, -0.5), (-0.015, -0.5)]), gold)
    c.fill(P([(-0.035, -0.475), (0.035, -0.475), (0.035, -0.455), (-0.035, -0.455)]), gold)
    for x in (-0.12, 0.0, 0.12):
        c.circle(cx + x * k, cy - 0.235 * k, 0.012 * k, hexc('#2a7a3a'))


def f_spain(c):
    c.hstripes([hexc('#AA151B'), hexc('#F1BF00'), hexc('#AA151B')], [1, 2, 1])
    arms_spain(c, 0.5, 0.5, 0.42)


def arms_portugal(c, cx, cy, r):
    """The armillary sphere (radius r) with the shield of Portugal (5 blue escutcheons, 7 castles)."""
    yel, dark = hexc('#FFE000'), hexc('#3a2a00')
    red, blue = hexc('#E8112D'), hexc('#003399')
    w = 0.1 * r
    c.ring(cx, cy, r - w / 2 - 0.012 * r, r + w / 2 + 0.012 * r, dark)
    c.ring(cx, cy, r - w / 2, r + w / 2, yel)
    for rx, ry, rot, oy in ((r, 0.28 * r, 0.0, 0.0), (r * 0.87, 0.2 * r, 0.0, -0.5 * r),
                            (r * 0.87, 0.2 * r, 0.0, 0.5 * r), (r * 0.98, 0.3 * r, -0.45, 0.0),
                            (0.3 * r, r, 0.0, 0.0)):
        c.ellipse_ring(cx, cy + oy, rx, ry, w * 0.8 + 0.024 * r, dark, rot=rot)
        c.ellipse_ring(cx, cy + oy, rx, ry, w * 0.8, yel, rot=rot)
    # shield
    sw, sh = 0.42 * r, 0.5 * r
    t = np.linspace(0, math.pi, 60)
    pts = [(cx - sw, cy - sh), (cx + sw, cy - sh)] + [(cx + sw * math.cos(a), cy + 0.1 * r + (sh - 0.1 * r) * 1.0
                                                       * math.sin(a)) for a in t]
    shield = np.array(pts)
    c.fill(offset_poly(shield, 0.02 * r), dark)
    c.fill(shield, red)
    inner = offset_poly(shield, -0.15 * r)
    c.fill(inner, WHITE)
    for dx, dy in ((0, -0.18), (-0.17, 0.02), (0, 0.02), (0.17, 0.02), (0, 0.22)):
        ex, ey = cx + dx * r, cy + dy * r
        esc = np.array([(ex - 0.065 * r, ey - 0.08 * r), (ex + 0.065 * r, ey - 0.08 * r),
                        (ex + 0.065 * r, ey + 0.03 * r), (ex, ey + 0.09 * r), (ex - 0.065 * r, ey + 0.03 * r)])
        c.fill(esc, blue)
        for px_, py_ in ((0, -0.04), (-0.03, 0.0), (0.03, 0.0), (0, 0.04)):
            c.circle(ex + px_ * r, ey + py_ * r, 0.012 * r, WHITE)
    cas = [(-0.335, -0.42), (0.0, -0.42), (0.335, -0.42), (-0.345, -0.05), (0.345, -0.05), (-0.29, 0.29),
           (0.29, 0.29)]
    for dx, dy in cas:
        x, y = cx + dx * r, cy + dy * r
        q = 0.045 * r
        c.fill(np.array([(x - q, y + q), (x + q, y + q), (x + q, y - q * 0.3), (x + q * 0.5, y - q * 0.3),
                         (x + q * 0.5, y - q), (x - q * 0.5, y - q), (x - q * 0.5, y - q * 0.3), (x - q, y - q * 0.3)]),
               yel)


def f_portugal(c):
    c.rect(0, 0, c.W, 1, hexc('#FF0000'))
    c.rect(0, 0, 0.6, 1, hexc('#006600'))
    arms_portugal(c, 0.6, 0.5, 0.25)


def f_netherlands(c):
    c.hstripes([hexc('#AE1C28'), WHITE, hexc('#21468B')])


def f_poland(c):
    c.hstripes([WHITE, hexc('#DC143C')])


def f_ukraine(c):
    c.hstripes([hexc('#0057B7'), hexc('#FFD700')])


def f_russia(c):
    c.hstripes([WHITE, hexc('#0039A6'), hexc('#D52B1E')])


def f_sweden(c):
    c.rect(0, 0, c.W, 1, hexc('#006AA7'))
    y = hexc('#FECC00')
    c.rect(0.5, 0, 0.7, 1, y)
    c.rect(0, 0.4, c.W, 0.6, y)


def f_greece(c):
    blue = hexc('#0D5EAF')
    for i in range(9):
        c.rect(0, i / 9, c.W, (i + 1) / 9 + 1e-3, blue if i % 2 == 0 else WHITE)
    c.rect(0, 0, 5 / 9, 5 / 9, blue)
    c.rect(2 / 9, 0, 3 / 9, 5 / 9, WHITE)
    c.rect(0, 2 / 9, 5 / 9, 3 / 9, WHITE)


def f_romania(c):
    c.vstripes([hexc('#002B7F'), hexc('#FCD116'), hexc('#CE1126')])


def f_switzerland(c):
    c.rect(0, 0, c.W, 1, hexc('#DA291C'))
    u = 1 / 32
    c.rect(13 * u, 6 * u, 19 * u, 26 * u, WHITE)
    c.rect(6 * u, 13 * u, 26 * u, 19 * u, WHITE)


def f_ireland(c):
    c.vstripes([hexc('#169B62'), WHITE, hexc('#FF883E')])


def f_usa(c):
    red, blue = hexc('#B22234'), hexc('#3C3B6E')
    for i in range(13):
        c.rect(0, i / 13, c.W, (i + 1) / 13 + 1e-3, red if i % 2 == 0 else WHITE)
    c.rect(0, 0, 0.76, 7 / 13, blue)
    for row in range(9):
        y = 0.054 * (row + 1)
        cols = range(1, 12, 2) if row % 2 == 0 else range(2, 11, 2)
        for col in cols:
            c.star(0.063 * col, y, 0.0616 / 2, WHITE)


MAPLE = ("m -90 2030 45 -863 a 95 95 0 0 0 -111 -98 l -859 151 116 -320 a 65 65 0 0 0 -20 -73 l -941 -762 "
         "212 -99 a 65 65 0 0 0 34 -79 l -186 -572 542 115 a 65 65 0 0 0 73 -38 l 105 -247 423 454 "
         "a 65 65 0 0 0 111 -57 l -204 -1052 327 189 a 65 65 0 0 0 91 -27 l 332 -652 332 652 "
         "a 65 65 0 0 0 91 27 l 327 -189 -204 1052 a 65 65 0 0 0 111 57 l 423 -454 105 247 "
         "a 65 65 0 0 0 73 38 l 542 -115 -186 572 a 65 65 0 0 0 34 79 l 212 99 -941 762 "
         "a 65 65 0 0 0 -20 73 l 116 320 -859 -151 a 95 95 0 0 0 -111 98 l 45 863 z")


def maple_leaf():
    """The maple leaf of the Canadian flag (an 11-point leaf) from the outline in the flag's construction sheet,
    in units where the flag is 9600 x 4800 and (0, 0) is its centre."""
    tok = MAPLE.replace(',', ' ').split()
    pts, cur, cmd, i = [], np.zeros(2), None, 0
    while i < len(tok):
        t = tok[i]
        if t in 'mlaz':
            cmd = t
            i += 1
            if t == 'z':
                break
            continue
        if cmd == 'm':
            cur = cur + [float(tok[i]), float(tok[i + 1])]
            pts.append(cur.copy())
            i += 2
            cmd = 'l'
        elif cmd == 'l':
            cur = cur + [float(tok[i]), float(tok[i + 1])]
            pts.append(cur.copy())
            i += 2
        elif cmd == 'a':
            rx, ry, phi, la, sw, dx, dy = [float(v) for v in tok[i:i + 7]]
            nxt = cur + [dx, dy]
            pts.extend(svg_arc(cur, nxt, rx, ry, math.radians(phi), int(la), int(sw)))
            cur = nxt
            i += 7
    return np.array(pts)


def f_canada(c):
    red = hexc('#D52B1E')
    c.rect(0, 0, c.W, 1, WHITE)
    c.rect(0, 0, 0.5, 1, red)
    c.rect(1.5, 0, 2.0, 1, red)
    leaf = maple_leaf() / 4800.0 + np.array([1.0, 0.5])
    c.fill(leaf, red)


def arms_mexico(c, cx, cy, s):
    """Mexico's coat of arms, simplified: the eagle on a prickly pear devouring a snake, above the lake, inside a
    wreath of oak and laurel tied with a tricolour ribbon. s = height."""
    brown, dbrown = hexc('#8a5a2b'), hexc('#4a2a10')
    green, dgreen, lgreen = hexc('#3a7d2a'), hexc('#1f5a1c'), hexc('#7ab648')
    k = s
    P = lambda pts: np.array([(cx + x * k, cy + y * k) for x, y in pts])
    # wreath
    for side, col in ((-1, dgreen), (1, green)):
        for a in np.linspace(math.radians(95), math.radians(185), 9):
            ang = a if side < 0 else math.pi - a      # y is down: 90 degrees is the bottom of the wreath
            x, y = 0.42 * math.cos(ang), 0.08 + 0.36 * math.sin(ang)
            tangent = ang + math.pi / 2
            c.ellipse(cx + x * k, cy + y * k, 0.05 * k, 0.022 * k, col, rot=tangent + 0.5 * side)
            c.ellipse(cx + x * 0.9 * k, cy + (0.08 + 0.36 * 0.9 * math.sin(ang)) * k, 0.04 * k, 0.018 * k, col,
                      rot=tangent - 0.6 * side)
    # ribbon
    for i, col in enumerate((hexc('#006847'), WHITE, hexc('#CE1126'))):
        c.fill(P([(-0.12 + i * 0.08, 0.42), (-0.04 + i * 0.08, 0.42), (-0.04 + i * 0.08, 0.47),
                  (-0.12 + i * 0.08, 0.47)]), col)
    # lake and rock
    c.ellipse(cx, cy + 0.33 * k, 0.17 * k, 0.035 * k, hexc('#3a7fbf'))
    c.ellipse(cx, cy + 0.3 * k, 0.07 * k, 0.03 * k, hexc('#8a7a6a'))
    # cactus
    for (x, y, rx, ry, rot) in ((0.0, 0.2, 0.05, 0.09, 0.0), (-0.07, 0.13, 0.04, 0.07, -0.6),
                                (0.07, 0.12, 0.04, 0.07, 0.6), (-0.11, 0.05, 0.03, 0.05, -0.9),
                                (0.1, 0.04, 0.03, 0.055, 0.9)):
        c.ellipse(cx + x * k, cy + y * k, rx * k, ry * k, green, rot=rot)
    for (x, y) in ((-0.13, 0.0), (0.12, -0.01), (-0.04, 0.08), (0.05, 0.07)):
        c.circle(cx + x * k, cy + y * k, 0.014 * k, hexc('#c8102e'))
    # wings (raised)
    c.fill(P([(0.02, -0.08), (0.14, -0.3), (0.24, -0.4), (0.3, -0.36), (0.26, -0.28), (0.3, -0.26),
              (0.24, -0.18), (0.27, -0.15), (0.18, -0.08), (0.2, -0.04), (0.1, -0.02)]), brown)
    c.fill(P([(-0.02, -0.1), (-0.06, -0.3), (-0.02, -0.42), (0.05, -0.44), (0.04, -0.36), (0.09, -0.36),
              (0.07, -0.28), (0.11, -0.24), (0.06, -0.12)]), dbrown)
    # body, tail, legs
    c.ellipse(cx + 0.03 * k, cy - 0.06 * k, 0.085 * k, 0.12 * k, brown, rot=-0.5)
    c.fill(P([(0.08, 0.0), (0.2, 0.05), (0.17, 0.09), (0.06, 0.04)]), dbrown)
    c.line((cx - 0.0 * k, cy + 0.0 * k), (cx - 0.02 * k, cy + 0.08 * k), 0.025 * k, hexc('#c8a050'))
    # head and beak, facing the hoist
    c.circle(cx - 0.1 * k, cy - 0.17 * k, 0.045 * k, brown)
    c.fill(P([(-0.13, -0.19), (-0.19, -0.16), (-0.13, -0.15)]), hexc('#e0b030'))
    c.circle(cx - 0.11 * k, cy - 0.18 * k, 0.008 * k, BLACK)
    # snake from the beak
    sn = np.array([(-0.19, -0.16), (-0.22, -0.1), (-0.17, -0.05), (-0.22, 0.0), (-0.18, 0.05)])
    c.polyline(P(sn), 0.022 * k, hexc('#2f8f3a'))


def f_mexico(c):
    c.vstripes([hexc('#006847'), WHITE, hexc('#CE1126')])
    arms_mexico(c, c.W / 2, 0.5, 0.46)


BRAZIL_STARS = [  # (u, v) in units of the globe's radius (v down), size 1..5 (5 = biggest)
    (0.27, -0.33, 4),                                                          # Spica, above the band
    (-0.62, 0.30, 5), (-0.78, 0.16, 3), (-0.52, 0.47, 3), (-0.73, 0.42, 2), (-0.88, 0.34, 2),   # Canis Major
    (-0.86, 0.04, 4),                                                          # Procyon
    (-0.32, 0.66, 5),                                                          # Canopus
    (0.0, 0.73, 4), (-0.16, 0.52, 3), (0.0, 0.33, 3), (0.14, 0.47, 3), (0.07, 0.57, 1),     # Crux
    (0.58, 0.32, 4), (0.48, 0.2, 2), (0.66, 0.18, 2), (0.66, 0.44, 2), (0.58, 0.56, 2), (0.47, 0.64, 2),
    (0.36, 0.58, 2), (0.32, 0.46, 2), (0.5, 0.42, 1),                          # Scorpius
    (0.28, 0.79, 3), (0.17, 0.86, 2), (0.38, 0.85, 2),                         # Triangulum Australe
    (0.0, 0.92, 2),                                                            # Sigma Octantis
    (0.42, 0.06, 2),                                                           # Hydra
]


def f_brazil(c):
    green, yel, blue = hexc('#009C3B'), hexc('#FFDF00'), hexc('#002776')
    c.rect(0, 0, c.W, 1, green)
    u = 1 / 14
    W = c.W
    c.fill(np.array([(1.7 * u, 0.5), (W / 2, 1.7 * u), (W - 1.7 * u, 0.5), (W / 2, 1 - 1.7 * u)]), yel)
    cx, cy, R = W / 2, 0.5, 3.5 * u
    c.circle(cx, cy, R, blue)
    # the band: arcs of radius 8 and 8.5 units about a point 2 units left of and 7.5 units below the centre
    m = c.mask()
    ax, ay = cx - 2 * u, cy + 7.5 * u
    c.fill(arc_band(ax, ay, 8 * u, 8.5 * u, math.radians(200), math.radians(340), 400), 255, dst=m)
    disc = c.mask()
    c.circle(cx, cy, R, 255, dst=disc)
    c.img[(m > 0) & (disc > 0)] = WHITE
    c.text_arc('ORDEM E PROGRESSO', ax, ay, 8.25 * u, -math.acos(2 / 8.25), 0.5 * u, hexc('#009C3B'),
               track=0.02)
    for (sx, sy, sz) in BRAZIL_STARS:
        r = R * (0.02 + 0.012 * sz)
        c.star(cx + sx * R, cy + sy * R, r, WHITE)


def sun_of_may(c, cx, cy, r):
    gold, brown = hexc('#F6B40E'), hexc('#85340A')
    face = 0.42 * r
    for k in range(32):
        a = -math.pi / 2 + 2 * math.pi * k / 32
        d = np.array([math.cos(a), math.sin(a)])
        n = np.array([-d[1], d[0]])
        C = np.array([cx, cy])
        if k % 2 == 0:
            tri = np.array([C + d * face * 0.95 + n * 0.075 * r, C + d * r, C + d * face * 0.95 - n * 0.075 * r])
            c.fill(offset_poly(tri, 0.012 * r), brown)
            c.fill(tri, gold)
        else:
            t = np.linspace(0, 1, 40)
            rr = face * 0.95 + (r * 0.93 - face * 0.95) * t
            wob = 0.035 * r * np.sin(t * 3 * math.pi) * (1 - t * 0.5)
            wid = 0.05 * r * (1 - t) + 0.004 * r
            cl = C + np.outer(rr, d) + np.outer(wob, n)
            left = cl + np.outer(wid, n)
            right = cl - np.outer(wid, n)
            shape = np.concatenate([left, right[::-1]])
            c.fill(offset_poly(shape[::3], 0.0) if False else shape, brown)
            sh2 = np.concatenate([cl + np.outer(wid * 0.6, n), (cl - np.outer(wid * 0.6, n))[::-1]])
            c.fill(sh2, gold)
    c.circle(cx, cy, face + 0.02 * r, brown)
    c.circle(cx, cy, face, gold)
    # the face
    for sgn in (-1, 1):
        c.ellipse(cx + sgn * 0.15 * r, cy - 0.07 * r, 0.06 * r, 0.03 * r, brown)
        c.ellipse(cx + sgn * 0.15 * r, cy - 0.07 * r, 0.025 * r, 0.025 * r, gold)
        c.line((cx + sgn * 0.08 * r, cy - 0.15 * r), (cx + sgn * 0.23 * r, cy - 0.14 * r), 0.018 * r, brown)
    c.line((cx, cy - 0.06 * r), (cx - 0.02 * r, cy + 0.08 * r), 0.018 * r, brown)
    c.line((cx - 0.02 * r, cy + 0.08 * r), (cx + 0.03 * r, cy + 0.09 * r), 0.018 * r, brown)
    c.ellipse(cx, cy + 0.2 * r, 0.09 * r, 0.03 * r, brown)
    c.ellipse(cx, cy + 0.2 * r, 0.06 * r, 0.012 * r, gold)


def f_argentina(c):
    c.hstripes([hexc('#74ACDF'), WHITE, hexc('#74ACDF')])
    sun_of_may(c, c.W / 2, 0.5, 0.145)


def f_colombia(c):
    c.hstripes([hexc('#FCD116'), hexc('#003893'), hexc('#CE1126')], [2, 1, 1])


def f_peru(c):
    c.vstripes([hexc('#D91023'), WHITE, hexc('#D91023')])


def f_chile(c):
    c.hstripes([WHITE, hexc('#D52B1E')])
    c.rect(0, 0, 0.5, 0.5, hexc('#0039A6'))
    c.star(0.25, 0.25, 0.125, WHITE)


def f_venezuela(c):
    c.hstripes([hexc('#FFCC00'), hexc('#00247D'), hexc('#CF142B')])
    R, yc = 0.3, 0.69
    for k in range(8):
        phi = math.radians(-72 + 144 * k / 7)
        x, y = c.W / 2 + R * math.sin(phi), yc - R * math.cos(phi)
        c.star(x, y, 0.033, WHITE, rot=-math.pi / 2 + phi)


def f_cuba(c):
    blue = hexc('#002A8F')
    for i in range(5):
        c.rect(0, i / 5, c.W, (i + 1) / 5 + 1e-3, blue if i % 2 == 0 else WHITE)
    h = math.sqrt(3) / 2
    c.fill(np.array([(0, 0), (h, 0.5), (0, 1)]), hexc('#CB1515'))
    c.star(h / 3, 0.5, 0.15, WHITE)


def f_jamaica(c):
    green, gold = hexc('#009B3A'), hexc('#FED100')
    c.rect(0, 0, c.W, 1, BLACK)
    c.fill(np.array([(0, 0), (c.W, 0), (c.W / 2, 0.5)]), green)
    c.fill(np.array([(0, 1), (c.W, 1), (c.W / 2, 0.5)]), green)
    hw = 0.08
    for p0, p1 in (((0, 0), (c.W, 1)), ((0, 1), (c.W, 0))):
        d = np.array(p1, float) - p0
        d /= np.linalg.norm(d)
        c.fill(band_pts(np.array(p0) - d * 0.3, np.array(p1) + d * 0.3, hw), gold)


def f_australia(c):
    c.rect(0, 0, c.W, 1, hexc('#012169'))
    s = c.sub(0, 0, 60, 30).area(1.0, 0.5)
    union_jack(s)
    s.paste()
    c.star(0.5, 0.75, 0.15, WHITE, n=7, ratio=4 / 9)
    for (x, y, r, n) in ((1.5, 5 / 6, 1 / 14, 7), (1.25, 0.4375, 1 / 14, 7), (1.5, 1 / 6, 1 / 14, 7),
                         (1.722, 0.371, 1 / 14, 7), (1.6, 0.5417, 1 / 24, 5)):
        c.star(x, y, r, WHITE, n=n, ratio=4 / 9 if n == 7 else None)


def f_new_zealand(c):
    c.rect(0, 0, c.W, 1, hexc('#012169'))
    s = c.sub(0, 0, 60, 30).area(1.0, 0.5)
    union_jack(s)
    s.paste()
    red = hexc('#C8102E')
    for (x, y, r) in ((1.5, 0.8, 0.07), (1.29, 0.465, 0.06), (1.5, 0.2, 0.06), (1.715, 0.41, 0.05)):
        c.star(x, y, r, WHITE)
        c.star(x, y, r * 0.72, red)


# code, name, aspect (width / height), drawing function
COUNTRIES = [
    ('IN', 'INDIA', 1.5, f_india), ('ID', 'INDONESIA', 1.5, f_indonesia), ('PK', 'PAKISTAN', 1.5, f_pakistan),
    ('BD', 'BANGLADESH', 5 / 3, f_bangladesh), ('PH', 'PHILIPPINES', 2.0, f_philippines),
    ('VN', 'VIETNAM', 1.5, f_vietnam), ('JP', 'JAPAN', 1.5, f_japan), ('KR', 'SOUTH KOREA', 1.5, f_korea),
    ('CN', 'CHINA', 1.5, f_china), ('TH', 'THAILAND', 1.5, f_thailand), ('MY', 'MALAYSIA', 2.0, f_malaysia),
    ('TR', 'TURKEY', 1.5, f_turkey), ('UZ', 'UZBEKISTAN', 2.0, f_uzbekistan), ('AE', 'UAE', 2.0, f_uae),
    ('EG', 'EGYPT', 1.5, f_egypt), ('MA', 'MOROCCO', 1.5, f_morocco), ('DZ', 'ALGERIA', 1.5, f_algeria),
    ('NG', 'NIGERIA', 2.0, f_nigeria), ('KE', 'KENYA', 1.5, f_kenya), ('ZA', 'SOUTH AFRICA', 1.5, f_south_africa),
    ('GH', 'GHANA', 1.5, f_ghana), ('ET', 'ETHIOPIA', 2.0, f_ethiopia), ('DE', 'GERMANY', 5 / 3, f_germany),
    ('GB', 'UK', 2.0, f_uk), ('FR', 'FRANCE', 1.5, f_france), ('IT', 'ITALY', 1.5, f_italy),
    ('ES', 'SPAIN', 1.5, f_spain), ('PT', 'PORTUGAL', 1.5, f_portugal), ('NL', 'NETHERLANDS', 1.5, f_netherlands),
    ('PL', 'POLAND', 1.6, f_poland), ('UA', 'UKRAINE', 1.5, f_ukraine), ('RU', 'RUSSIA', 1.5, f_russia),
    ('SE', 'SWEDEN', 1.6, f_sweden), ('GR', 'GREECE', 1.5, f_greece), ('RO', 'ROMANIA', 1.5, f_romania),
    ('CH', 'SWITZERLAND', 1.0, f_switzerland), ('IE', 'IRELAND', 2.0, f_ireland), ('US', 'USA', 1.9, f_usa),
    ('CA', 'CANADA', 2.0, f_canada), ('MX', 'MEXICO', 1.75, f_mexico), ('BR', 'BRAZIL', 10 / 7, f_brazil),
    ('AR', 'ARGENTINA', 14 / 9, f_argentina), ('CO', 'COLOMBIA', 1.5, f_colombia), ('PE', 'PERU', 1.5, f_peru),
    ('CL', 'CHILE', 1.5, f_chile), ('VE', 'VENEZUELA', 1.5, f_venezuela), ('CU', 'CUBA', 2.0, f_cuba),
    ('JM', 'JAMAICA', 2.0, f_jamaica), ('AU', 'AUSTRALIA', 2.0, f_australia),
    ('NZ', 'NEW ZEALAND', 2.0, f_new_zealand),
]
BY_CODE = {c[0]: c for c in COUNTRIES}


def draw(code, height=1024, ss=4):
    _, _, aspect, fn = BY_CODE[code]
    c = Canvas(aspect, 1.0, height, ss=ss)
    fn(c)
    return c.finish()


def texture(code, height=1024):
    """The flag as an RGB array, cached on disk."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f'{code}_{height}.png')
    if os.path.exists(path):
        return np.asarray(Image.open(path).convert('RGB'))
    img = draw(code, height)
    Image.fromarray(img).save(path)
    return img


def emoji(code, size=128):
    f = ImageFont.truetype('/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf', 109)
    s = ''.join(chr(0x1F1E6 + ord(ch) - 65) for ch in code)
    im = Image.new('RGBA', (140, 140), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((0, 0), s, font=f, embedded_color=True)
    return im


if __name__ == '__main__':
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'preview', 'flags.png')
    only = sys.argv[2].split(',') if len(sys.argv) > 2 else None
    items = [c for c in COUNTRIES if not only or c[0] in only]
    cols = 5
    cw, ch = 560, 220
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * cw, rows * ch), (50, 50, 50))
    d = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(FONT_BOLD, 18)
    for i, (code, name, aspect, fn) in enumerate(items):
        img = draw(code, height=180, ss=4)
        x, y = (i % cols) * cw + 8, (i // cols) * ch + 8
        sheet.paste(Image.fromarray(img), (x, y))
        e = emoji(code).resize((170, 170), Image.BICUBIC)
        sheet.paste(e, (x + int(180 * aspect) + 8, y - 4), e)
        d.text((x, y + 184), f'{code} {name}', fill=(255, 255, 255), font=font)
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    sheet.save(out)
    print('sheet:', out)
