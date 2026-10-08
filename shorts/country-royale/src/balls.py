"""Flag balls: each flag wrapped onto a glossy sphere (countryball style, with a dark outline), plus their eyes.

The flag is mapped onto the disc with the elliptical grid mapping (disc to square), so the whole flag shows: the
middle is undistorted and the edges of the flag wrap round the sides of the ball. Wide flags are wrapped a little
further, so a 1:2 flag keeps its canton and its fly in view. The shading (a soft key light from the top left, a
specular highlight, a darker rim) is baked into a mip chain of premultiplied RGBA sprites; the eyes are drawn on each
frame because they move, blink and react.
"""
import math

import cv2
import numpy as np

import flags as FL

BASE = 512          # largest sprite, pixels across
OUTLINE = 0.045     # outline thickness, in ball radii
LIGHT = np.array([-0.48, -0.62, 0.62])
LIGHT = LIGHT / np.linalg.norm(LIGHT)


def disc_to_square(u, v):
    u2, v2 = u * u, v * v
    r2 = 2 * math.sqrt(2)
    a = 2 + u2 - v2
    s = 0.5 * np.sqrt(np.maximum(0, a + r2 * u)) - 0.5 * np.sqrt(np.maximum(0, a - r2 * u))
    b = 2 - u2 + v2
    t = 0.5 * np.sqrt(np.maximum(0, b + r2 * v)) - 0.5 * np.sqrt(np.maximum(0, b - r2 * v))
    return np.clip(s, -1, 1), np.clip(t, -1, 1)


def flag_uv(u, v, aspect):
    """Flag coordinates (x in 0..aspect, y in 0..1) seen at the point (u, v) of the unit disc."""
    s, t = disc_to_square(u, v)
    half = aspect / 2
    fx = half + 0.5 * s + (half - 0.5) * s * np.abs(s) ** 2
    fy = (t + 1) / 2
    return fx, fy


def render_ball(code, size=BASE, ss=2):
    """The shaded ball, premultiplied RGBA float32 in 0..1, size x size."""
    tex = FL.texture(code, 1024).astype(np.float32) / 255.0
    th, tw = tex.shape[:2]
    aspect = tw / th
    n = size * ss
    # pre-filter the texture to about the sampling rate in the middle of the ball
    scale = min(1.0, (n * 1.4) / th)
    if scale < 1.0:
        tex = cv2.resize(tex, (max(2, int(tw * scale)), max(2, int(th * scale))), interpolation=cv2.INTER_AREA)
    th2, tw2 = tex.shape[:2]
    c = (np.arange(n) + 0.5) / n * 2 - 1
    u, v = np.meshgrid(c, c)
    rho = np.sqrt(u * u + v * v)
    k = 1.0 / (1.0 - OUTLINE)
    uu, vv = u * k, v * k
    rr = np.minimum(np.sqrt(uu * uu + vv * vv), 0.99999)
    sc = np.where(rr > 0, np.minimum(1.0, 0.99999 / np.maximum(rr, 1e-9)), 1.0)
    fx, fy = flag_uv(uu * np.minimum(sc, 1), vv * np.minimum(sc, 1), aspect)
    mx = (fx / aspect * tw2 - 0.5).astype(np.float32)
    my = (fy * th2 - 0.5).astype(np.float32)
    col = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # sphere normal on the inner disc
    nz = np.sqrt(np.clip(1 - rr * rr, 0, 1))
    nx, ny = uu * np.minimum(sc, 1), vv * np.minimum(sc, 1)
    ndl = np.clip(nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2], 0, 1)
    shade = 0.58 + 0.5 * ndl
    shade *= 1 - 0.28 * (1 - nz) ** 2
    h = LIGHT + np.array([0, 0, 1.0])
    h /= np.linalg.norm(h)
    ndh = np.clip(nx * h[0] + ny * h[1] + nz * h[2], 0, 1)
    spec = 0.55 * ndh ** 70 + 0.12 * ndh ** 9
    rim = 0.18 * (1 - nz) ** 3 * np.clip(-(nx * LIGHT[0] + ny * LIGHT[1]) + 0.3, 0, 1)
    rgb = col * shade[..., None] + (spec + rim)[..., None]
    # dark outline (countryball style)
    inner = np.clip((1 - OUTLINE - rho) * n / 2 + 0.5, 0, 1)[..., None]
    outline = np.array([0.06, 0.06, 0.09], np.float32)
    rgb = rgb * inner + outline * (1 - inner)
    alpha = np.clip((1 - rho) * n / 2 + 0.5, 0, 1)[..., None]
    out = np.concatenate([np.clip(rgb, 0, 1) * alpha, alpha], -1).astype(np.float32)
    if ss > 1:
        out = cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)
    return out


class Mips:
    """A ball's sprite at 512, 256, 128, 64 and 32 pixels."""

    def __init__(self, sprite):
        self.levels = [sprite]
        while self.levels[-1].shape[0] > 32:
            s = self.levels[-1].shape[0] // 2
            self.levels.append(cv2.resize(self.levels[-1], (s, s), interpolation=cv2.INTER_AREA))

    def pick(self, d):
        """The smallest level at least d pixels across."""
        best = self.levels[0]
        for lv in self.levels:
            if lv.shape[0] >= d:
                best = lv
        return best


EYE_COLOR = (1.0, 1.0, 1.0)
EYE_EDGE = (0.04, 0.04, 0.06)


def draw_eyes(img, look=(0.0, 0.0), blink=1.0, mood='normal', amount=1.0):
    """Countryball eyes (white, no pupils) drawn onto a sprite (premultiplied RGBA float, modified in place).
    look: where they look, -1..1 in each axis; blink: 1 open, 0 shut; mood: normal, shock, happy, sad, dead."""
    n = img.shape[0]
    R = n / 2
    sh = 3
    S = 1 << sh
    cx0, cy0 = R, R
    lx, ly = look
    th = max(1.0, 0.05 * R)
    for sgn in (-1, 1):
        ex = cx0 + (sgn * 0.31 + lx * 0.13) * R
        ey = cy0 + (-0.16 + ly * 0.11) * R
        rx, ry = 0.185 * R, 0.24 * R
        if mood == 'shock':
            rx, ry = rx * 1.22, ry * 1.25
        if mood in ('happy', 'dead', 'sad'):
            t = int(max(1, round(th * 1.1)))
            if mood == 'happy':     # ^ ^
                pts = np.array([(ex - rx, ey + ry * 0.25), (ex, ey - ry * 0.55), (ex + rx, ey + ry * 0.25)])
                cv2.polylines(img, [np.round(pts * S).astype(np.int32)], False, (*EYE_EDGE, 1.0), t * 2,
                              cv2.LINE_AA, sh)
                cv2.polylines(img, [np.round(pts * S).astype(np.int32)], False, (*EYE_COLOR, 1.0), t,
                              cv2.LINE_AA, sh)
            elif mood == 'dead':    # x x
                for a, b in (((-1, -1), (1, 1)), ((-1, 1), (1, -1))):
                    p0 = (int((ex + a[0] * rx * 0.8) * S), int((ey + a[1] * ry * 0.65) * S))
                    p1 = (int((ex + b[0] * rx * 0.8) * S), int((ey + b[1] * ry * 0.65) * S))
                    cv2.line(img, p0, p1, (*EYE_EDGE, 1.0), t * 2 + 1, cv2.LINE_AA, sh)
                    cv2.line(img, p0, p1, (*EYE_COLOR, 1.0), t, cv2.LINE_AA, sh)
            else:                   # sad: half-closed, tilted
                ry2 = ry * 0.55
                c = (int(ex * S), int((ey + ry * 0.3) * S))
                ax = (int(rx * S), int(ry2 * S))
                cv2.ellipse(img, c, ax, -12 * sgn, 0, 360, (*EYE_EDGE, 1.0), -1, cv2.LINE_AA, sh)
                ax2 = (int(max(1, (rx - th)) * S), int(max(1, (ry2 - th)) * S))
                cv2.ellipse(img, c, ax2, -12 * sgn, 0, 360, (*EYE_COLOR, 1.0), -1, cv2.LINE_AA, sh)
            continue
        ry *= max(0.12, blink)
        c = (int(round(ex * S)), int(round(ey * S)))
        cv2.ellipse(img, c, (int(rx * S), int(ry * S)), 0, 0, 360, (*EYE_EDGE, 1.0), -1, cv2.LINE_AA, sh)
        cv2.ellipse(img, c, (int(max(1, rx - th) * S), int(max(0.5, ry - th) * S)), 0, 0, 360,
                    (*EYE_COLOR, 1.0), -1, cv2.LINE_AA, sh)
    return img


if __name__ == '__main__':
    import os
    import sys
    from PIL import Image
    out = sys.argv[1] if len(sys.argv) > 1 else '../preview/balls.png'
    codes = [c[0] for c in FL.COUNTRIES]
    cols = 10
    D = 150
    sheet = np.zeros((5 * (D + 20), cols * (D + 20), 3), np.float32) + np.array([0.05, 0.05, 0.12])
    for i, code in enumerate(codes):
        m = Mips(render_ball(code))
        spr = cv2.resize(m.pick(D * 2), (D * 2, D * 2), interpolation=cv2.INTER_AREA).copy()
        draw_eyes(spr, look=(0.3, 0.1), mood=['normal', 'shock', 'happy', 'sad', 'dead'][i % 5] if i < 5 else
                  'normal')
        spr = cv2.resize(spr, (D, D), interpolation=cv2.INTER_AREA)
        x, y = (i % cols) * (D + 20) + 10, (i // cols) * (D + 20) + 10
        a = spr[..., 3:4]
        sheet[y:y + D, x:x + D] = sheet[y:y + D, x:x + D] * (1 - a) + spr[..., :3]
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    Image.fromarray((np.clip(sheet, 0, 1) * 255).astype(np.uint8)).save(out)
    print('balls:', out)
