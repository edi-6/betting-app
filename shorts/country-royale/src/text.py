"""Text and emoji as premultiplied RGBA sprites (float32, 0..1), cached, and a compositor for them."""
import functools

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
NARROW = '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf'
EMOJI = '/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf'


@functools.lru_cache(maxsize=64)
def _font(path, size):
    return ImageFont.truetype(path, size)


@functools.lru_cache(maxsize=2048)
def text(s, size, fill=(255, 255, 255), stroke=0, stroke_fill=(0, 0, 0), font=BOLD, shadow=0, grad=None):
    """A line of text. grad: optional (top colour, bottom colour) fill gradient."""
    f = _font(font, size)
    pad = stroke + shadow + 4
    l, t, r, b = f.getbbox(s, stroke_width=stroke)
    w, h = r - l + 2 * pad, b - t + 2 * pad
    org = (pad - l, pad - t)
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if shadow:
        sh = Image.new('L', (w, h), 0)
        ImageDraw.Draw(sh).text((org[0] + shadow, org[1] + shadow), s, font=f, fill=255, stroke_width=stroke)
        sh = np.asarray(sh, np.float32) / 255.0
        sh = cv2.GaussianBlur(sh, (0, 0), max(1.0, shadow * 0.6)) * 0.6
    if stroke:
        d.text(org, s, font=f, fill=stroke_fill + (255,), stroke_width=stroke, stroke_fill=stroke_fill + (255,))
    if grad is None:
        d.text(org, s, font=f, fill=tuple(fill) + (255,))
        arr = np.asarray(im, np.float32) / 255.0
    else:
        m = Image.new('L', (w, h), 0)
        ImageDraw.Draw(m).text(org, s, font=f, fill=255)
        m = np.asarray(m, np.float32)[..., None] / 255.0
        arr = np.asarray(im, np.float32) / 255.0
        ys = np.linspace(0, 1, h)[:, None, None]
        top, bot = np.array(grad[0], np.float32) / 255, np.array(grad[1], np.float32) / 255
        g = top * (1 - ys) + bot * ys
        arr[..., :3] = arr[..., :3] * (1 - m) + g * m
        arr[..., 3:] = np.maximum(arr[..., 3:], m)
    out = arr.copy()
    out[..., :3] *= out[..., 3:]
    if shadow:
        a = out[..., 3:]
        out = out + np.concatenate([np.zeros_like(out[..., :3]), sh[..., None]], -1) * (1 - a)
    out.setflags(write=False)
    return out


@functools.lru_cache(maxsize=64)
def emoji(ch, size):
    f = _font(EMOJI, 109)
    im = Image.new('RGBA', (160, 160), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((8, 8), ch, font=f, embedded_color=True)
    arr = np.asarray(im, np.float32) / 255.0
    ys, xs = np.nonzero(arr[..., 3] > 0.01)
    arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = arr.shape[:2]
    s = size / h
    arr = cv2.resize(arr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    arr[..., :3] *= arr[..., 3:]
    arr.setflags(write=False)
    return arr


def hstack(*sprites, gap=8):
    """Sprites side by side, centred vertically."""
    h = max(s.shape[0] for s in sprites)
    w = sum(s.shape[1] for s in sprites) + gap * (len(sprites) - 1)
    out = np.zeros((h, w, 4), np.float32)
    x = 0
    for s in sprites:
        y = (h - s.shape[0]) // 2
        out[y:y + s.shape[0], x:x + s.shape[1]] = s
        x += s.shape[1] + gap
    return out


def over(dst, spr, x, y, opacity=1.0, scale=1.0, angle=0.0, additive=False):
    """Composites a premultiplied sprite onto dst (float RGB) with its centre at (x, y)."""
    if opacity <= 0.003 or scale <= 0.01:
        return
    h, w = spr.shape[:2]
    if scale != 1.0 or angle != 0.0:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        bw = int(abs(w * scale * np.cos(np.radians(angle))) + abs(h * scale * np.sin(np.radians(angle)))) + 4
        bh = int(abs(h * scale * np.cos(np.radians(angle))) + abs(w * scale * np.sin(np.radians(angle)))) + 4
        m[0, 2] += bw / 2 - w / 2
        m[1, 2] += bh / 2 - h / 2
        interp = cv2.INTER_AREA if scale < 0.7 and angle == 0.0 else cv2.INTER_LINEAR
        if interp == cv2.INTER_AREA:
            spr = cv2.resize(spr, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                             interpolation=cv2.INTER_AREA)
        else:
            spr = cv2.warpAffine(spr, m, (bw, bh), flags=interp, borderMode=cv2.BORDER_CONSTANT)
        h, w = spr.shape[:2]
    x0, y0 = int(round(x - w / 2)), int(round(y - h / 2))
    x1, y1 = x0 + w, y0 + h
    H, W = dst.shape[:2]
    sx0, sy0 = max(0, -x0), max(0, -y0)
    x0c, y0c, x1c, y1c = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1c <= x0c or y1c <= y0c:
        return
    s = spr[sy0:sy0 + (y1c - y0c), sx0:sx0 + (x1c - x0c)]
    d = dst[y0c:y1c, x0c:x1c]
    if additive:
        d += s[..., :3] * opacity
    else:
        d *= 1 - s[..., 3:] * opacity
        d += s[..., :3] * opacity
