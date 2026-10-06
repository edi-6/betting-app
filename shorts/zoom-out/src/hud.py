"""Pasting RGBA sprites (the pixel font's text) onto a frame."""
import numpy as np


def over(dst, src, x, y, opacity=1.0):
    """Alpha-blend the RGBA uint8 sprite src onto the float RGB frame dst (in place) with its top left at (x, y)."""
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
