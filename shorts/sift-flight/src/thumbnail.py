"""Render the 1080x1920 cover: a frame of the flight (by default the dive off the spire, the gold stream winding
away below to the lake and the falls) and unless --no-title the title in the game's font over a shaded band at the
top, with the game's yellow splash text.

python thumbnail.py OUT.jpg [--no-title] [--t SECONDS] [--ss 3]
"""
import argparse

import numpy as np
from PIL import Image

import director as DR
import flight as FL
import pixelfont as PF

T_COVER = 1.25


def blit(img, spr, x, y):
    """Blit an RGBA sprite onto a float image, in place."""
    h, w = spr.shape[:2]
    H, W = img.shape[:2]
    x, y = int(round(x)), int(round(y))
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    s = spr[y0 - y:y1 - y, x0 - x:x1 - x]
    a = s[..., 3:4].astype(np.float32) / 255.0
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - a) + s[..., :3] * a


def splash(text, px, angle=-16.0):
    """The game's yellow splash text (as on the title screen), tilted."""
    spr = PF.render(text, px=px, color=(255, 255, 0), outline=0, shadow=True)
    return np.array(Image.fromarray(spr, 'RGBA').rotate(-angle, resample=Image.NEAREST, expand=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--no-title', action='store_true')
    ap.add_argument('--t', type=float, default=T_COVER)
    ap.add_argument('--ss', type=float, default=3.0)
    a = ap.parse_args()
    r, fl, meta = DR.setup(scale=1.0, ss=a.ss)
    img = DR.render_frame(r, fl, meta, int(round(a.t * FL.FPS))).astype(np.float32)
    H, W = img.shape[:2]
    if not a.no_title:
        lines = [('POV:', (255, 255, 255), None, 0.22),
                 ('FIRST ELYTRA FLIGHT', (255, 255, 255), None, 0.90),
                 ('IN THE NEW DIMENSION', None, ((255, 236, 130), (255, 150, 30)), 0.90)]
        y0 = 120
        band = np.zeros(H)
        band[y0 - 60:y0 + 330] = 1.0
        band = np.convolve(band, np.ones(120) / 120, 'same')
        img *= (1.0 - 0.45 * band)[:, None, None]
        for txt, col, grad, fill in lines:
            w1 = PF.render(txt, px=1, outline=1).shape[1]
            px = max(1, int(fill * W / w1))
            spr = PF.render(txt, px=px, color=col or (255, 255, 255), grad=grad, outline=1)
            x = (W - spr.shape[1]) / 2
            blit(img, spr, x, y0)
            if txt.startswith('IN THE'):
                sp = splash('THE SIFT!', 7)
                blit(img, sp, x + spr.shape[1] - sp.shape[1] * 0.55, y0 + spr.shape[0] - 6)
            y0 += spr.shape[0] + 22
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(a.out, quality=94)
    print('thumbnail:', a.out, flush=True)


if __name__ == '__main__':
    main()
