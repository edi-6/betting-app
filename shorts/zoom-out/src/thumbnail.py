"""Render the 1080x1920 cover image: Steve's eye close up, the whole world in its pupil, with the title over it
and, unless --no-title, "THE WHOLE WORLD = 1 PIXEL" under the pupil.

python thumbnail.py OUT.jpg [--no-title] [--L3 0.16] [--ss 2]
"""
import argparse

import numpy as np
from PIL import Image

import hud as HUDM
import pixelfont as PF
import zhud as ZH
import ztimeline as TL
from main import H, W, Zoom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--no-title', action='store_true')
    ap.add_argument('--L3', type=float, default=0.3, help='view width at the eye (m): the pupil is 1/16 m')
    ap.add_argument('--ss', type=float, default=2.0)
    args = ap.parse_args()
    z = Zoom(ss=args.ss)
    L3 = args.L3
    img = z.frame_3d(L3, 0.0)
    img = img * z.vig
    out = np.clip(img, 0, 255).astype(np.float32)
    # the pupil's outline
    s = TL.PX / L3 * W
    x0, y0 = int(round(W / 2 - s / 2)), int(round(H / 2 - s / 2))
    x1, y1 = int(round(W / 2 + s / 2)), int(round(H / 2 + s / 2))
    b = 5
    for (ya, yb, xa, xb) in ((y0 - b, y0, x0 - b, x1 + b), (y1, y1 + b, x0 - b, x1 + b), (y0, y1, x0 - b, x0),
                             (y0, y1, x1, x1 + b)):
        out[max(ya, 0):yb, max(xa, 0):xb] = 255.0
    if not args.no_title:
        band = np.zeros(H)
        band[60:420] = 1.0
        band[1430:1840] = 1.0
        band = np.convolve(band, np.ones(120) / 120, 'same')
        out *= (1.0 - 0.55 * band)[:, None, None]
        y = 110
        for txt in ('HOW BIG IS A', 'MINECRAFT WORLD?'):
            spr = PF.render(txt, px=11, color=(255, 255, 255), outline=1)
            HUDM.over(out, spr, (W - spr.shape[1]) / 2, y)
            y += spr.shape[0] + 14
        y = 1480
        for txt, px, grad in (('THE WHOLE WORLD', 10, ZH.GOLD), ('= 1 PIXEL', 14, ZH.GOLD)):
            spr = PF.render(txt, px=px, grad=grad, outline=1)
            HUDM.over(out, spr, (W - spr.shape[1]) / 2, y)
            y += spr.shape[0] + 18
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(args.out, quality=93)
    print('thumbnail:', args.out, flush=True)


if __name__ == '__main__':
    main()
