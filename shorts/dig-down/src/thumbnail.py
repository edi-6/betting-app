"""The 1080x1920 cover: Steve on his last block with five diamonds, the lava right under him, "I DUG STRAIGHT DOWN"
over the top and, unless --no-title, "...BIG MISTAKE" under it.

python thumbnail.py OUT.jpg [--no-title] [--t SECONDS_AFTER_LAST]
"""
import argparse

import numpy as np
from PIL import Image

import hud as HD
import pixelfont as PF
from main import FPS, H, W, Show


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--no-title', action='store_true')
    ap.add_argument('--t', type=float, default=0.7)
    args = ap.parse_args()
    show = Show()
    t = show.s.events['last'] + args.t
    out = show.frame(int(round(t * FPS)), hud=False).astype(np.float32)
    if not args.no_title:
        band = np.zeros(H)
        band[40:420] = 1.0
        band[1420:1800] = 1.0
        band = np.convolve(band, np.ones(140) / 140, 'same')
        out *= (1.0 - 0.6 * band)[:, None, None]
        y = 110
        for txt in ('I DUG', 'STRAIGHT DOWN'):
            spr = PF.render(txt, px=13, color=(255, 255, 255), outline=1)
            HD.over(out, spr, (W - spr.shape[1]) / 2, y)
            y += spr.shape[0] + 16
        spr = PF.render('...BIG MISTAKE', px=12, grad=HD.COLS['red'], outline=1)
        HD.over(out, spr, (W - spr.shape[1]) / 2, 1520)
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(args.out, quality=93)
    print('thumbnail:', args.out, flush=True)


if __name__ == '__main__':
    main()
