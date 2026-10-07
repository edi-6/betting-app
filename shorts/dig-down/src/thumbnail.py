"""The 1080x1920 cover: a close-up of Steve raising his pickaxe over the last block, the lava right under it,
"I DUG STRAIGHT DOWN" over the top and, unless --no-title, "...BIG MISTAKE" on a dark plate over the lava.

python thumbnail.py OUT.jpg [--no-title] [--t SECONDS_AFTER_LAST] [--zoom Z]
"""
import argparse

import numpy as np
from PIL import Image

import hud as HD
import pixelfont as PF
from main import FPS, H, W, Show

FEET_PX = 900                   # where his feet go on the cover (the lava fills the frame under them)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--no-title', action='store_true')
    ap.add_argument('--t', type=float, default=0.75)          # the pickaxe raised over his head
    ap.add_argument('--zoom', type=float, default=1.7)
    args = ap.parse_args()
    show = Show()
    t = show.s.events['last'] + args.t
    _, fy = show.s.feet_at(t)
    cy = fy - (H / 2 - FEET_PX) / (96 * args.zoom)
    out = show.frame(int(round(t * FPS)), hud=False, cam=(0.5, cy, args.zoom)).astype(np.float32)
    if not args.no_title:
        band = np.zeros(H)
        band[40:420] = 1.0
        band = np.convolve(band, np.ones(140) / 140, 'same')
        out *= (1.0 - 0.6 * band)[:, None, None]
        y = 110
        for txt in ('I DUG', 'STRAIGHT DOWN'):
            spr = PF.render(txt, px=13, color=(255, 255, 255), outline=1)
            HD.over(out, spr, (W - spr.shape[1]) / 2, y)
            y += spr.shape[0] + 16
        # the game's translucent text plate, so the red reads over the lava
        spr = PF.render('...BIG MISTAKE', px=12, grad=HD.COLS['red'], outline=1)
        x0, y0 = int((W - spr.shape[1]) / 2), 1590
        out[y0 - 28:y0 + spr.shape[0] + 22, x0 - 34:x0 + spr.shape[1] + 34] *= 0.3
        HD.over(out, spr, x0, y0)
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(args.out, quality=93)
    print('thumbnail:', args.out, flush=True)


if __name__ == '__main__':
    main()
