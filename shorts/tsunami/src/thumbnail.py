"""The 1080x1920 cover: the 1,000-block wall towering over the village street, the villagers and the iron golem
lined up in front of it, staring up; titled 1 vs 1,000 / BLOCK TSUNAMI. --no-title for the same frame without text.

python thumbnail.py OUT.jpg [--no-title]
"""
import argparse

import numpy as np
from PIL import Image

import edit as E
import hud as HD
import pixelfont as PF
import scene as SC
import wall as WL
from main import H, W, Show


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--no-title', action='store_true')
    ap.add_argument('--front', type=float, default=1150.0)
    args = ap.parse_args()
    r, wd = SC.make_renderer(width=W, height=H, ss=2.0, shadow_res=4096)
    show = Show(r, wd)
    t = WL.t_at(args.front)
    shot = E.Shot('cover', 1.0, 4, [(0, t), (1, t)],
                  [(0, (0.5, -35.6, 2.1), (0.3, 40.0, 38.0), 80), (1, (0.5, -35.6, 2.1), (0.3, 40.0, 38.0), 80)],
                  nc=(0.0, -26.0, 3.0), people=E._r4_crowd, hide=(1, 5, 6, 7), far=15000.0, near=0.3, fog=0.0007)
    img, _ = show.render_shot(shot, 0.0, 0.0)
    out = img.astype(np.float32)
    if not args.no_title:
        band = np.zeros(H)
        band[40:430] = 1.0
        band = np.convolve(band, np.ones(140) / 140, 'same')
        out *= (1.0 - 0.45 * band)[:, None, None]
        y = 110
        for txt, px, grad in (('1 vs 1,000', 19, ((255, 240, 120), (255, 160, 20))),
                              ('BLOCK TSUNAMI', 12, None)):
            spr = PF.render(txt, px=px, color=(255, 255, 255), grad=grad, outline=1)
            HD.over(out, spr, (W - spr.shape[1]) / 2, y)
            y += spr.shape[0] + 16
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(args.out, quality=93)
    print('thumbnail:', args.out, flush=True)


if __name__ == '__main__':
    main()
