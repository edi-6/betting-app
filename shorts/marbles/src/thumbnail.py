"""The 1080x1920 cover: the machine from the front with the picture three quarters done (the sunset, the sun, a green
something standing in the meadow, its face not there yet) and marbles still raining in, titled 10,000 MARBLES /
GUESS THE PICTURE. --no-title for the same frame without the text.

python thumbnail.py OUT.jpg [--no-title] [--level 0.74]
"""
import argparse

import numpy as np
from PIL import Image

import hud as HD
import machine as M
import pixelfont as PF
import scene as SC
from main import H, W, Show, GLASS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--no-title', action='store_true')
    ap.add_argument('--level', type=float, default=0.74, help='how full the tank is (fraction of the marbles)')
    args = ap.parse_args()
    show = Show()
    m = show.marbles
    target = int(args.level * M.N)
    ts = float(np.sort(m.count_t)[target])
    r, _ = SC.make_renderer(width=W, height=H, ss=2.0, shadow_res=4096)
    r.near_half = 60.0
    cam = {'eye': (2.0, M.MY - 78.0, 47.0), 'target': (0.0, M.MY, 47.0), 'fov': 56.0}
    r.render(cam, marbles=m.instances(ts), glass=GLASS, near_center=(0.0, M.MY, 45.0))
    out = r.finish().astype(np.float32)
    if not args.no_title:
        band = np.zeros(H)
        band[30:400] = 1.0
        band[1540:1860] = 1.0
        band = np.convolve(band, np.ones(120) / 120, 'same')
        out *= (1.0 - 0.55 * band)[:, None, None]
        y = 100
        for txt, px, grad in (('10,000', 17, ((255, 240, 120), (255, 170, 20))), ('MARBLES', 13, None)):
            spr = PF.render(txt, px=px, color=(255, 255, 255), grad=grad, outline=1)
            HD.over(out, spr, (W - spr.shape[1]) / 2, y)
            y += spr.shape[0] + 14
        spr = PF.render('GUESS THE PICTURE', px=9, grad=((170, 255, 120), (60, 200, 40)), outline=1)
        HD.over(out, spr, (W - spr.shape[1]) / 2, 1640)
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(args.out, quality=93)
    print('thumbnail:', args.out, flush=True)


if __name__ == '__main__':
    main()
