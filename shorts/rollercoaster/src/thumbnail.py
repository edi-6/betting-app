"""Render the 1080x1920 cover: three bands cut on the diagonal, one per dimension, each a frame of the ride - the
drop into the canyon (the Overworld), the jump over the lava with the ghast's fireball coming (the Nether), the
dragon swooping over the cart (the End) - each labelled with its dimension, and unless --no-title, the title in the
game's font.

python thumbnail.py OUT.jpg [--no-title] [--frames F1,F2,F3] [--shift Y1,Y2,Y3] [--ss 2]
"""
import argparse

import numpy as np
from PIL import Image

import director as DR
import hud as HUD
import main as M
import pixelfont as PF
import ride as RD

FRAMES = '30,1180,640'          # video frames (at 60 fps) for the three bands; the End's is counted from its start
SHIFT = '420,-40,-300'          # how far down each frame is looked at through its band (pixels at 1080 wide)
CUTS = ((700, 590), (1330, 1220))   # the two diagonal cuts: (y at the left edge, y at the right edge)


def band_masks(W, H):
    x = np.arange(W)[None, :]
    y = np.arange(H)[:, None]
    k = W / 1080.0
    c1 = (CUTS[0][0] + (CUTS[0][1] - CUTS[0][0]) * x / W) * k
    c2 = (CUTS[1][0] + (CUTS[1][1] - CUTS[1][0]) * x / W) * k
    return [y < c1, (y >= c1) & (y < c2), y >= c2], (c1, c2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--no-title', action='store_true')
    ap.add_argument('--frames', default=FRAMES)
    ap.add_argument('--shift', default=SHIFT)
    ap.add_argument('--ss', type=float, default=2.0)
    ap.add_argument('--preview', action='store_true')
    a = ap.parse_args()
    r, ride = M.setup(a.preview, ss=a.ss)
    fr = [int(v) for v in a.frames.split(',')]
    fr[2] += int(round(ride.starts[2] * RD.FPS))
    shifts = [float(v) for v in a.shift.split(',')]
    imgs = [DR.render_frame(r, ride, f, hud=False) for f in fr]
    H, W = imgs[0].shape[:2]
    k = W / 1080.0
    masks, (c1, c2) = band_masks(W, H)
    out = np.zeros((H, W, 3), np.float32)
    for img, m, dy in zip(imgs, masks, shifts):
        d = int(round(dy * k))
        src = np.roll(img, -d, axis=0).astype(np.float32)
        out[m] = src[m]
    # the cuts: a black line with a thin bright core
    y = np.arange(H)[:, None]
    for c in (c1, c2):
        dist = np.abs(y - c)
        out *= np.clip(dist / (8 * k), 0, 1)[..., None]
        out = np.where((dist < 1.6 * k)[..., None], 235.0, out)
    # the labels, one per band, in its dimension's colour
    labels = (('OVERWORLD', ((140, 255, 110), (60, 170, 40)), 0.34),
              ('NETHER', ((255, 150, 80), (200, 40, 20)), 0.62),
              ('THE END', ((235, 170, 255), (150, 70, 220)), 0.90))
    for (txt, grad, yy) in labels:
        spr = PF.render(txt, px=int(round(7 * k)), grad=grad, outline=1)
        HUD.blit_f(out, spr, 40 * k, H * yy - spr.shape[0] / 2)
    if not a.no_title:
        lines = [('MINECRAFT ROLLERCOASTER', (255, 255, 255), None, 8),
                 ('ALL 3 DIMENSIONS', None, ((255, 244, 140), (255, 160, 20)), 13)]
        y0 = int(150 * k)
        band = np.zeros(H)
        band[y0 - int(50 * k):y0 + int(290 * k)] = 1.0
        band = np.convolve(band, np.ones(int(110 * k)) / int(110 * k), 'same')
        out *= (1.0 - 0.45 * band)[:, None, None]
        for txt, col, grad, px in lines:
            spr = PF.render(txt, px=int(round(px * k)), color=col or (255, 255, 255), grad=grad, outline=1)
            HUD.blit_f(out, spr, (W - spr.shape[1]) / 2, y0)
            y0 += spr.shape[0] + int(26 * k)
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(a.out, quality=93)
    print('thumbnail:', a.out, 'frames', fr, flush=True)


if __name__ == '__main__':
    main()
