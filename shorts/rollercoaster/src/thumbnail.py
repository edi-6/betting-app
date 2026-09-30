"""Render the 1080x1920 cover: four bands cut on the diagonal, one per dimension, each a frame of the ride - the drop
into the canyon (the Overworld), the jump over the lava with the ghast's fireball coming (the Nether), the dragon
diving at the cart (the End), the pink meadow under the turquoise sky (the Sift, with the game's yellow splash text:
NEW DIMENSION!) - each labelled with its dimension, and unless --no-title, the title in the game's font.

python thumbnail.py OUT.jpg [--no-title] [--frames A45,B1179,C206,F54] [--shift Y1,Y2,Y3,Y4] [--ss 2]
(a frame is a segment's letter and a frame counted from that segment's start)
"""
import argparse

import numpy as np
from PIL import Image

import director as DR
import hud as HUD
import main as M
import pixelfont as PF
import ride as RD

FRAMES = 'A45,B539,C206,F54'
SHIFT = '560,20,-1040,-980'    # how far down each frame is looked at through its band (pixels at 1080 wide)
CUTS = ((720, 640), (1130, 1050), (1540, 1460))   # the diagonal cuts: (y at the left edge, y at the right edge)
LABELS = (('OVERWORLD', ((140, 255, 110), (60, 170, 40))), ('NETHER', ((255, 150, 80), (200, 40, 20))),
          ('THE END', ((235, 170, 255), (150, 70, 220))), ('THE SIFT', ((150, 250, 240), (250, 130, 180))))


def cut_lines(W):
    x = np.arange(W)[None, :]
    k = W / 1080.0
    return [(a + (b - a) * x / W) * k for (a, b) in CUTS]


def splash(text, px, angle=-18.0):
    """The game's yellow splash text (as on the title screen), tilted."""
    spr = PF.render(text, px=px, color=(255, 255, 0), outline=0, shadow=True)
    im = Image.fromarray(spr, 'RGBA').rotate(-angle, resample=Image.NEAREST, expand=True)
    return np.array(im)


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
    names = [sg.name for sg in ride.segments]
    fr = []
    for spec in a.frames.split(','):
        i = names.index(spec[0])
        fr.append(int(round(ride.starts[i] * RD.FPS)) + int(spec[1:]))
    shifts = [float(v) for v in a.shift.split(',')]
    imgs = [DR.render_frame(r, ride, f, hud=False) for f in fr]
    H, W = imgs[0].shape[:2]
    k = W / 1080.0
    cuts = cut_lines(W)
    y = np.arange(H)[:, None]
    bounds = [np.zeros((1, W))] + cuts + [np.full((1, W), float(H))]
    out = np.zeros((H, W, 3), np.float32)
    for img, dy, lo, hi in zip(imgs, shifts, bounds[:-1], bounds[1:]):
        m = (y >= lo) & (y < hi)
        src = np.roll(img, -int(round(dy * k)), axis=0).astype(np.float32)
        out[m] = src[m]
    # the cuts: a black line with a thin bright core
    for c in cuts:
        dist = np.abs(y - c)
        out *= np.clip(dist / (8 * k), 0, 1)[..., None]
        out = np.where((dist < 1.6 * k)[..., None], 235.0, out)
    # the labels, one per band, in its dimension's colour: bottom left, just above the cut below it
    x_l = 36 * k
    for (txt, grad), hi in zip(LABELS, bounds[1:]):
        yb = float(hi[0, int(x_l)]) if hi.shape[1] > 1 else H - 60 * k
        spr = PF.render(txt, px=int(round(6 * k)), grad=grad, outline=1)
        HUD.blit_f(out, spr, x_l, yb - spr.shape[0] - 22 * k)
        if txt == 'THE SIFT':
            sp = splash('NEW DIMENSION!', int(round(7 * k)))
            HUD.blit_f(out, sp, x_l + spr.shape[1] + 18 * k, yb - spr.shape[0] - 22 * k - sp.shape[0] * 0.55)
    if not a.no_title:
        lines = [('MINECRAFT ROLLERCOASTER', (255, 255, 255), None, 0.80),
                 ('ALL 4 DIMENSIONS', None, ((255, 244, 140), (255, 160, 20)), 0.90)]
        y0 = int(110 * k)
        band = np.zeros(H)
        band[y0 - int(50 * k):y0 + int(250 * k)] = 1.0
        band = np.convolve(band, np.ones(int(110 * k)) / int(110 * k), 'same')
        out *= (1.0 - 0.45 * band)[:, None, None]
        for txt, col, grad, fill in lines:
            # the biggest whole pixel size that fits the line in `fill` of the width
            w1 = PF.render(txt, px=1, outline=1).shape[1]
            px = max(1, int(fill * W / w1))
            spr = PF.render(txt, px=px, color=col or (255, 255, 255), grad=grad, outline=1)
            HUD.blit_f(out, spr, (W - spr.shape[1]) / 2, y0)
            y0 += spr.shape[0] + int(24 * k)
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(a.out, quality=93)
    print('thumbnail:', a.out, 'frames', fr, flush=True)


if __name__ == '__main__':
    main()
