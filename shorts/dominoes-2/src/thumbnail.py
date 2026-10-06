"""Render the 1080x1920 cover image: from the top of the cliff in the storm, the whole field down and the words
glowing blood red: LOOK BEHIND YOU (the O's are his eyes). With the counter at 100,000 and, unless --no-title, the
title in the game's font under it. --y FRONT_Y instead shows the field halfway down, the wave at FRONT_Y.

python thumbnail.py OUT.jpg [--no-title] [--y FRONT_Y] [--ss 2]
"""
import argparse

import numpy as np
from PIL import Image

import effects as FX
import hud as HUDM
import layout as LY
import main as M
import pixelfont as PF
import scene as SC

FRONT_Y = 168.0                            # where the wave is: past BEHIND, before YOU


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--no-title', action='store_true')
    ap.add_argument('--y', type=float, default=None)
    ap.add_argument('--ss', type=float, default=2.0)
    args = ap.parse_args()
    show = M.Show()
    C = show.chains
    if args.y is None:
        t = show.ev['done'] + 0.95
    else:
        row = int(round((args.y - LY.FIELD_Y0) / LY.SPACING))
        t = float(C.col_start[show.layout.ncol // 2] + C.column.t0[row])
    i = int(np.argmin(np.abs(show.times - t)))
    t = float(show.times[i])
    W, H = M.W, M.H
    r, _ = SC.make_renderer(show.layout, width=W, height=H, ss=args.ss, shadow_res=4096)
    fx = FX.Effects()
    s0 = show.ev['strikes'][0][1]
    r.set_ground_region(*FX.ground_patch(None, scorch=(s0 + 0.3, LY.FEEDER_Y - 0.3, 1.3)))
    img = M.render_frame(r, show, fx, i, t, False)
    ev = show.ev
    img = HUDM.HUD(W, H, total=LY.TOTAL).draw(img, show.dom.fallen(t), (t - ev['done']) if t >= ev['done'] else None)
    if not args.no_title:
        out = img.astype(np.float32)
        lines = [('100,000 DOMINOES', (255, 255, 255), None, 7),
                 ('PART 2', None, ((255, 240, 120), (255, 150, 20)), 17)]
        y0 = 1560
        band = np.zeros(H)
        band[y0 - 50:y0 + 330] = 1.0
        band = np.convolve(band, np.ones(120) / 120, 'same')
        out *= (1.0 - 0.62 * band)[:, None, None]
        for txt, col, grad, px in lines:
            spr = PF.render(txt, px=px, color=col or (255, 255, 255), grad=grad, outline=1)
            HUDM.over(out, spr, (W - spr.shape[1]) / 2, y0)
            y0 += spr.shape[0] + 26
        img = np.clip(out, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(args.out, quality=93)
    print('thumbnail:', args.out, 'at t = %.2f (frame %d), %d down' % (t, i, show.dom.fallen(t)), flush=True)


if __name__ == '__main__':
    main()
