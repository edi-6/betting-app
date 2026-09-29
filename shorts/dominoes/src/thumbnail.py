"""Render the 1080x1920 cover image: the field halfway through falling, seen from above and to the side. The shirt
and the face up to the nose are down, the eyes are still hidden in the white of the standing dominoes. With the
counter and, unless --no-title, a line of text in the game's font.

python thumbnail.py OUT.jpg [--no-title] [--row R] [--cam X,Y,Z,TX,TY]
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

ROW = 90                                   # the wave has just passed the nose; the eyes are next
CAM = '-17.0,26.0,40.0,-1.0,62.0'          # eye x, y, z, target x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--no-title', action='store_true')
    ap.add_argument('--row', type=int, default=ROW)
    ap.add_argument('--cam', default=CAM)
    ap.add_argument('--width', type=int, default=1080)
    ap.add_argument('--ss', type=float, default=2.0)
    args = ap.parse_args()
    show = M.Show()
    C = show.chains
    t = float(C.col_start[LY.NCOL // 2] + C.column.t0[args.row])
    x, y, z, tx, ty = (float(v) for v in args.cam.split(','))
    cam = {'eye': (x, y, z), 'target': (tx, ty, 0.0), 'fov': 42.0, 'up': (0.0, 0.0, 1.0)}
    Wd = args.width
    Hd = Wd * 16 // 9
    r, _ = SC.make_renderer(show.layout, width=Wd, height=Hd, ss=args.ss, shadow_res=4096)
    fx = FX.Effects()
    # the blast happened: its crater is there
    fx.crater = (M.CREEPER_X[1], M.CREEPER_Y, 1.7)
    r.set_ground_region(*FX.ground_patch(fx.crater))
    props, _ = show.dom.instances(t)
    r.water_time = t
    r.render(cam, voxels=None, props=props, fx={}, near_center=np.array([tx, ty, 0.0]), glow=0.0,
             sculk=(0.0, 0.0))
    img = r.finish()
    img = HUDM.HUD(Wd, Hd).draw(img, show.dom.fallen(t))
    if not args.no_title:
        k = Wd / 1080.0
        out = img.astype(np.float32)
        lines = [('WAIT FOR', (255, 255, 255), None), ('THE END...', None, ((255, 240, 120), (255, 150, 20)))]
        y0 = int(1330 * k)
        # a dark band behind the text
        band = np.zeros(Hd)
        band[y0 - int(40 * k):y0 + int(330 * k)] = 1.0
        band = np.convolve(band, np.ones(int(120 * k)) / int(120 * k), 'same')
        out *= (1.0 - 0.5 * band)[:, None, None]
        for txt, col, grad in lines:
            spr = PF.render(txt, px=int(17 * k), color=col or (255, 255, 255), grad=grad, outline=1)
            HUDM.over(out, spr, (Wd - spr.shape[1]) / 2, y0)
            y0 += spr.shape[0] + int(18 * k)
        img = np.clip(out, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(args.out, quality=93)
    print('thumbnail:', args.out, 'at t = %.2f, %d down' % (t, show.dom.fallen(t)))


if __name__ == '__main__':
    main()
