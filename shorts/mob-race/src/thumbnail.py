"""Render the 1080x1920 cover image: a frame of the recorded race re-rendered at a higher quality from a closer
camera (the pig dropping into the lava at the first trapdoor, the Warden just over it in a burst of gold sparkles),
with the race's header and standings strip and, unless --no-title, the title over the stone below.

python thumbnail.py OUT.jpg [--no-title] [--frame N] [--closer K | --cam X,Z,DIST] [--title-y Y]
"""
import argparse

import numpy as np
from PIL import Image

import course as C
import director as D
import effects as FX
import hud as HUDM
import main as M
import props as PR
import scene as SC
import timeline as T

FRAME = 320            # the first trapdoor has dropped: the Warden just made it over, the pig is going in
CAM = '-0.2,-20.0,19'  # x, z, distance


def title_block(img, k, lines, y=470):
    """Big stacked words with a dark band behind them, from y (in 1080-wide pixels) down."""
    out = img.astype(np.float32)
    sprites = [HUDM.text_sprite(txt, HUDM.font(900, int(size * k)), fill, stroke=int(10 * k),
                                gradient=grad) for (txt, size, fill, grad) in lines]
    heights = [s.shape[0] for s in sprites]
    y = int(y * k)
    top = y - int(40 * k)
    bot = y + sum(heights) + int(20 * k)
    band = np.zeros(out.shape[0])
    band[top:bot] = 1.0
    ramp = int(90 * k)
    band[top - ramp:top] = np.linspace(0, 1, ramp)
    band[bot:bot + ramp] = np.linspace(1, 0, ramp)
    out *= (1.0 - 0.45 * band)[:, None, None]
    for s in sprites:
        HUDM.over(out, s, (out.shape[1] - s.shape[1]) // 2, y)
        y += s.shape[0] - int(18 * k)
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--no-title', action='store_true')
    ap.add_argument('--frame', type=int, default=FRAME)
    ap.add_argument('--closer', type=float, default=0.85, help='camera distance factor')
    ap.add_argument('--cam', default=CAM, help='x,z,distance ("" for the video\'s own camera, moved --closer)')
    ap.add_argument('--title-y', type=int, default=1440)
    ap.add_argument('--width', type=int, default=1080)
    ap.add_argument('--ss', type=float, default=2.0)
    args = ap.parse_args()
    course = C.build()
    rec = M.record(course)
    alive = [m['name'] for m in rec[-1]['marbles'] if m['out'] is None]
    winner = alive[0] if len(alive) == 1 else None
    cams, _ = D.cameras(rec, course, winner)
    story = HUDM.story(rec, course, winner)
    fx = FX.Effects(course)
    for i in range(args.frame + 1):
        M.advance_fx(fx, rec[i], i, story, winner)
    cam = dict(cams[args.frame])
    if args.cam:
        x, z, d = (float(v) for v in args.cam.split(','))
        cam['target'] = (x, 0.0, z)
        cam['eye'] = (x, -d, z + 0.14 * d)
    else:
        tgt = np.array(cam['target'], float)
        eye = np.array(cam['eye'], float)
        cam['eye'] = tuple(tgt + (eye - tgt) * args.closer)
    W = args.width
    Hh = W * 16 // 9
    r = SC.make_renderer(course, width=W, height=Hh, ss=args.ss, shadow_res=4096)
    img = M.render_frame(r, course, rec[args.frame], cam, PR.Marbles(T.LINEUP), fx, False)
    hud = HUDM.HUD(W, Hh)
    s = HUDM.state_at(args.frame, story, rec)
    s['banner'] = None
    s['card'] = None
    img = hud.draw(img, s)
    if not args.no_title:
        k = W / 1080.0
        img = title_block(img, k, [('ONLY ONE', 150, HUDM.WHITE, None),
                                   ('SURVIVES', 170, HUDM.WHITE, ((255, 236, 90), (255, 120, 20)))], args.title_y)
    Image.fromarray(img).save(args.out, quality=93)
    print('thumbnail:', args.out)


if __name__ == '__main__':
    main()
