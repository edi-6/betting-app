"""Contact sheet of the edit: a few frames of every shot (the cold open first), small, with the HUD, to check the
cameras and the story. Usage: python check_cams.py OUT.png [width] [frames per shot]"""
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

import main as M
import press as P
import scene as SC
import timeline as T
from hud import HUD


def main():
    out = sys.argv[1]
    wd = int(sys.argv[2]) if len(sys.argv) > 2 else 270
    per = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    sched = T.schedule()
    shots = T.shots()
    recs = M.analyse(sched)
    st = M.story(recs)
    r = SC.make_renderer(width=wd, height=wd * 16 // 9, ss=1.0, shadow_res=2048)
    hud = HUD(wd, wd * 16 // 9)
    lens = {}
    for si, f, _ in sched:
        lens[si] = f + 1
    pick = set()
    start = {}
    for i, (si, f, _) in enumerate(sched):
        start.setdefault(si, i)
    for si, n in lens.items():
        for j in range(per):
            pick.add(start[si] + int(round((n - 1) * (j + 0.5) / per)))
    tiles = []
    t0 = time.time()
    w = P.Press(T.plan())
    for i, (si, f, scale) in enumerate(sched):
        M.advance(w, shots[si], f, scale)
        if i not in pick:
            continue
        eye, tgt, fov = T.cam_path(shots[si]['keys'], f / T.FPS)
        img = M.render_frame(r, w, {'eye': eye, 'target': tgt, 'fov': fov}, False,
                             M.alarm_lights(i, st, st['press_break']))
        hs = M.hud_state(i, st, recs)
        img = hud.draw(img, hs)
        im = Image.fromarray(img)
        ImageDraw.Draw(im).text((4, 4), f'{shots[si]["name"]} {i} t={w.t:.2f}', fill=(255, 255, 0))
        tiles.append(np.asarray(im))
    cols = min(per * 3, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    h, wd2 = tiles[0].shape[:2]
    sheet = np.zeros((rows * h, cols * wd2, 3), np.uint8)
    for k, t in enumerate(tiles):
        sheet[(k // cols) * h:(k // cols + 1) * h, (k % cols) * wd2:(k % cols + 1) * wd2] = t
    Image.fromarray(sheet).save(out)
    print('saved', out, f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
