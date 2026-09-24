"""Render a clean 1080x1920 cover image: Steve being dragged into the black hole, its lensed ring and glowing disk,
the ripped-up arena below, with the title.

python thumbnail.py OUT.png [--no-title]
"""
import sys

import numpy as np
from PIL import Image

import blackhole as BH
import main as M
import scene
import timeline as T
from hud import font, text_sprite, over, add, glow_sprite, PURPLE

T_SIM = 14.2
CAM = {'eye': (-36.0, -44.0, 28.0), 'target': (-9.0, -7.0, 22.0), 'fov': 60.0}


def main(out, cam=CAM, width=1080, ss=2.0, title=True):
    w = BH.World(T.plan())
    M.skip_to(w, T_SIM)
    vox, ranges, props, arrows, fx = w.instances()
    gl = [(a, b, 0.0, float(bd.white), float(bd.swell), bd.g.center, bd.pose()) for a, b, bd in ranges]
    bh = M.bh_params(w, w.t)
    r, _ = scene.make_renderer(width=width, height=width * 16 // 9, ss=ss, shadow_res=4096)
    r.sculk_r = 0.0
    r.set_ground_region(*w.ground.mesh())
    r.render(cam, voxels=vox, props=props, arrows=arrows, fx=fx, near_center=M.NEAR_CENTER, giants=gl, glow=5.0,
             sculk=(0.0, 0.0), bh=bh)
    img = r.finish().astype(np.float32)
    if title:
        k = width / 1080.0
        top = int(img.shape[0] * 0.34)
        img[:top] *= np.linspace(0.45, 1.0, top)[:, None, None]
        g = glow_sprite(int(1000 * k), int(520 * k), PURPLE, int(90 * k))
        add(img, g, (img.shape[1] - g.shape[1]) // 2, int(40 * k), gain=0.35)
        t1 = text_sprite('BLACK HOLE', font(900, int(150 * k)), stroke=int(9 * k), tracking=int(4 * k))
        t2 = text_sprite('SIZE 1,000', font(900, int(150 * k)), stroke=int(9 * k), tracking=int(4 * k),
                         gradient=((236, 214, 255), (170, 96, 255)))
        over(img, t1, (img.shape[1] - t1.shape[1]) // 2, int(130 * k))
        over(img, t2, (img.shape[1] - t2.shape[1]) // 2, int(130 * k) + t1.shape[0] - int(40 * k))
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(out)
    print('saved', out)


if __name__ == '__main__':
    main(sys.argv[1], title='--no-title' not in sys.argv)
