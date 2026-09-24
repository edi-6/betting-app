"""Render a clean 1080x1920 cover image: the four giants side by side under everything the battle throws at them
(the volley of arrows in flight, the three blocks of anvils and the wall of TNT hanging in the sky), with the
question the video answers.

python thumbnail.py OUT.png [--no-title]
"""
import sys

import numpy as np
from PIL import Image

import arena as A
import main as M
import scene
import timeline as T
from hud import font, text_sprite, over, GOLD

CAM = {'eye': (-4.0, -112.0, 26.0), 'target': (-4.0, 0.0, 40.0), 'fov': 60.0}


def main(out, cam=CAM, width=1080, ss=2.0, title=True):
    sim = A.Arena(T.plan())
    volley = A.Arena(T.plan())
    while volley.t < 2.35:                       # the arrows halfway to their targets
        volley.step_frame(1.0)
    sim.n_appear[:] = -1.0                       # everything hanging in the sky at once
    sim.t_appear[:] = -1.0
    sim.t = 0.5
    vox, ranges, props, _, fx = sim.instances()
    arrows = volley.arrow_instances()
    gl = [(a, b, 0.0, 0.0, 1.0, sim.giants[gi].center) for gi, (a, b) in enumerate(ranges)]
    r, _ = scene.make_renderer(width=width, height=width * 16 // 9, ss=ss, shadow_res=4096)
    r.set_ground_region(*sim.ground.mesh())
    r.render(cam, voxels=vox, props=props, arrows=arrows, fx=fx, near_center=M.NEAR_CENTER, giants=gl, glow=6.0,
             sculk=M.SCULK)
    img = r.finish().astype(np.float32)
    if title:
        k = width / 1080.0
        top = int(img.shape[0] * 0.3)
        img[:top] *= np.linspace(0.6, 1.0, top)[:, None, None]
        t1 = text_sprite('WHO', font(900, int(150 * k)), stroke=int(9 * k), tracking=int(4 * k))
        t2 = text_sprite('SURVIVES?', font(900, int(150 * k)), fill=GOLD, stroke=int(9 * k), tracking=int(4 * k),
                         gradient=((255, 236, 140), (240, 150, 20)))
        over(img, t1, (img.shape[1] - t1.shape[1]) // 2, int(120 * k))
        over(img, t2, (img.shape[1] - t2.shape[1]) // 2, int(120 * k) + t1.shape[0] - int(40 * k))
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(out)
    print('saved', out)


if __name__ == '__main__':
    main(sys.argv[1], title='--no-title' not in sys.argv)
