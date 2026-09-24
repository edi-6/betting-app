"""Render a clean 1080x1920 cover image: bedrock under the press at full pressure, sparks flying, the hoses
spraying oil and the alarm going, with the HUD and the question in the game's pixel font.

python thumbnail.py OUT.jpg [--no-title]
"""
import sys

import numpy as np
from PIL import Image

import main as M
import pixelfont as PF
import press as P
import scene as SC
import timeline as T
from hud import HUD, over

TAU = 3.35                     # seconds into the bedrock test: the press at ~90 % of what it can take
CAM = {'eye': (-3.6, -23.5, 6.6), 'target': (0.0, 0.0, 10.9), 'fov': 48.0}


def main(out, title=True, width=1080, ss=2.0):
    w = P.Press(T.cold_plan())
    while w.t < TAU - 1e-6:
        w.step_frame(1.0)
    r = SC.make_renderer(width=width, height=width * 16 // 9, ss=ss, shadow_res=4096)
    lights = M.alarm_lights(1, {'hoses': 0}, None)
    img = M.render_frame(r, w, CAM, False, lights)
    hud = HUD(width, width * 16 // 9)
    hs = {'boss': [dict(name='Bedrock', frac=1.0, col='pink'),
                   dict(name='Hydraulic Press', frac=0.08, col='red')],
          'xp': dict(tons=99999.0, danger=1.0),
          'hotbar': dict(sel=8.0, crossed={j: 9.0 for j in range(8)}),
          'vignette': 0.35, 'red': 0.12}
    img = hud.draw(img, hs).astype(np.float32)
    if title:
        k = width / 1080.0
        top, bot = int(330 * k), int(900 * k)
        ramp = np.clip(np.concatenate([np.linspace(0.0, 1.0, int(80 * k)), np.ones(bot - top - int(160 * k)),
                                       np.linspace(1.0, 0.0, int(80 * k))]), 0, 1)
        img[top:bot] *= (1.0 - 0.55 * ramp)[:, None, None]
        t1 = PF.render('CAN IT CRUSH', int(11 * k), (255, 255, 255), shadow=True, outline=1)
        t2 = PF.render('BEDROCK?', int(19 * k), (255, 170, 0), shadow=True, outline=1,
                       grad=((255, 236, 96), (255, 120, 0)))
        over(img, t1, (width - t1.shape[1]) // 2, int(410 * k))
        over(img, t2, (width - t2.shape[1]) // 2, int(410 * k) + t1.shape[0] + int(18 * k))
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(out, quality=94)
    print('saved', out)


if __name__ == '__main__':
    main(sys.argv[1], title='--no-title' not in sys.argv)
