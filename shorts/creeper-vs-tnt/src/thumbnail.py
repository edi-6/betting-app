"""Render a clean 1080x1920 thumbnail: the 10,000-TNT block about to hit the creeper, with a bold title.

python thumbnail.py OUT.png
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import scene
import sim as S
import timeline as T
from hud import FONT_DIR, _shadowed


def title_sprite(lines, size, fill=(255, 255, 255), stroke=(16, 16, 20)):
    font = ImageFont.truetype(os.path.join(FONT_DIR, 'Montserrat-900.ttf'), size)
    boxes = [font.getbbox(l, stroke_width=10) for l in lines]
    w = max(b[2] - b[0] for b in boxes) + 60
    lh = [b[3] - b[1] for b in boxes]
    h = sum(lh) + 30 * (len(lines) - 1) + 60
    img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    y = 30
    for l, b, hh in zip(lines, boxes, lh):
        x = (w - (b[2] - b[0])) // 2 - b[0]
        col = fill if l != lines[-1] else (255, 214, 64)
        d.text((x, y - b[1]), l, font=font, fill=col, stroke_width=10, stroke_fill=stroke)
        y += hh + 30
    spr, pad = _shadowed(np.array(img), blur=10, offset=(0, 8), strength=0.55, pad=24)
    return spr


def main(out):
    rd = {r['key']: r for r in T.round_defs()}['r5']
    sim = S.RoundSim(rd['formation'], **rd['sim'])
    n_frames = 34                       # 1.13 s: the block is closing in and no TNT is mid-flash
    for _ in range(n_frames):
        sim.step_frame()
    vox, tnt, fx = sim.instances()
    r, _ = scene.make_renderer(width=1080, height=1920, ss=2.0, shadow_res=4096)
    r.set_ground_region(*sim.ground.mesh())
    cam = {'eye': (40.0, -24.0, 12.0), 'target': (0.0, -10.0, 15.0), 'fov': 58.0}
    r.render(cam, voxels=vox, tnt=tnt, fx=fx)
    img = r.finish().astype(np.float32)
    # darken the top a touch so the title pops
    grad = np.linspace(0.62, 1.0, 560)[:, None, None]
    img[:560] *= grad
    title = title_sprite(['CREEPER vs', '10,000 TNT'], 124)
    from hud import _over
    _over(img, title, (1080 - title.shape[1]) // 2, 60)
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(out)
    print('saved', out)


if __name__ == '__main__':
    main(sys.argv[1])
