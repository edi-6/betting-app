"""Render a clean 1080x1920 thumbnail: the first wave of the 10,000 arrows a split second before it hits the
zombie, with a bold title.

python thumbnail.py OUT.png
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import main as M
import scene
import timeline as T
from hud import FONT_DIR, _over, _shadowed

VIDEO_T = 2.85                     # slow motion, the wall of arrows about to land
CAM = {'eye': (30.0, -16.0, 16.0), 'target': (0.0, -6.0, 25.0), 'fov': 60.0}


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


def main(out, cam=CAM, video_t=VIDEO_T, width=1080, ss=2.0, title=True):
    rd = {r['key']: r for r in T.round_defs()}['r5']
    sim = M.make_sim(rd)
    n = T.sec(video_t)
    scales = T.frame_scales(rd.get('scale'), n)
    for f in range(n):
        sim.step_frame(scales[f])
    r, _ = scene.make_renderer(width=width, height=width * 16 // 9, ss=ss, shadow_res=4096)
    img = M.render_frame(r, cam, sim, False).astype(np.float32)
    if title:
        # darken the top a touch so the title pops
        top = int(img.shape[0] * 0.29)
        img[:top] *= np.linspace(0.62, 1.0, top)[:, None, None]
        spr = title_sprite(['ZOMBIE vs', '10,000 ARROWS'], 112 * width // 1080)
        _over(img, spr, (img.shape[1] - spr.shape[1]) // 2, 60 * width // 1080)
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(out)
    print('saved', out)


if __name__ == '__main__':
    main(sys.argv[1])
