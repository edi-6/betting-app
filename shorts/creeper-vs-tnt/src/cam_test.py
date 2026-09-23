"""Render candidate cameras for one moment of a round into a labelled contact sheet.

python cam_test.py OUT.png ROUND_KEY SIM_TIME CAMS_JSON [WIDTH]
  CAMS_JSON: list of [eye, target, fov]
"""
import json
import sys

import numpy as np
from PIL import Image, ImageDraw

import scene
import sim as S
import timeline as T

out, key, t_sim = sys.argv[1], sys.argv[2], float(sys.argv[3])
cams = json.loads(sys.argv[4])
width = int(sys.argv[5]) if len(sys.argv) > 5 else 270
rd = {r['key']: r for r in T.round_defs()}[key]
sim = S.RoundSim(rd['formation'], **rd['sim'])
for _ in range(T.sec(t_sim)):
    sim.step_frame()
r, _ = scene.make_renderer(width=width, height=int(width * 16 / 9), shadow_res=2048)
r.set_ground_region(*sim.ground.mesh())
vox, tnt, fx = sim.instances()
tiles = []
for i, (eye, tgt, fov) in enumerate(cams):
    r.render({'eye': eye, 'target': tgt, 'fov': fov}, voxels=vox, tnt=tnt, fx=fx)
    im = Image.fromarray(r.finish())
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 22, 16), fill=(0, 0, 0))
    d.text((4, 2), str(i), fill=(255, 255, 0))
    tiles.append(np.array(im))
Image.fromarray(np.concatenate(tiles, 1)).save(out)
