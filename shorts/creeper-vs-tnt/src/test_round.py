"""Contact sheet of one round for look development.

python test_round.py OUT.png FORMATION_JSON FRAMES [CAM_JSON] [WIDTH]
  FORMATION_JSON: kwargs for sim.block_formation, optionally {"sim": {...}} for RoundSim kwargs
  FRAMES: comma separated frame numbers
"""
import json
import sys
import time

import numpy as np
from PIL import Image

import scene
import sim as S

out = sys.argv[1]
cfg = json.loads(sys.argv[2])
frames = [int(f) for f in sys.argv[3].split(',')]
cam = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {'eye': [22.0, -26.0, 16.0], 'target': [0.0, -3.0, 15.0], 'fov': 58.0}
width = int(sys.argv[5]) if len(sys.argv) > 5 else 324
sim_kw = cfg.pop('sim', {})
form = S.block_formation(**cfg)
sim = S.RoundSim(form, **sim_kw)
r, _ = scene.make_renderer(width=width, height=int(width * 16 / 9), shadow_res=2048)
tiles = []
t_sim = t_ren = 0.0
for f in range(max(frames) + 1):
    t0 = time.time()
    sim.step_frame()
    t_sim += time.time() - t0
    if f in frames:
        t0 = time.time()
        if sim.ground.dirty:
            r.set_ground_region(*sim.ground.mesh())
        vox, tnt, fx = sim.instances()
        r.render(cam, voxels=vox, tnt=tnt, fx=fx)
        tiles.append(r.finish())
        t_ren += time.time() - t0
        print(f'frame {f}: destroyed {sim.destroyed} debris {len(sim.dp)} tnt {len(tnt)} puffs {len(fx["puffs"])} '
              f'flashes {len(fx["flashes"])} lights {len(fx["lights"])}', flush=True)
print(f'sim {t_sim:.1f}s render {t_ren:.1f}s')
cols = min(len(tiles), 6)
rows = (len(tiles) + cols - 1) // cols
h, w = tiles[0].shape[:2]
sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
for n, tl in enumerate(tiles):
    sheet[(n // cols) * h:(n // cols + 1) * h, (n % cols) * w:(n % cols + 1) * w] = tl
Image.fromarray(sheet).save(out)
