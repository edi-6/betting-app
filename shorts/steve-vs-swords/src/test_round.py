"""Contact sheet of a timeline round (low res) for review: python test_round.py OUT r4 f1,f2,... [cam_json]"""
import sys, json, time
import numpy as np
from PIL import Image
import scene, sim as S, timeline as T

out, key = sys.argv[1], sys.argv[2]
frames = [int(f) for f in sys.argv[3].split(',')]
cam = json.loads(sys.argv[4]) if len(sys.argv) > 4 else None
rd = {r['key']: r for r in T.round_defs()}[key]
sim = S.RoundSim(rd['formation'], **rd['sim'])
r, _ = scene.make_renderer(width=324, height=576, shadow_res=2048)
tiles = []
for f in range(max(frames) + 1):
    sim.step_frame()
    if f in frames:
        t = f / T.FPS
        if cam is None:
            shot = rd['shots'][max(k for k, s in enumerate(rd['shots']) if t >= s['start'] - 1e-9)]
            eye, tgt, fov = T.cam_path(shot['keys'], t)
            c = {'eye': eye, 'target': tgt, 'fov': fov}
        else:
            c = cam
        vox, sw = sim.instances()
        r.render(c, voxels=vox, swords=sw)
        tiles.append(r.finish())
        print(f, 'destroyed', sim.destroyed, 'debris', len(sim.dp), 'chunks', len(sim.chunks), 'vox inst', len(vox), flush=True)
cols = min(len(tiles), 6)
rows = (len(tiles) + cols - 1) // cols
h, w = tiles[0].shape[:2]
sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
for n, tl in enumerate(tiles):
    sheet[(n // cols) * h:(n // cols + 1) * h, (n % cols) * w:(n % cols + 1) * w] = tl
Image.fromarray(sheet).save(out)
