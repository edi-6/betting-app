"""Contact sheet of chosen moments of a round (or the hook), rendered through the real pipeline: pre-roll,
slow-motion schedule, shot list, arrow-cam and damage overlay. For look development.

python test_round.py OUT.png KEY TIMES [WIDTH] [CAM_JSON]
  KEY: r1..r5 or hook;  TIMES: comma separated video times in seconds (for the hook: time within the hook)
  CAM_JSON: optional {"eye": [..], "target": [..], "fov": ..} that overrides the shot cameras
"""
import json
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

import main as M
import scene
import sim as S
import timeline as T

out, key = sys.argv[1], sys.argv[2]
times = [float(t) for t in sys.argv[3].split(',')]
width = int(sys.argv[4]) if len(sys.argv) > 4 else 270
cam_override = json.loads(sys.argv[5]) if len(sys.argv) > 5 else None
want = {T.sec(t) for t in times}
r, _ = scene.make_renderer(width=width, height=int(width * 16 / 9), shadow_res=2048)
tiles = []
t0 = time.time()


def shoot(sim, eye, tgt, fov, label, hurt=0.0, follow=None):
    cam = cam_override or {'eye': eye, 'target': tgt, 'fov': fov}
    im = Image.fromarray(M.render_frame(r, cam, sim, False, hurt=hurt, follow=follow))
    fx = sim.instances()[2]
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 60, 14), fill=(0, 0, 0))
    d.text((3, 1), label, fill=(255, 255, 0))
    tiles.append(np.array(im))
    st = np.bincount(sim.state, minlength=7)
    print(f'{label}: sim t {sim.t:.2f} destroyed {sim.destroyed} arrows fly {st[S.FLY]} stuck {st[S.STUCK]} '
          f'ground {st[S.GROUND]} debris {len(sim.dp)} puffs {len(fx["puffs"])}', flush=True)


if key == 'hook':
    g = 0
    for hs in T.hook_shots():
        rd = {x['key']: x for x in T.round_defs()}[hs['round']]
        sim = S.RoundSim(rd['formation'], **rd['sim'])
        for _ in range(T.sec(hs['sim_start'])):
            sim.step_frame(1.0)
        nf = T.sec(hs['dur'])
        scales = T.frame_scales(hs.get('scale'), nf)
        follow = M.FollowCam(sim, hs['follow']) if 'follow' in hs else None
        for f in range(nf):
            sim.step_frame(scales[f])
            if follow is not None:
                eye, tgt, fov, _ = follow(f / T.FPS)
            else:
                eye, tgt, fov = T.cam_path(hs['keys'], f / T.FPS)
            if g in want:
                shoot(sim, eye, tgt, fov, f'{g / T.FPS:.2f}', follow=follow)
            g += 1
else:
    rd = {x['key']: x for x in T.round_defs()}[key]
    a = M.analyse_round(rd)
    hp = M.hp_curve(rd, a)
    sim = M.make_sim(rd)
    follows = {}
    flash_start = -100
    for f in range(max(want) + 1):
        sim.step_frame(a['scales'][f])
        t = f / T.FPS
        si = max(k for k, s in enumerate(rd['shots']) if t >= s['start'] - 1e-9)
        shot = rd['shots'][si]
        follow = None
        if 'follow' in shot:
            if si not in follows:
                follows[si] = M.FollowCam(sim, shot['follow'])
            follow = follows[si]
            eye, tgt, fov, _ = follow(t - shot['start'])
        else:
            eye, tgt, fov = T.cam_path(shot['keys'], t)
        if f > 0 and hp[f] < hp[f - 1] and f - flash_start >= M.HURT_GAP:
            flash_start = f
        if f in want:
            shoot(sim, eye, tgt, fov, f'{t:.2f} hp{hp[f]}', M.hurt_amount(f - flash_start), follow)
print(f'{time.time() - t0:.0f}s')
cols = min(len(tiles), 6)
rows = (len(tiles) + cols - 1) // cols
h, w = tiles[0].shape[:2]
sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
for n, tl in enumerate(tiles):
    sheet[(n // cols) * h:(n // cols + 1) * h, (n % cols) * w:(n % cols + 1) * w] = tl
Image.fromarray(sheet).save(out)
