"""Dev helper: render a few fixed views of the scene at chosen moments, to check the world, the dominoes and
Herobrine without the director.

python stills.py OUT_DIR [--scale 0.5] [--views pov,giant,...]
"""
import argparse
import os
import time

import numpy as np
from PIL import Image

import dominoes as DM
import effects as FX
import herobrine as HB
import layout as LY
import picture as PIC
import props as PR
import scene as SC


def look(eye, target, fov=60.0):
    f = np.asarray(target, float) - np.asarray(eye, float)
    down = np.degrees(np.arcsin(np.clip(-f[2] / np.linalg.norm(f), -1, 1)))
    up = (0.0, 0.0, 1.0) if down < 80 else (0.0, -1.0, 0.0)
    return {'eye': tuple(eye), 'target': tuple(target), 'fov': fov, 'up': up}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--scale', type=float, default=0.5)
    ap.add_argument('--views', default='')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    L = LY.Layout(picture=PIC.picture)
    C = DM.load(L)
    D = PR.Dominoes(L, C, PIC.kinds(L.field['x'], L.field['y']))
    print(f'[stills] layout + chains {time.time() - t0:.1f}s', flush=True)
    W, H = int(1080 * a.scale), int(1920 * a.scale)
    r, _ = SC.make_renderer(L, width=W, height=H, ss=1.0, shadow_res=2048)
    fx = FX.Effects()
    r.set_ground_region(*FX.ground_patch(None))
    hb = HB.Herobrine()
    first = np.array([L.run['x'][0], L.run['y'][0]])
    fe = np.array(LY.FINAL_EYE)
    p = np.radians(LY.FINAL_PITCH)
    final_tgt = fe + np.array([0.0, -np.cos(p), np.sin(p)])
    views = {
        'pov': (-0.3, look((first[0], first[1] - 1.25, 1.62), (first[0], first[1] + 5.0, -0.35), 70.0),
                [((-11.0, LY.GIANT_Y + 1.0, 0.0), 0.0)]),
        'growth': (C.t_giant - 1.5, look((20.0, LY.GIANT_Y - 30.0, 6.0), (0.0, LY.GIANT_Y - 6.0, 11.0), 60.0), []),
        'giant': (C.t_land - 0.35, look((-24.0, LY.GIANT_Y - 4.0, 4.0), (0.0, LY.GIANT_Y + 14.0, 8.0), 64.0), []),
        'race': (C.t_race + 2.4, look((0.5, LY.RACE_Y0 + 8.0, 7.5), (0.0, LY.RACE_Y0 + 30.0, 0.0), 60.0),
                 [((-15.0, -66.0, 0.0), np.radians(-60.0))]),
        'gap': (C.t_stop + 1.0, look((2.7, 12.3, 1.5), (0.3, 60.0, 12.0), 50.0),
                [((0.0, LY.CLIFF_Y + 1.5, LY.CLIFF_Z), 0.0)]),
        'drone': (C.t_field0 + 6.0, look((0.0, 150.0, 70.0), (0.0, 60.0, 0.0), 62.0), []),
        'final': (C.t_end + 0.5, look(fe, final_tgt, LY.FINAL_FOV), []),
        'behind': (C.t_end + 2.0, look(fe, fe + np.array([0.0, 1.0, -0.05]), 64.0),
                   [((0.0, fe[1] + 2.2, LY.CLIFF_Z), 0.0)]),
    }
    names = a.views.split(',') if a.views else list(views)
    for name in names:
        t, cam, heros = views[name]
        props, P = D.instances(t)
        vox = [hb.instances(pos, yaw) for pos, yaw in heros]
        bits, puffs, flashes, lights = fx.render_data()
        vox = np.concatenate(vox) if vox else None
        near = np.array(cam['target'], float)
        bh = {'c': (0.0, 0.0, -9999.0), 'r': 0.0, 'sway': 0.0, 'sky_mul': (1, 1, 1), 'light_mul': (1, 1, 1)}
        ts = time.time()
        r.render(cam, voxels=vox, props=props, fx={'puffs': puffs, 'flashes': flashes,
                                                   'lights': np.zeros((0, 8), np.float32)},
                 near_center=near, glow=4.0, sculk=(0.0, 0.0), bh=bh)
        img = r.finish()
        Image.fromarray(img).save(os.path.join(a.out, f'{name}.png'))
        print(f'  {name} t={t:.2f} fallen {D.fallen(t)}  {time.time() - ts:.1f}s', flush=True)


if __name__ == '__main__':
    main()
