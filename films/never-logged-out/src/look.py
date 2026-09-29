"""Dev viewer: render views of a world variant.

python look.py OUT_PREFIX variant preset --cam=ex,ey,ez,tx,ty,tz,fov [--cam=...] [--w 1280 --h 720]
"""
import argparse
import time

import numpy as np
from PIL import Image

import looks
import scene as SC
import worldgen as WG
import decals as DC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('variant')
    ap.add_argument('preset')
    ap.add_argument('--cam', action='append', dest='cams')
    ap.add_argument('--w', type=int, default=1280)
    ap.add_argument('--h', type=int, default=720)
    a = ap.parse_args()
    t = time.time()
    w = WG.build(a.variant)
    print(f'[look] built {time.time() - t:.1f}s', flush=True)
    e = looks.get(a.preset)
    r = SC.make_renderer(a.w, a.h, skies=(e['sky'],))
    SC.register_world(r, a.variant, w, cache_name=a.variant + '_dev')
    r.use_world(a.variant)
    dec = DC.Decals(r, w)
    for i, c in enumerate(a.cams):
        v = [float(x) for x in c.split(',')]
        cam = dict(eye=v[0:3], target=v[3:6], fov=v[6] if len(v) > 6 else 70.0)
        t = time.time()
        r.render(cam, e, instances=dec.instances())
        img = r.finish(e)
        Image.fromarray(img).save(f'{a.out}_{i}.png')
        print(f'[look] view {i}: {time.time() - t:.2f}s', flush=True)


if __name__ == '__main__':
    main()
