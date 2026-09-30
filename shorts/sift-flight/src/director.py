"""Each frame of the flight: the rider's eye (flight.view), the valley's life and the fireworks (fx), rendered
supersampled, with motion blur from the camera's movement across the shutter, and - over the last moments - blended
into the first frame's animation so the video loops without a seam.
"""
import hashlib
import os
import time

import numpy as np

import flight as FL
import fx
import looks
import mobs
import props as PR
import scene as SC
import world_valley as WV

W, H = 1080, 1920
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, '..', 'cache')
STANDARD = {'size', 'origin', 'ids', 'state', 'hidden', 'tint', 'signs', 'frames'}
SOURCES = ('world_valley.py', 'flight.py', 'blocks.py', 'voxel.py', 'textures.py')
SHUTTER = 0.5            # of a frame (a 180-degree shutter)
LOOP_BLEND = (14.2, 15.0)


def cache_path():
    h = hashlib.md5()
    for f in SOURCES:
        with open(os.path.join(HERE, f), 'rb') as fh:
            h.update(fh.read())
    return os.path.join(CACHE, f'world_valley_{h.hexdigest()[:10]}.npz')


def load_world(r=None, fl=None):
    """Build (or load from the cache) the valley, upload it if a renderer is given, and return its metadata."""
    t = time.time()
    path = cache_path()
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        mesh, light, meta = d['mesh'].item(), d['light'], d['meta'].item()
    else:
        w = WV.build(fl)
        mesh, light = w.mesh(), w.light()
        meta = {k: v for k, v in w.__dict__.items() if k not in STANDARD}
        meta['origin'] = w.origin
        meta['size'] = w.size
        os.makedirs(CACHE, exist_ok=True)
        np.savez(path, mesh=np.array(mesh, dtype=object), light=light, meta=np.array(meta, dtype=object))
    if r is not None:
        r.add_world('valley', mesh, light, meta['origin'])
        n = sum(len(v[1]) // 3 for v in mesh.values())
        print(f'[director] valley: {n} triangles, {time.time() - t:.1f}s', flush=True)
    return meta


def setup(preview=False, ss=1.0, scale=None):
    """The renderer (the sky baked, the kinds registered, the valley uploaded), the flight and the valley's metadata."""
    k = scale or (0.5 if preview else 1.0)
    r = SC.make_renderer(int(W * k), int(H * k), ss=ss, skies=('sift_gold',), near_half=40.0, far_half=260.0)
    PR.register(r)
    mobs.register(r)
    fx.register(r)
    fl = FL.Flight()
    meta = load_world(r, fl)
    fx.setup(fl, meta)
    return r, fl, meta


def _render(r, fl, meta, t, t_anim, mblur=True):
    cam, eye, axes, v = FL.view(fl, t)
    env = looks.get()
    # the eye stops down when it looks into the low sun (a function of the view, so it is the same every time)
    toward = float(np.dot(axes[0], np.asarray(env['light_dir'], float)))
    env['exposure'] = float(env['exposure'] * (1.0 - 0.34 * FL.smooth(0.55, 0.97, toward)))
    sc = dict(t=t, ta=t_anim, cam=cam, eye=eye, axes=axes, v=v, fl=fl, meta=meta, env=env, r=r)
    rows, lights, parts, streaks = fx.events(sc)
    r.use_world('valley')
    r.time = t_anim
    inst = PR.rows_to_instances(rows)
    lt = np.array(lights[:16], np.float32) if lights else None
    mb = None
    if mblur:
        dt = SHUTTER / FL.FPS
        mb = dict(cam0=FL.view(fl, t - dt / 2)[0], cam1=FL.view(fl, t + dt / 2)[0], maxpx=64.0, taps=16)
    r.render(cam, env, instances=inst, lights=lt, particles=parts, streaks=streaks, clip_z=(-1e9, 1e9), mblur=mb,
             plant_dist=320.0)
    return r.finish(env).astype(np.float32)


def render_frame(r, fl, meta, f, mblur=True):
    """Frame f (at 60 fps) as (H, W, 3) uint8."""
    t = f / FL.FPS
    img = _render(r, fl, meta, t, t, mblur)
    a, b = LOOP_BLEND
    if t > a:
        # the last moments: the rider stands where they began, looking where they looked; everything that moves
        # with time (the ichor, the blubs, the drifting motes) is faded over to the first frame's
        u = FL.smooth(a, b, t)
        img = img * (1 - u) + _render(r, fl, meta, t, t - FL.DURATION, mblur) * u
    return np.clip(img + 0.5, 0, 255).astype(np.uint8)
