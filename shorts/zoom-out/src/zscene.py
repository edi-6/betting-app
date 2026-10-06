"""The 3D scene of the zoom: builds (and caches) the spawn area from the map's fields, sets the renderer up for a
clear day seen from straight above, and gives the camera for a view L metres wide centred on Steve's pupil.

The camera looks straight down with north up. L is the frame's width at the pupil (its height is 16/9 of that).
Close in, the lens is wide (50 degrees, so his head and the blocks round him look solid); as the view widens it
narrows, so that by the hand-over to the map (an orthographic satellite picture of this same world) it's nearly
orthographic and the two line up.
"""
import os
import time

import numpy as np

import renderer as RD
import spawnworld as SW
import steve as STV
import zblocks as ZB
import zoommap as ZM
from noise import fbm2d
from vfx import puff_textures

CACHE = os.environ.get('ZOOM_CACHE', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cache'))
W, H = 1080, 1920
ASPECT = W / H
STEVE = STV.Steve()
PUPIL_Z = float(STEVE.pupil[2])


def _cached(name, build):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name)
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        return {k: d[k] for k in d.files}
    t = time.time()
    data = build()
    np.savez(path, **data)
    print(f'[cache] built {name} in {time.time() - t:.1f}s', flush=True)
    return data


def _src_hash(*mods):
    import hashlib
    h = hashlib.sha1()
    for m in mods:
        with open(m.__file__, 'rb') as f:
            h.update(f.read())
    return h.hexdigest()[:10]


def fields(ctx=None):
    def build():
        m = ZM.MapRenderer(64, 64, ctx=ctx)
        return {'f': m.fields(SW.HALF)}
    return _cached('zoom_fields_%s.npz' % _src_hash(ZM), build)['f']


def world_data(ctx=None):
    def build():
        w, v, i, chunks = SW.build(fields(ctx))
        ch = np.array([(c[0], c[1], *c[2], *c[3]) for c in chunks], np.float64)
        return {'vertices': v, 'indices': i, 'chunks': ch, 'smoke': np.array(w.smoke, np.float64),
                'lights': np.array(w.lights, np.float64), 'Z': w.Z}
    d = _cached('zoom_world_%s.npz' % _src_hash(ZM, SW, ZB), build)
    d['chunk_list'] = [(int(c[0]), int(c[1]), c[2:5], c[5:8]) for c in d['chunks']]
    return d


def sky_gradient(w=512, h=256):
    """The sky texture the renderer wants (only ever seen in the water's reflection, looking straight up)."""
    from sky import EL_MIN
    v = (np.arange(h) + 0.5) / h
    el = 90.0 - v * (90.0 - EL_MIN)
    k = np.clip(el / 90.0, 0, 1) ** 0.42
    zen = np.array([0.10, 0.28, 0.78])
    hor = np.array([0.62, 0.78, 0.98])
    col = hor[None] * (1 - k[:, None]) + zen[None] * k[:, None]
    return np.repeat(col[:, None, :], w, 1).astype(np.float32) * 1.15


def tint_noise(n=256):
    v = (np.arange(n) + 0.5) / n
    x, y = np.meshgrid(v, v)
    t = fbm2d(x * 4, y * 4, octaves=4, seed=77, period=4)
    t = (t - t.min()) / (t.max() - t.min())
    return t.astype(np.float32)


def make_renderer(width=W, height=H, ss=1.0, shadow_res=4096):
    RD.SUN_DIR = ZM.SUN                                         # the same sun as the map's hillshading
    r = RD.Renderer(width, height, ss=ss, shadow_res=shadow_res, near_half=40.0, far_half=340.0)
    wd = world_data(r.ctx)
    r.far_center = (0.0, 0.0, 0.0)
    r.set_textures(ZB.world_textures(), sky_gradient(), tint_noise(), puff_textures())
    r.set_static(wd['vertices'], wd['indices'], list(wd['chunk_list']), ground_half=None)
    # a clear, bright day
    r.sun_col = np.array([1.0, 0.96, 0.88]) * 3.0
    r.sky_amb = np.array([0.46, 0.58, 0.86]) * 0.62
    r.gnd_amb = np.array([0.36, 0.40, 0.26]) * 0.45
    r.bounce = (0.05, 0.06, 0.035)
    r.fog = 0.0
    r.fog_col = (-1.0, 0.0, 0.0)
    r.exposure = 0.46
    r.sat = 0.82
    r.grade = 0.0
    r.contrast = 1.08
    r.vignette = 0.0
    r.bloom_thresh = 2.4
    r.bloom = 0.25
    r.sculk_r = 0.0
    r.glow = 1.6
    r.ssao_radius = 0.9
    r.puff_tint = np.array([0.92, 0.92, 0.94])
    return r, wd


# ---------------------------------------------------------------------------------------------
# the camera
# ---------------------------------------------------------------------------------------------
_FOV_L = np.log([1.5, 30.0, 150.0])
_FOV_V = np.array([50.0, 30.0, 4.0])


def fov_for(L):
    """Vertical field of view (degrees) for a view L metres wide."""
    return float(np.interp(np.log(max(L, 1e-9)), _FOV_L, _FOV_V))


def camera(L):
    """A camera straight above the pupil whose view is L metres wide at the pupil."""
    fov = fov_for(L)
    th = np.tan(np.radians(fov) / 2) * ASPECT                  # tan of half the horizontal fov
    hgt = (L / 2) / th
    eye = (0.0, 0.0, PUPIL_Z + hgt)
    return dict(eye=eye, target=(0.0, 0.0, PUPIL_Z), up=(0.0, 1.0, 0.0), fov=fov, near=max(0.02 * hgt, 1e-4),
                far=1.6 * hgt + 30.0), hgt


def setup_frame(r, L):
    """Per-frame renderer settings that follow the scale: shadows, ambient occlusion."""
    cam, hgt = camera(L)
    half = max(0.6, 1.08 * L)
    r.near_half = half
    ao = float(np.clip(0.045 * L, 0.02, 1.2))
    r.ssao_radius = ao / float(np.clip(hgt / 45.0, 0.6, 2.5))
    return cam, (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------------------------
# the satellite pictures the map shows round spawn
# ---------------------------------------------------------------------------------------------
SAT = ((160.0, 16.0), (320.0, 8.0))           # (half size m, pixels a metre): a sharp inner one, a wider outer one


def capture(r, half, ppm, tile=1024, margin=64):
    """An orthographic picture of the 3D world from straight above, [-half, half]^2, row 0 = north, rendered in
    overlapping tiles (r must be a square renderer of tile + 2 margin pixels)."""
    n = int(round(2 * half * ppm))
    nt = int(np.ceil(n / tile))
    out = np.zeros((nt * tile, nt * tile, 3), np.uint8)
    tm = tile / ppm
    full = (tile + 2 * margin) / ppm
    sv = (r.near_half, r.ssao_radius)
    r.near_half = full * 0.75 + 8.0
    r.ssao_radius = 1.2 / 2.5
    for ty in range(nt):
        for tx in range(nt):
            cx = -half + (tx + 0.5) * tm
            cy = half - (ty + 0.5) * tm
            cam = dict(eye=(cx, cy, 300.0), target=(cx, cy, 0.0), up=(0.0, 1.0, 0.0), fov=10.0, ortho=full / 2,
                       near=150.0, far=330.0)
            r.render(cam, near_center=(cx, cy, 0.0))
            img = r.finish()
            out[ty * tile:(ty + 1) * tile, tx * tile:(tx + 1) * tile] = img[margin:margin + tile, margin:margin + tile]
    r.near_half, r.ssao_radius = sv
    return out[:n, :n]


def satellites():
    """The two satellite pictures (cached): [(half, image), ...]."""
    def build():
        tile, margin = 1024, 64
        r, _ = make_renderer(tile + 2 * margin, tile + 2 * margin, ss=1.0)
        d = {}
        for k, (half, ppm) in enumerate(SAT):
            d['sat%d' % k] = capture(r, half, ppm, tile, margin)
        r.ctx.release()             # (its own GL context: done with before the main renderer is made)
        return d
    d = _cached('zoom_sat_%s.npz' % _src_hash(ZM, SW, ZB, RD, STV, __import__('zscene')), build)
    return [(half, d['sat%d' % k]) for k, (half, _) in enumerate(SAT)]
