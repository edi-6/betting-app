"""Scene setup: builds (and caches) the coast and the sky, and sets the renderer up for a late afternoon by the sea."""
import os
import time

import numpy as np

import blocks as BL
import renderer as RD
import sky as SKY
import world as WD
from noise import fbm2d
from vfx import puff_textures

CACHE = os.environ.get('TSUNAMI_CACHE', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cache'))


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


def world_data():
    def build():
        w = WD.build()
        ch = np.array([(c[0], c[1], *c[2], *c[3]) for c in w['chunks']], np.float64)
        return {'vertices': w['vertices'], 'indices': w['indices'], 'chunks': ch, 'H': w['H']}
    d = _cached('coast_v2.npz', build)
    d['chunk_list'] = [(int(c[0]), int(c[1]), c[2:5], c[5:8]) for c in d['chunks']]
    return d


def sky_data(width=8192, height=2048, coverage=0.40):
    def build():
        import clouds
        return {'sky': clouds.make_sky_with_clouds(width, height, coverage=coverage).astype(np.float16)}
    sun = '_'.join(f'{v:.2f}' for v in SKY.SUN_DIR)
    return _cached(f'sky_{width}x{height}_c{coverage:.2f}_s{sun}_v1.npz', build)['sky'].astype(np.float32)


def tint_noise(n=256):
    v = (np.arange(n) + 0.5) / n
    x, y = np.meshgrid(v, v)
    t = fbm2d(x * 4, y * 4, octaves=4, seed=77, period=4)
    t = (t - t.min()) / (t.max() - t.min())
    return t.astype(np.float32)


def make_renderer(width=1080, height=1920, ss=1.0, shadow_res=4096, sky_res=(8192, 2048)):
    RD.SUN_DIR = SKY.SUN_DIR
    wd = world_data()
    sky = sky_data(*sky_res)
    r = RD.Renderer(width, height, ss=ss, shadow_res=shadow_res, near_half=60.0, far_half=420.0)
    r.far_center = (0.0, -40.0, 10.0)
    r.near_origin = (0.0, -40.0)
    r.set_textures(BL.block_textures(), sky, tint_noise(), puff_textures())
    r.set_static(wd['vertices'], wd['indices'], wd['chunk_list'], ground_half=0.01)
    r.add_prop_kind('block', BL.debris_mesh(), BL.debris_layers(), sheen=-2.0)
    # late afternoon by the sea: a warm low sun, cool sky light
    r.sun_col = np.array([1.0, 0.82, 0.62]) * 3.1
    r.sky_amb = np.array([0.46, 0.56, 0.84]) * 0.95
    r.gnd_amb = np.array([0.40, 0.38, 0.28]) * 0.6
    r.bounce = (0.10, 0.09, 0.06)
    r.fog = 0.0011
    r.fog_col = (-1.0, 0.0, 0.0)
    r.exposure = 0.66
    r.sat = 1.08
    r.contrast = 1.06
    r.vignette = 0.18
    r.bloom_thresh = 2.4
    r.bloom = 0.30
    r.sculk_r = 0.0
    r.glow = 2.2
    r.ssao_radius = 0.7
    r.puff_tint = np.array([0.97, 0.98, 1.0])
    r.puff_k = (0.30, 0.78, 0.72, 0.12)         # sea spray: white, lit through by the sky as much as the sun
    return r, wd
