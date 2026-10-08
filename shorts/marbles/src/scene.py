"""Scene setup: builds (and caches) the world and the sky, and configures the renderer for a sunny afternoon."""
import os
import time

import numpy as np

import blocks as BL
import machine as M
import renderer as RD
import sky as SKY
import world as WD
from noise import fbm2d
from vfx import puff_textures

CACHE = os.environ.get('MARBLES_CACHE', os.path.join(os.path.dirname(__file__), '..', 'cache'))


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
        w = WD.build_world()
        ch = np.array([(c[0], c[1], *c[2], *c[3]) for c in w['chunks']], np.float64)
        wv, wi = WD.water_mesh(w['heightmap'], w['river'])
        return {'vertices': w['vertices'], 'indices': w['indices'], 'chunks': ch, 'heightmap': w['heightmap'],
                'river': w['river'], 'trees': w['trees'], 'water_v': wv, 'water_i': wi}
    d = _cached('marbles_world_v3.npz', build)
    d['chunk_list'] = [(int(c[0]), int(c[1]), c[2:5], c[5:8]) for c in d['chunks']]
    return d


def sky_data(width=8192, height=2048, coverage=0.42):
    def build():
        import clouds
        return {'sky': clouds.make_sky_with_clouds(width, height, coverage=coverage).astype(np.float16)}
    sun = '_'.join(f'{v:.2f}' for v in SKY.SUN_DIR)
    return _cached(f'sky_day_{width}x{height}_c{coverage:.2f}_s{sun}_v1.npz', build)['sky'].astype(np.float32)


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
    r = RD.Renderer(width, height, ss=ss, shadow_res=shadow_res, near_half=34.0, far_half=330.0)
    r.far_center = (0.0, M.MY - 60.0, 30.0)
    r.near_origin = (M.MX, M.MY)
    r.set_textures(BL.world_textures(), sky, tint_noise(), puff_textures())
    v, i, chunks = wd['vertices'], wd['indices'], list(wd['chunk_list'])
    wv, wi = wd['water_v'], wd['water_i']
    chunks.append((len(i), len(wi), wv[:, :3].min(0), wv[:, :3].max(0)))
    v = np.concatenate([v, wv])
    i = np.concatenate([i, wi + len(wd['vertices'])]).astype(np.uint32)
    r.set_static(v, i, chunks, ground_half=0.01)
    # a bright, clear afternoon
    r.sun_col = np.array([1.0, 0.95, 0.86]) * 3.0
    r.sky_amb = np.array([0.46, 0.58, 0.86]) * 0.62
    r.gnd_amb = np.array([0.36, 0.40, 0.26]) * 0.45
    r.bounce = (0.05, 0.06, 0.035)
    r.fog = 0.0020
    r.fog_col = (-1.0, 0.0, 0.0)
    r.exposure = 0.62
    r.sat = 1.04
    r.contrast = 1.08
    r.vignette = 0.2
    r.bloom_thresh = 2.2
    r.bloom = 0.32
    r.sculk_r = 0.0
    r.glow = 1.6                       # the end rods
    r.ssao_radius = 0.6
    r.puff_tint = np.array([0.95, 0.93, 0.88])
    return r, wd
