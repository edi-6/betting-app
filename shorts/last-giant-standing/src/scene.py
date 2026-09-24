"""Scene setup shared by previews and the final render: builds/caches assets and configures the renderer."""
import os
import time
import numpy as np

import world as W
import sky as SKY
from noise import fbm2d
from textures import make_block_textures
from renderer import Renderer
import anvil as AN
import arrows as AR
import tnt as TN
from ground import REG
from vfx import puff_textures

CLEARING = 52.0          # the arena: a wide clearing in the forest for the four giants
CACHE = os.environ.get('SVS_CACHE', os.path.join(os.path.dirname(__file__), '..', 'cache'))


def _cached(name, build):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name)
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        return {k: d[k] for k in d.files}
    t = time.time()
    data = build()
    np.savez(path, **data)
    print(f'[cache] built {name} in {time.time() - t:.1f}s')
    return data


def world_data():
    def build():
        w = W.build_world(clearing=CLEARING)
        ch = np.array([(c[0], c[1], *c[2], *c[3]) for c in w['chunks']], np.float64)
        return {'vertices': w['vertices'], 'indices': w['indices'], 'chunks': ch,
                'heightmap': w['heightmap'], 'trees': w['trees']}
    d = _cached(f'world_arena_c{CLEARING:.0f}_v1.npz', build)
    d['chunk_list'] = [(int(c[0]), int(c[1]), c[2:5], c[5:8]) for c in d['chunks']]
    return d


def sky_data(width=8192, height=2048, coverage=0.37):
    def build():
        import clouds
        return {'sky': clouds.make_sky_with_clouds(width, height, coverage=coverage).astype(np.float16)}
    sun = '_'.join(f'{v:.2f}' for v in SKY.SUN_DIR)
    return _cached(f'sky_{width}x{height}_c{coverage:.2f}_s{sun}_golden_v1.npz', build)['sky'].astype(np.float32)


def tint_noise(n=256):
    v = (np.arange(n) + 0.5) / n
    x, y = np.meshgrid(v, v)
    t = fbm2d(x * 4, y * 4, octaves=4, seed=77, period=4)
    t = (t - t.min()) / (t.max() - t.min())
    return t.astype(np.float32)


def make_renderer(width=1080, height=1920, ss=1.0, shadow_res=4096, sky_res=(8192, 2048)):
    wd = world_data()
    sky = sky_data(*sky_res)
    r = Renderer(width, height, ss=ss, shadow_res=shadow_res)
    bt = make_block_textures()
    r.set_textures([bt[n] for n in W.LAYERS], sky, tint_noise(), puff_textures())
    r.set_static(wd['vertices'], wd['indices'], wd['chunk_list'], hole=REG)
    r.add_prop_kind('anvil', AN.build_mesh(), AN.textures(), sheen=0.22)
    r.add_prop_kind('tnt', TN.build_mesh(), TN.texture_layers(), sheen=-1.0)
    r.set_arrow_textures(AR.texture_layers())
    r.set_arrow_mesh(AR.build_mesh())
    return r, wd
