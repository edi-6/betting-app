"""Shared setup: the renderer with the block atlas, noise, moon and sky panoramas loaded, and worlds registered."""
import os
import time

import numpy as np

import gfx
import sky as SKY
import voxel as VX
from noise import fbm2d

CACHE = SKY.CACHE


def noise_tex(n=256, seed=3):
    path = os.path.join(CACHE, f'noise_{n}_{seed}.npy')
    if os.path.exists(path):
        return np.load(path)
    v = (np.arange(n) + 0.5) / n
    x, y = np.meshgrid(v, v)
    f = fbm2d(x * 8, y * 8, octaves=4, seed=seed, period=8)
    f = (f - f.min()) / (f.max() - f.min())
    os.makedirs(CACHE, exist_ok=True)
    np.save(path, f.astype(np.float32))
    return f.astype(np.float32)


def make_renderer(width=1920, height=1080, ss=1.0, skies=('sunset',), shadow_res=4096, near_half=40.0,
                  far_half=220.0):
    t = time.time()
    # the skies first: baking one uses (and releases) a GL context of its own, which must not happen once the
    # renderer's context is live (its later draws would go nowhere)
    panos = {s: SKY.panorama(s).astype(np.float32) for s in skies}
    r = gfx.Renderer(width, height, ss=ss, shadow_res=shadow_res, near_half=near_half, far_half=far_half)
    at = VX.atlas()
    li = {k: at[k] for k in ('water', 'grass_top', 'grass_side', 'lava', 'ichor')}
    # what sways in the wind (and is lit as foliage): here the Sift's leaves, its vines and its grass
    li.update(oak_leaves=at['sift_leaves'], spruce_leaves=at['sift_vines'], birch_leaves=at['pink_grass'],
              tall_grass=at['blue_grass'])
    r.set_blocks(at.rgba, at.emit, noise_tex(), li)
    r.set_moon(SKY.moon_texture())
    for s in skies:
        r.set_sky(s, panos[s])
    print(f'[scene] renderer {width}x{height} ss {ss} ready in {time.time() - t:.1f}s', flush=True)
    return r


def register_world(r, key, world, cache_name=None, sources=('blocks.py', 'voxel.py', 'textures.py')):
    """Mesh + light a voxel world and upload it (both cached on disk by name and the hash of its sources).
    world may be a function building it (only called when the cache misses)."""
    t = time.time()
    if cache_name:
        import hashlib
        h = hashlib.md5()
        here = os.path.dirname(os.path.abspath(__file__))
        for f in sources:
            with open(os.path.join(here, f), 'rb') as fh:
                h.update(fh.read())
        cache_name = f'{cache_name}_{h.hexdigest()[:8]}'
    path = os.path.join(CACHE, f'world_{cache_name}.npz') if cache_name else None
    if path and os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        mesh = d['mesh'].item()
        light = d['light']
    else:
        if callable(world):
            world = world()
        mesh = world.mesh()
        light = world.light()
        if path:
            os.makedirs(CACHE, exist_ok=True)
            np.savez(path, mesh=np.array(mesh, dtype=object), light=light)
    r.add_world(key, mesh, light, world.origin)
    n = sum(len(v[1]) // 3 for v in mesh.values())
    print(f'[scene] world {key}: {n} triangles, {time.time() - t:.1f}s', flush=True)
    return mesh
