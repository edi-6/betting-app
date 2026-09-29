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
    r = gfx.Renderer(width, height, ss=ss, shadow_res=shadow_res, near_half=near_half, far_half=far_half)
    at = VX.atlas()
    li = {k: at[k] for k in ('water', 'grass_top', 'grass_side', 'oak_leaves', 'spruce_leaves', 'birch_leaves',
                             'tall_grass')}
    r.set_blocks(at.rgba, at.emit, noise_tex(), li)
    r.set_moon(SKY.moon_texture())
    for s in skies:
        r.set_sky(s, SKY.panorama(s).astype(np.float32))
    print(f'[scene] renderer {width}x{height} ss {ss} ready in {time.time() - t:.1f}s', flush=True)
    return r


def register_world(r, key, world, cache_name=None):
    """Mesh + light a voxel world and upload it (both cached on disk by name)."""
    t = time.time()
    path = os.path.join(CACHE, f'world_{cache_name}.npz') if cache_name else None
    if path and os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        mesh = d['mesh'].item()
        light = d['light']
    else:
        mesh = world.mesh()
        light = world.light()
        if path:
            os.makedirs(CACHE, exist_ok=True)
            np.savez(path, mesh=np.array(mesh, dtype=object), light=light)
    r.add_world(key, mesh, light, world.origin)
    n = sum(len(v[1]) // 3 for v in mesh.values())
    print(f'[scene] world {key}: {n} triangles, {time.time() - t:.1f}s', flush=True)
    return mesh
