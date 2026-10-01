"""Sky panoramas for the film (equirectangular, linear HDR, row 0 = zenith, last row = 12 degrees below the horizon):
an analytic gradient per time of day plus volumetric clouds raymarched once on the GPU (the technique of the earlier
videos' clouds.py, with the light direction and colours as parameters), and the square pixel-art moon.

Presets: sunset (the perfect world), dusk, night (the footsteps, the finale), wrong (the changed world: a sickly
green sky under an enormous moon), void (underground: never seen).
"""
import os

import numpy as np
import moderngl

EL_MIN = -12.0
CACHE = os.environ.get('COASTER_CACHE', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cache'))


def sun_dir(az_deg, el_deg):
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


PRESETS = {
    # the coaster: golden hour over the badlands (the sun low in the west-south-west, a few clouds)
    'golden': dict(sun=sun_dir(196.0, 13.0), zenith=(0.13, 0.26, 0.60), horizon_sun=(1.75, 1.00, 0.52),
                   horizon_away=(0.52, 0.60, 0.84), glow=(2.2, 1.1, 0.45), glow_pow=(4.0, 24.0, 500.0),
                   cloud_sun=(3.6, 2.1, 1.0), cloud_amb_lo=(0.22, 0.24, 0.34), cloud_amb_hi=(0.55, 0.58, 0.72),
                   coverage=0.42, sun_scale=1.0),
    # the Nether has no sky: a smoky red haze that the fog blends into
    'nether': dict(sun=sun_dir(0.0, 60.0), zenith=(0.030, 0.008, 0.006), horizon_sun=(0.20, 0.055, 0.025),
                   horizon_away=(0.20, 0.055, 0.025), glow=(0.0, 0.0, 0.0), glow_pow=(4.0, 20.0, 400.0),
                   clouds=False),
    # the End: the game's dark, blotchy purple-grey sky (drawn in pixel blocks)
    'end': dict(sun=sun_dir(0.0, 60.0), clouds=False, end=True),
    # the deep dark: no sky, a cold near-black the fog blends into
    'deep': dict(sun=sun_dir(0.0, 60.0), zenith=(0.004, 0.010, 0.016), horizon_sun=(0.010, 0.026, 0.036),
                 horizon_away=(0.010, 0.026, 0.036), glow=(0.0, 0.0, 0.0), glow_pow=(4.0, 20.0, 400.0),
                 clouds=False),
    # the Sift: a turquoise sky, a pink glow round its sun, pink-lit clouds
    'sift': dict(sun=sun_dir(235.0, 30.0), zenith=(0.012, 0.20, 0.25), horizon_sun=(0.56, 0.62, 0.66),
                 horizon_away=(0.10, 0.46, 0.50), glow=(1.1, 0.45, 0.62), glow_pow=(4.0, 22.0, 450.0),
                 cloud_sun=(1.9, 1.3, 1.5), cloud_amb_lo=(0.10, 0.24, 0.28), cloud_amb_hi=(0.34, 0.54, 0.58),
                 coverage=0.24, sun_scale=0.9),
    # the flight: the Sift at golden hour (the sun low in the west-south-west, peach at the horizon under it, the
    # turquoise sky deepening overhead, a few clouds lit pink and gold)
    'sift_gold': dict(sun=sun_dir(165.0, 12.0), zenith=(0.020, 0.150, 0.240), horizon_sun=(1.55, 0.86, 0.62),
                      horizon_away=(0.20, 0.46, 0.52), glow=(2.0, 0.85, 0.80), glow_pow=(4.0, 22.0, 450.0),
                      cloud_sun=(3.4, 1.8, 1.55), cloud_amb_lo=(0.14, 0.22, 0.28), cloud_amb_hi=(0.40, 0.52, 0.58),
                      coverage=0.34, sun_scale=1.0, ridges=2.0),
    # the flight by night: a moon high in the west-north-west, a deep teal-blue sky, clouds faintly moonlit
    'sift_night': dict(sun=sun_dir(165.0, 32.0), zenith=(0.002, 0.007, 0.016), horizon_sun=(0.030, 0.042, 0.062),
                       horizon_away=(0.012, 0.026, 0.040), glow=(0.035, 0.045, 0.065), glow_pow=(4.0, 22.0, 450.0),
                       cloud_sun=(0.16, 0.19, 0.25), cloud_amb_lo=(0.008, 0.012, 0.020),
                       cloud_amb_hi=(0.020, 0.030, 0.045), coverage=0.30, sun_scale=0.5, ridges=2.1),
    'sunset': dict(sun=sun_dir(200.0, 5.0), zenith=(0.10, 0.17, 0.40), horizon_sun=(1.70, 0.78, 0.32),
                   horizon_away=(0.42, 0.40, 0.62), glow=(2.4, 1.0, 0.35), glow_pow=(4.0, 22.0, 400.0),
                   cloud_sun=(3.4, 1.55, 0.62), cloud_amb_lo=(0.18, 0.16, 0.26), cloud_amb_hi=(0.42, 0.36, 0.52),
                   coverage=0.47, sun_scale=1.0),
    'dusk': dict(sun=sun_dir(200.0, -3.0), zenith=(0.03, 0.05, 0.14), horizon_sun=(0.60, 0.26, 0.14),
                 horizon_away=(0.08, 0.09, 0.18), glow=(0.7, 0.24, 0.08), glow_pow=(3.0, 12.0, 200.0),
                 cloud_sun=(0.9, 0.35, 0.18), cloud_amb_lo=(0.05, 0.05, 0.10), cloud_amb_hi=(0.10, 0.10, 0.20),
                 coverage=0.47, sun_scale=0.6),
    'night': dict(sun=sun_dir(20.0, 38.0), zenith=(0.004, 0.007, 0.018), horizon_sun=(0.020, 0.028, 0.050),
                  horizon_away=(0.012, 0.016, 0.032), glow=(0.020, 0.026, 0.040), glow_pow=(6.0, 40.0, 800.0),
                  cloud_sun=(0.10, 0.12, 0.17), cloud_amb_lo=(0.006, 0.008, 0.014),
                  cloud_amb_hi=(0.014, 0.018, 0.030), coverage=0.44, sun_scale=0.25),
    'wrong': dict(sun=sun_dir(262.0, 10.0), zenith=(0.010, 0.030, 0.022), horizon_sun=(0.16, 0.30, 0.12),
                  horizon_away=(0.03, 0.07, 0.05), glow=(0.30, 0.40, 0.16), glow_pow=(3.0, 16.0, 300.0),
                  cloud_sun=(0.34, 0.46, 0.22), cloud_amb_lo=(0.012, 0.030, 0.020),
                  cloud_amb_hi=(0.03, 0.06, 0.04), coverage=0.56, sun_scale=0.35),
}


def gradient(p, width, height):
    rows = (np.arange(height) + 0.5) / height
    el = np.radians(90.0 - rows * (90.0 - EL_MIN))
    phi = (np.arange(width) + 0.5) / width * 2 * np.pi
    EL, PHI = np.meshgrid(el, phi, indexing='ij')
    dx, dy, dz = np.cos(EL) * np.cos(PHI), np.cos(EL) * np.sin(PHI), np.sin(EL)
    s = np.asarray(p['sun'], float)
    sh = s[:2] / np.linalg.norm(s[:2])
    mu_h = (np.cos(PHI) * sh[0] + np.sin(PHI) * sh[1]) * 0.5 + 0.5
    hz = np.array(p['horizon_away'])[None, None, :] + (np.array(p['horizon_sun']) - np.array(p['horizon_away']))[
        None, None, :] * (mu_h ** 2.2)[..., None]
    k = np.power(np.clip(dz, 0, 1), 0.42)
    sky = hz * (1 - k[..., None]) + np.array(p['zenith'])[None, None, :] * k[..., None]
    mu = np.clip(dx * s[0] + dy * s[1] + dz * s[2], 0, 1)
    a, b, c = p['glow_pow']
    glow = 0.55 * mu ** a + 0.35 * mu ** b + 0.8 * mu ** c
    sky = sky + glow[..., None] * np.array(p['glow'])[None, None, :]
    below = dz < 0
    fade = np.clip(-dz / 0.2, 0, 1)[..., None]
    sky = np.where(below[..., None], hz * (1 - 0.6 * fade), sky)
    return sky.astype(np.float32)


# the cloud raymarch (see the domino video's clouds.py for the technique)
from clouds_gl import BASE_FS, DETAIL_FS, FS_VS, MARCH_FS  # noqa: E402


def bake_clouds(p, width, height, tile=512, verbose=True):
    ctx = moderngl.create_standalone_context(backend='egl', require=430)
    quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], np.float32).tobytes())

    def volume(fs, res):
        prog = ctx.program(vertex_shader=FS_VS, fragment_shader=fs)
        vao = ctx.vertex_array(prog, [(quad, '2f', 'in_pos')])
        tex = ctx.texture((res, res), 4, dtype='f4')
        fbo = ctx.framebuffer(color_attachments=[tex])
        fbo.use()
        vol = np.zeros((res, res, res, 2), np.float32)
        for z in range(res):
            prog['u_z'] = (z + 0.5) / res
            vao.render(moderngl.TRIANGLE_STRIP)
            vol[z] = np.frombuffer(fbo.read(components=4, dtype='f4'), np.float32).reshape(res, res, 4)[..., :2]
        return vol

    base = volume(BASE_FS, 128)
    detail = volume(DETAIL_FS, 64)

    def stretch(a):
        lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
        return np.clip((a - lo) / (hi - lo), 0, 1).astype(np.float32)
    base[..., 0] = stretch(base[..., 0])
    base[..., 1] = stretch(base[..., 1])
    detail[..., 0] = stretch(detail[..., 0])
    tb = ctx.texture3d((128, 128, 128), 2, np.ascontiguousarray(base).tobytes(), dtype='f4')
    td = ctx.texture3d((64, 64, 64), 1, np.ascontiguousarray(detail[..., 0]).tobytes(), dtype='f4')
    for t in (tb, td):
        t.filter = (moderngl.LINEAR, moderngl.LINEAR)
        t.repeat_x = t.repeat_y = t.repeat_z = True
    prog = ctx.program(vertex_shader=FS_VS, fragment_shader=MARCH_FS)
    vao = ctx.vertex_array(prog, [(quad, '2f', 'in_pos')])
    tb.use(0)
    td.use(1)
    prog['u_base'] = 0
    prog['u_detail'] = 1
    s = np.asarray(p['sun'], float)
    if s[2] < 0.05:
        s = s.copy()
        s[2] = 0.05                               # light the clouds from just above the horizon
        s /= np.linalg.norm(s)
    prog['u_sun'] = tuple(s)
    prog['u_sun_col'] = tuple(float(v) for v in p['cloud_sun'])
    prog['u_amb_lo'] = tuple(float(v) for v in p['cloud_amb_lo'])
    prog['u_amb_hi'] = tuple(float(v) for v in p['cloud_amb_hi'])
    prog['u_el_min'] = EL_MIN
    prog['u_coverage'] = float(p['coverage'])
    out = np.zeros((height, width, 4), np.float32)
    tex = ctx.texture((tile, tile), 4, dtype='f4')
    fbo = ctx.framebuffer(color_attachments=[tex])
    fbo.use()
    import time
    t0 = time.time()
    for ty in range(0, height, tile):
        for tx in range(0, width, tile):
            prog['u_tile0'] = (tx / width, ty / height)
            prog['u_tile_size'] = (tile / width, tile / height)
            vao.render(moderngl.TRIANGLE_STRIP)
            out[ty:ty + tile, tx:tx + tile] = np.frombuffer(fbo.read(components=4, dtype='f4'),
                                                            np.float32).reshape(tile, tile, 4)
        if verbose:
            print(f'  clouds row {ty // tile + 1}/{height // tile}  {time.time() - t0:.0f}s', flush=True)
    ctx.release()
    return out


def panorama(name, width=4096, height=1024):
    """Cached panorama for a preset (float16 (H, W, 3))."""
    os.makedirs(CACHE, exist_ok=True)
    import hashlib
    key = hashlib.md5(repr(sorted((k, np.round(np.asarray(v, float), 5).tolist()) for k, v in
                                  PRESETS.get(name, {}).items())).encode()).hexdigest()[:8]
    path = os.path.join(CACHE, f'sky_{name}_{width}x{height}_{key}.npy')
    if os.path.exists(path):
        return np.load(path)
    if name == 'void':
        pano = np.zeros((height, width, 3), np.float16)
    elif PRESETS[name].get('end'):
        pano = end_sky(width, height).astype(np.float16)
    elif not PRESETS[name].get('clouds', True):
        pano = gradient(PRESETS[name], width, height).astype(np.float16)
    else:
        p = PRESETS[name]
        clear = gradient(p, width, height)
        c = bake_clouds(p, width, height)
        pano = (clear * c[..., 3:4] + c[..., :3] * p.get('sun_scale', 1.0))
        if p.get('ridges'):
            pano = add_ridges(pano, p)
        pano = pano.astype(np.float16)
    np.save(path, pano)
    return pano


def add_ridges(pano, p, seed=5):
    """Far mountain ranges round the horizon, two layers of them, hazed nearly to the sky's colour (lighter and
    bluer the further), their tops catching a little of the sun's glow on the side it sets."""
    H, W, _ = pano.shape
    rows = (np.arange(H) + 0.5) / H
    el = 90.0 - rows * (90.0 - EL_MIN)
    phi = (np.arange(W) + 0.5) / W * 2 * np.pi
    rng = np.random.default_rng(seed)
    hz = gradient(p, W, H)
    k0 = int(np.argmin(np.abs(el - 0.3)))
    horizon = hz[k0]                                    # the sky's colour just over the horizon, per column
    s = np.asarray(p['sun'], float)
    mu = np.clip(np.cos(phi) * s[0] / np.hypot(s[0], s[1]) + np.sin(phi) * s[1] / np.hypot(s[0], s[1]), 0, 1)

    def profile(octaves, amp, base):
        h = np.zeros(W)
        for o in range(octaves):
            k = 3 * 2 ** o
            ph = rng.uniform(0, 2 * np.pi, 3)
            h += (np.sin(k * phi + ph[0]) + 0.5 * np.sin(2.1 * k * phi + ph[1]) + 0.3 * np.sin(3.3 * k * phi + ph[2]))\
                / 2 ** o
        h = (h - h.min()) / (h.max() - h.min())
        return base + amp * h ** 1.4

    out = pano.copy()
    for (octs, amp, base, shade, blue) in ((5, 4.2, 0.6, 0.84, 0.05), (6, 3.0, -0.4, 0.70, 0.03)):
        top = profile(octs, amp, base)
        col = horizon * shade + np.array([0.0, 0.01, blue])[None, :] * np.clip(horizon.mean(1, keepdims=True) / 0.3, 0, 1)
        col = col * (1.0 - 0.35 * mu[:, None] ** 3) + np.array([0.20, 0.10, 0.06])[None, :] * mu[:, None] ** 6
        cover = np.clip((top[None, :] - el[:, None]) / 0.12, 0, 1)        # soft along the ridge line
        below = np.clip((top[None, :] - el[:, None]) / 3.0, 0, 1)          # a touch darker lower down
        c = col[None, :, :] * (1.0 - 0.10 * below[..., None])
        out = out * (1 - cover[..., None]) + c * cover[..., None]
    return out


def end_sky(width, height, seed=11):
    """The End's sky: a dark purple-grey texture of blotches, pixelated (like the game's end_sky, seen on the inside
    of a box), a little lighter towards the horizon."""
    rng = np.random.default_rng(seed)
    px = 16                                            # one 'texel' of the sky every 16 panorama pixels
    gh, gw = height // px, width // px
    blot = rng.random((gh, gw))
    big = np.kron(rng.random((gh // 4 + 1, gw // 4 + 1)), np.ones((4, 4)))[:gh, :gw]
    v = 0.55 * blot + 0.45 * big
    base = np.array([0.030, 0.022, 0.040])
    hi = np.array([0.075, 0.050, 0.095])
    col = base[None, None] + (hi - base)[None, None] * (v ** 2)[..., None]
    col = np.kron(col, np.ones((px, px, 1)))
    rows = (np.arange(height) + 0.5) / height
    el = 90.0 - rows * (90.0 - EL_MIN)
    lift = 1.0 + 0.5 * np.clip(1 - el / 40.0, 0, 1)
    return (col * lift[:, None, None]).astype(np.float32)


def moon_texture(n=32, seed=4):
    """The square moon: pale, with darker mare patches (RGBA uint8)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((n, n, 4), np.uint8)
    base = np.array((226, 230, 238), float)
    col = np.tile(base, (n, n, 1))
    blob = np.kron(rng.random((n // 4, n // 4)), np.ones((4, 4)))
    col *= (0.86 + 0.14 * blob)[..., None]
    for _ in range(9):
        cx, cy = rng.integers(3, n - 5, 2)
        r = rng.integers(2, 5)
        yy, xx = np.mgrid[0:n, 0:n]
        m = (np.abs(xx - cx) + np.abs(yy - cy)) < r
        col[m] *= 0.78
    col *= (1 + 0.05 * (rng.random((n, n)) - 0.5))[..., None]
    col[0, :] = col[-1, :] = col[:, 0] = col[:, -1] = base * 0.7
    img[..., :3] = np.clip(np.round(col / 6) * 6, 0, 255)
    img[..., 3] = 255
    return img


if __name__ == '__main__':
    import sys
    from PIL import Image
    for name in sys.argv[1:] or ['sunset']:
        pano = panorama(name).astype(np.float32)
        img = np.clip(pano / (1 + pano) * 1.8, 0, 1) ** (1 / 2.2)
        Image.fromarray((img * 255).astype(np.uint8)).resize((2048, 512)).save(f'/tmp/sky_{name}.png')
        print(name, pano.shape, pano.max())
