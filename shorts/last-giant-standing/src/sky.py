"""Procedural sky panorama (equirectangular, linear HDR) with a layer of cumulus clouds."""
import numpy as np
from noise import fbm2d, perlin2d

# golden hour: the sun low in the front-right of the giant (who faces -Y)
_AZ, _EL = np.radians(-55.0), np.radians(13.0)
SUN_DIR = np.array([np.cos(_EL) * np.cos(_AZ), np.cos(_EL) * np.sin(_AZ), np.sin(_EL)])
SUN_DIR = SUN_DIR / np.linalg.norm(SUN_DIR)

ZENITH = np.array([0.08, 0.17, 0.47])
HORIZON_SUN = np.array([1.30, 0.68, 0.30])      # warm horizon under the sun
HORIZON_AWAY = np.array([0.62, 0.52, 0.66])     # soft violet on the far side
SUN_GLOW = np.array([1.25, 0.58, 0.20])
# cloud lighting (used by clouds.py)
CLOUD_SUN = np.array([1.0, 0.64, 0.38])
CLOUD_AMB_LO = np.array([0.24, 0.23, 0.38])
CLOUD_AMB_HI = np.array([0.60, 0.55, 0.74])
EL_MIN = -12.0  # degrees covered below the horizon


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def cloud_field(n=2048, seed=11):
    """Tileable 2D cloud density map in [0,1] on an n x n grid (one tile = 1 unit of the cloud plane)."""
    v = (np.arange(n) + 0.5) / n
    x, y = np.meshgrid(v, v)
    base_period = 6
    # domain warp for natural, billowy shapes
    wx = fbm2d(x * base_period, y * base_period, octaves=3, seed=seed + 1, period=base_period)
    wy = fbm2d(x * base_period + 5.2, y * base_period + 1.3, octaves=3, seed=seed + 2, period=base_period)
    xw = x * base_period + 0.9 * wx
    yw = y * base_period + 0.9 * wy
    shape = fbm2d(xw, yw, octaves=6, gain=0.52, seed=seed, period=base_period)
    # billow detail (abs noise gives puffy cauliflower edges)
    det = np.abs(perlin2d(xw * 4, yw * 4, seed=seed + 7, period=base_period * 4))
    det += 0.5 * np.abs(perlin2d(xw * 9, yw * 9, seed=seed + 9, period=base_period * 9))
    d = shape + 0.22 * det
    lo, hi = np.percentile(d, 55), np.percentile(d, 99.5)
    return np.clip((d - lo) / (hi - lo), 0, 1)


def _bilinear_wrap(field, u, v):
    n = field.shape[0]
    x = (u % 1.0) * n - 0.5
    y = (v % 1.0) * n - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = x - x0
    fy = y - y0
    x0 %= n
    y0 %= n
    x1 = (x0 + 1) % n
    y1 = (y0 + 1) % n
    a = field[y0, x0] * (1 - fx) + field[y0, x1] * fx
    b = field[y1, x0] * (1 - fx) + field[y1, x1] * fx
    return a * (1 - fy) + b * fy


def make_sky(width=8192, height=2048, seed=11, clouds=True):
    """Return float32 (height, width, 3) linear radiance panorama.

    Column -> azimuth phi in [0, 2pi) measured from +X towards +Y; row 0 = zenith, last row = EL_MIN.
    """
    rows = (np.arange(height) + 0.5) / height
    el = np.radians(90.0 - rows * (90.0 - EL_MIN))            # elevation per row
    cols = (np.arange(width) + 0.5) / width
    phi = cols * 2 * np.pi
    EL, PHI = np.meshgrid(el, phi, indexing='ij')
    dx = np.cos(EL) * np.cos(PHI)
    dy = np.cos(EL) * np.sin(PHI)
    dz = np.sin(EL)

    # --- clear sky gradient -------------------------------------------------------------------
    t = np.clip(dz, 0, 1)
    k = np.power(t, 0.5)
    # horizon colour swings from warm (towards the sun) to violet (away from it)
    sh = SUN_DIR[:2] / np.linalg.norm(SUN_DIR[:2])
    mu_h = (np.cos(PHI) * sh[0] + np.sin(PHI) * sh[1]) * 0.5 + 0.5
    hz = HORIZON_AWAY[None, None, :] + (HORIZON_SUN - HORIZON_AWAY)[None, None, :] * (mu_h ** 1.6)[..., None]
    sky = hz * (1 - k[..., None]) + ZENITH[None, None, :] * k[..., None]
    mu = dx * SUN_DIR[0] + dy * SUN_DIR[1] + dz * SUN_DIR[2]
    glow = 0.75 * np.power(np.clip(mu, 0, 1), 5) + 0.45 * np.power(np.clip(mu, 0, 1), 60) \
        + 1.4 * np.power(np.clip(mu, 0, 1), 900)
    sky = sky + glow[..., None] * SUN_GLOW
    # a warm haze band along the horizon
    haze = np.exp(-np.clip(dz, 0, None) * 10.0)
    sky = sky * (1 - 0.3 * haze[..., None]) + 0.3 * haze[..., None] * (hz * 1.1)
    below = dz < 0
    sky[below] = (hz * 0.85)[below]

    if not clouds:
        return sky.astype(np.float32)
    # --- cloud layer ----------------------------------------------------------------------------
    field = cloud_field(2048, seed)
    H = 1.0
    sel = dz > 0.012
    s = H / np.maximum(dz, 0.012)
    px = dx * s
    py = dy * s
    scale = 5.5                      # cloud plane units per noise tile
    u = px / scale
    v = py / scale
    dens = _bilinear_wrap(field, u, v)
    # lighting: sample density a little towards the sun; denser there -> we're on the shadow side
    sd = SUN_DIR[:2] / np.linalg.norm(SUN_DIR[:2])
    off = 0.018
    dens_s = _bilinear_wrap(field, u + sd[0] * off, v + sd[1] * off)
    cover = smoothstep(0.10, 0.55, dens)
    thick = smoothstep(0.25, 1.0, dens)
    lit = np.clip(0.5 + 2.2 * (dens - dens_s), 0, 1)
    sun_col = CLOUD_SUN
    shadow_col = CLOUD_AMB_HI * 0.9
    ccol = shadow_col[None, None, :] * (1 - lit[..., None]) + sun_col[None, None, :] * lit[..., None]
    ccol = ccol * (1.0 - 0.28 * thick[..., None])
    ccol = ccol * 1.08 + 0.10 * np.array([0.9, 0.95, 1.0])
    # fade clouds into haze at low elevations (they are very far away there)
    fade = smoothstep(0.02, 0.28, dz)
    a = cover * fade * sel
    far_tint = smoothstep(0.0, 0.35, dz)[..., None]
    ccol = ccol * far_tint + (1 - far_tint) * np.array([0.86, 0.72, 0.70])
    sky = sky * (1 - a[..., None]) + ccol * a[..., None]
    return sky.astype(np.float32)


if __name__ == '__main__':
    import sys
    from PIL import Image
    out = sys.argv[1]
    sky = make_sky(4096, 1024)
    img = np.clip(sky / (1 + sky) * 1.6, 0, 1) ** (1 / 2.2)
    Image.fromarray((img * 255).astype(np.uint8)).save(out)
