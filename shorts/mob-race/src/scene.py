"""The race's world: the marble run built on the face of a cliff on a sunny day. The cliff runs from grass at the
top, through stone with ores, into deepslate near the bottom, where the final is run over a lava lake. The course's
blocks are a static mesh; the parts that move (trapdoors, the pens' floors, pistons, TNT, the starting floor) and the
lava surfaces are a second mesh rebuilt every frame. Builds the renderer with the textures and the daylight.

Units: one Minecraft block is one unit. x to the right, y away from the camera (the cliff is behind the course, at
y = WALL_Y), z up. The course is in the plane y = 0.
"""
import numpy as np

import blocks as B
import course as C
import renderer as RD
import sky as SKY
from noise import fbm2d
from vfx import puff_textures

L = B.L
DEPTH = (-0.75, 0.8)              # the course's pieces stick out of the cliff from y = 0.8 to y = -0.75
WALL_Y = DEPTH[1]
SUN_DIR = np.array([-0.40, -0.62, 0.68])
SUN_DIR = SUN_DIR / np.linalg.norm(SUN_DIR)


class MeshBuilder:
    def __init__(self):
        self.v = []
        self.i = []
        self.n = 0

    def quad(self, p, nrm, uv, layer, tint=(1.0, 1.0, 1.0)):
        for k in range(4):
            self.v.append((*p[k], *nrm, *uv[k], layer, *tint))
        b = self.n
        self.i.extend((b, b + 1, b + 2, b, b + 2, b + 3))
        self.n += 4

    def arrays(self):
        if not self.v:
            return np.zeros((0, 12), np.float32), np.zeros(0, np.uint32)
        return np.array(self.v, np.float32).reshape(-1, 12), np.array(self.i, np.uint32)


def box(mb, x0, x1, y0, y1, z0, z1, layers, faces=('px', 'nx', 'ny', 'pz', 'nz'), tint=(1, 1, 1)):
    """An axis-aligned box with block textures tiled one per unit. layers: one layer or {face: layer}."""
    def lay(f):
        return layers[f] if isinstance(layers, dict) else layers
    for f in faces:
        if f == 'pz':
            p = [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
            uv = [(x0, -y0), (x1, -y0), (x1, -y1), (x0, -y1)]
            n = (0, 0, 1)
        elif f == 'nz':
            p = [(x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (x0, y0, z0)]
            uv = [(x0, y1), (x1, y1), (x1, y0), (x0, y0)]
            n = (0, 0, -1)
        elif f == 'px':
            p = [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)]
            uv = [(y0, -z0), (y1, -z0), (y1, -z1), (y0, -z1)]
            n = (1, 0, 0)
        elif f == 'nx':
            p = [(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)]
            uv = [(-y1, -z0), (-y0, -z0), (-y0, -z1), (-y1, -z1)]
            n = (-1, 0, 0)
        elif f == 'py':
            p = [(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)]
            uv = [(-x1, -z0), (-x0, -z0), (-x0, -z1), (-x1, -z1)]
            n = (0, 1, 0)
        else:
            p = [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)]
            uv = [(x0, -z0), (x1, -z0), (x1, -z1), (x0, -z1)]
            n = (0, -1, 0)
        mb.quad(p, n, uv, lay(f), tint)


def oriented_box(mb, c, u, half_len, half_t, y0, y1, layer, v_scale=1.0, tint=(1, 1, 1), end_layer=None):
    """A box of length 2 half_len along the unit vector u (in x-z), 2 half_t thick across it, from y0 to y1 in depth;
    the texture runs along its length."""
    c = np.asarray(c, float)
    u = np.asarray(u, float) / np.linalg.norm(u)
    w = np.array([-u[1], u[0]])                 # across, pointing 'up' for a beam going right
    corners = {}
    for su in (-1, 1):
        for sw in (-1, 1):
            corners[(su, sw)] = c + su * half_len * u + sw * half_t * w

    def P(k, y):
        x, z = corners[k]
        return (float(x), float(y), float(z))

    L2 = 2 * half_len
    T2 = 2 * half_t * v_scale
    D = y1 - y0
    # front (towards the camera, -y) and back
    mb.quad([P((-1, -1), y0), P((1, -1), y0), P((1, 1), y0), P((-1, 1), y0)], (0, -1, 0),
            [(0, T2), (L2, T2), (L2, 0), (0, 0)], layer, tint)
    nw = (float(w[0]), 0.0, float(w[1]))
    # top (along +w) and bottom
    mb.quad([P((-1, 1), y0), P((1, 1), y0), P((1, 1), y1), P((-1, 1), y1)], nw,
            [(0, 0), (L2, 0), (L2, D), (0, D)], layer, tint)
    mb.quad([P((1, -1), y0), P((-1, -1), y0), P((-1, -1), y1), P((1, -1), y1)], (-nw[0], 0.0, -nw[2]),
            [(0, 0), (L2, 0), (L2, D), (0, D)], layer, tint)
    el = end_layer if end_layer is not None else layer
    nu = (float(u[0]), 0.0, float(u[1]))
    mb.quad([P((1, -1), y0), P((1, 1), y0), P((1, 1), y1), P((1, -1), y1)], nu,
            [(0, 0), (T2, 0), (T2, D), (0, D)], el, tint)
    mb.quad([P((-1, 1), y0), P((-1, -1), y0), P((-1, -1), y1), P((-1, 1), y1)], (-nu[0], 0.0, -nu[2]),
            [(0, 0), (T2, 0), (T2, D), (0, D)], el, tint)


# ---------------------------------------------------------------------------------------------
# the cliff
# ---------------------------------------------------------------------------------------------
def cliff_layer(x, z, rng, noise):
    """Which block the cliff face is at column x, row z (block coordinates of the block's lower corner)."""
    n = noise(x, z)
    if z >= 3:
        return None
    if z == 2:
        return L['grass_side']
    if z >= -1:
        return L['dirt']
    deep = z < -64 + 5.0 * n
    r = rng.random()
    if deep:
        if z < -118 + 3 * n:
            return L['bedrock'] if r < 0.35 else L['deepslate']
        if r < 0.012:
            return L['deep_diamond']
        return L['deepslate']
    if n > 0.62:
        return L['mossy'] if r < 0.6 else L['stone']
    if r < 0.030:
        return L['coal_ore']
    if r < 0.045:
        return L['iron_ore']
    if r < 0.052:
        return L['redstone_ore'] if z < -30 else L['coal_ore']
    if r < 0.057:
        return L['gold_ore'] if z < -20 else L['iron_ore']
    if r < 0.061:
        return L['lapis_ore'] if z < -35 else L['coal_ore']
    if r < 0.064:
        return L['diamond_ore'] if z < -45 else L['stone']
    if r < 0.066:
        return L['emerald_ore']
    return L['stone2'] if n < 0.3 else L['stone']


def build_static(course, seed=5):
    rng = np.random.default_rng(seed)
    mb = MeshBuilder()
    nz = fbm2d(np.linspace(0, 6, 64)[None, :], np.linspace(0, 40, 400)[:, None], octaves=3, seed=seed)
    nz = (nz - nz.min()) / (np.ptp(nz) + 1e-9)

    def noise(x, z):
        return float(nz[int(np.clip(-z / 136.0 * 399, 0, 399)), int(np.clip((x + 20) / 40.0 * 63, 0, 63))])

    z_lo = int(np.floor(course.z_end)) - 22
    # the cliff face behind the course; wider than the course, with some relief outside it
    for zi in range(z_lo, 3):
        for xi in range(-20, 20):
            lay = cliff_layer(xi, zi, rng, noise)
            if lay is None:
                continue
            y = WALL_Y
            if abs(xi + 0.5) > C.HALF + 1.5:
                y = WALL_Y - (1.0 if noise(xi * 1.7, zi * 1.3) > 0.55 else 0.0)
            box(mb, xi, xi + 1, y, y + 1.0, zi, zi + 1, lay, faces=('ny',))
            if y < WALL_Y:
                box(mb, xi, xi + 1, y, WALL_Y, zi, zi + 1, lay, faces=('px', 'nx', 'pz', 'nz'))
    # grass on top of the cliff, and a few blocks set back so the edge isn't a straight line
    for xi in range(-20, 20):
        box(mb, xi, xi + 1, WALL_Y, WALL_Y + 6.0, 2, 3, {'pz': L['grass_top'], 'ny': L['grass_side']},
            faces=('pz',))
    # the foot of the cliff: a netherrack shore and bedrock under the lava lake
    zl = course.lake[1]
    box(mb, -20.0, 20.0, -14.0, WALL_Y, zl - 8.0, zl, {'pz': L['netherrack'], 'ny': L['bedrock'],
                                                             'px': L['bedrock'], 'nx': L['bedrock']},
        faces=('ny', 'pz'))
    # the course's pieces
    y0, y1 = DEPTH
    for p in course.pieces:
        lay = L.get(p['mat'], L['stone_bricks'])
        if p['kind'] == 'box':
            if p['mat'] == 'iron_bars':
                box(mb, p['x0'], p['x1'], y0 + 0.2, y1, p['z0'], p['z1'], lay)
            else:
                box(mb, p['x0'], p['x1'], y0, y1, p['z0'], p['z1'], lay)
        elif p['kind'] == 'beam':
            a, b = np.array(p['a']), np.array(p['b'])
            L2 = np.linalg.norm(b - a) + p['t']
            oriented_box(mb, (a + b) / 2, b - a, L2 / 2, p['t'] / 2, y0, y1, lay)
        else:
            x, z = p['c']
            r = 0.16
            box(mb, x - r, x + r, y0 + 0.15, y1, z - r, z + r, L['end_rod'])
            box(mb, x - 0.3, x + 0.3, y1 - 0.12, y1, z - 0.3, z + 0.3, L['end_rod'])
    v, i = mb.arrays()
    lo = v[:, :3].min(0)
    hi = v[:, :3].max(0)
    return v, i, [(0, len(i), lo, hi)]


def day_sky(width=2048, height=512, seed=11):
    """A clear midday sky with fair-weather clouds (equirectangular, linear)."""
    rows = (np.arange(height) + 0.5) / height
    el = np.radians(90.0 - rows * (90.0 - SKY.EL_MIN))
    cols = (np.arange(width) + 0.5) / width
    phi = cols * 2 * np.pi
    EL, PHI = np.meshgrid(el, phi, indexing='ij')
    dz = np.sin(EL)
    t = np.clip(dz, 0, 1) ** 0.45
    zen = np.array([0.10, 0.30, 0.82])
    hor = np.array([0.62, 0.80, 1.00])
    sky = hor * (1 - t[..., None]) + zen * t[..., None]
    field = SKY.cloud_field(1024, seed)
    s = 1.0 / np.maximum(dz, 0.02)
    u = np.cos(EL) * np.cos(PHI) * s / 4.0
    v = np.cos(EL) * np.sin(PHI) * s / 4.0
    dens = SKY._bilinear_wrap(field, u, v)
    cover = SKY.smoothstep(0.15, 0.6, dens) * SKY.smoothstep(0.02, 0.2, dz)
    ccol = np.array([1.0, 1.0, 1.0]) * (0.95 - 0.25 * SKY.smoothstep(0.4, 1.0, dens))[..., None]
    sky = sky * (1 - cover[..., None]) + ccol * 1.25 * cover[..., None]
    sky[dz < 0] = hor * 0.9
    return sky.astype(np.float32)


def tint_noise(n=256):
    v = (np.arange(n) + 0.5) / n
    x, y = np.meshgrid(v, v)
    t = fbm2d(x * 4, y * 4, octaves=4, seed=77, period=4)
    t = (t - t.min()) / (t.max() - t.min())
    return t.astype(np.float32)


def make_renderer(course, width=1080, height=1920, ss=1.0, shadow_res=4096):
    RD.SUN_DIR = SUN_DIR
    r = RD.Renderer(width, height, ss=ss, shadow_res=shadow_res, near_half=26.0, far_half=90.0)
    r.far_center = (0.0, 0.0, (course.z_end + 4.0) / 2.0)
    layers, _ = B.textures()
    r.set_textures(layers, day_sky(), tint_noise(), puff_textures())
    v, i, chunks = build_static(course)
    r.set_static(v, i, chunks, ground_half=0.01)
    r.sun_col = np.array([1.0, 0.95, 0.86]) * 2.5
    r.sky_amb = np.array([0.42, 0.52, 0.78]) * 0.55
    r.gnd_amb = np.array([0.36, 0.32, 0.28]) * 0.4
    r.bounce = (0.05, 0.05, 0.045)
    r.fog = 0.0015
    r.fog_col = (-1.0, 0.0, 0.0)
    r.exposure = 0.62
    r.contrast = 1.12
    r.vignette = 0.22
    r.bloom_thresh = 2.6
    r.bloom = 0.35
    r.sculk_r = 0.0
    r.glow = 3.0
    r.ssao_radius = 0.6
    return r
