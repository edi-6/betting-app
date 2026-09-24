"""The studio: a dark workshop with a deepslate tile floor and stone brick walls lit by redstone lamps, and the
hydraulic press in the middle (static frame: base, columns, beam, cylinder, hoses; the ram plate and rod are
voxels, see press.py). Builds the renderer with the studio's textures and lighting.

Units: a Minecraft block is 4 world units. x to the right, y away from the camera, z up.
"""
import numpy as np

import pixelart as PA
import renderer as RD
from noise import fbm2d
from vfx import puff_textures

SL = PA.SL

# press geometry (world units)
BASE = (-7.0, 7.0, -5.5, 5.5, 0.0, 5.0)            # x0 x1 y0 y1 z0 z1
PLATEN = (-3.5, 3.5, -3.5, 3.5, 5.0, 6.0)          # the lower plate the blocks sit on
Z0 = PLATEN[5]                                     # top of the lower platen
COLS = ((-10.0, -7.0), (7.0, 10.0))                # the two columns (x ranges)
COL_Y = (-2.5, 2.5)
BEAM = (-11.0, 11.0, -3.0, 3.0, 30.0, 34.0)
CYL = (-3.0, 3.0, -3.0, 3.0, 24.0, 30.0)           # hydraulic cylinder under the beam
RAM_W = 7.0                                        # the ram plate: 7 x 7 x 2
RAM_H = 2.0
ROD_W = 2.2
ROD_TOP = 26.0                                     # the rod's top end (hidden inside the cylinder)
RAM_REST = 17.0                                    # bottom of the ram at rest
HOSES = ((-5.0, -0.3, 7.0, 25.5), (5.0, -0.3, 7.0, 25.5))   # x, y, z0, z1 of the two hoses
LAMPS = ((-18.0, 21.9, 22.0), (18.0, 21.9, 22.0))
NEAR_CENTER = (0.0, 0.0, 12.0)

KEY_DIR = np.array([0.42, -0.62, 0.66])
KEY_DIR = KEY_DIR / np.linalg.norm(KEY_DIR)


class MeshBuilder:
    def __init__(self):
        self.v = []
        self.i = []
        self.n = 0

    def quad(self, p, nrm, uv, layer, tint):
        for k in range(4):
            self.v.append((*p[k], *nrm, *uv[k], layer, *tint))
        b = self.n
        self.i.extend((b, b + 1, b + 2, b, b + 2, b + 3))
        self.n += 4

    def arrays(self):
        return np.array(self.v, np.float32).reshape(-1, 12), np.array(self.i, np.uint32)


def box_face(mb, x0, y0, z0, x1, y1, z1, face, layer, tint=(1, 1, 1), s=0.25):
    if face == 'pz':
        p = [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        uv = [(x0, -y0), (x1, -y0), (x1, -y1), (x0, -y1)]
        n = (0, 0, 1)
    elif face == 'nz':
        p = [(x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (x0, y0, z0)]
        uv = [(x0, y1), (x1, y1), (x1, y0), (x0, y0)]
        n = (0, 0, -1)
    elif face == 'px':
        p = [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)]
        uv = [(y0, -z0), (y1, -z0), (y1, -z1), (y0, -z1)]
        n = (1, 0, 0)
    elif face == 'nx':
        p = [(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)]
        uv = [(-y1, -z0), (-y0, -z0), (-y0, -z1), (-y1, -z1)]
        n = (-1, 0, 0)
    elif face == 'py':
        p = [(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)]
        uv = [(-x1, -z0), (-x0, -z0), (-x0, -z1), (-x1, -z1)]
        n = (0, 1, 0)
    else:
        p = [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)]
        uv = [(x0, -z0), (x1, -z0), (x1, -z1), (x0, -z1)]
        n = (0, -1, 0)
    mb.quad(p, n, [(u * s, v * s) for (u, v) in uv], layer, tint)


def box(mb, b, layers, tint=(1, 1, 1), faces=('px', 'nx', 'py', 'ny', 'pz', 'nz'), s=0.25):
    """layers: one layer or dict face -> layer."""
    x0, x1, y0, y1, z0, z1 = b
    for f in faces:
        lay = layers[f] if isinstance(layers, dict) else layers
        box_face(mb, x0, y0, z0, x1, y1, z1, f, lay, tint, s)


def build_static():
    mb = MeshBuilder()
    # the room
    box(mb, (-44.0, 44.0, -70.0, 22.0, -1.0, 0.0), SL['deepslate'], faces=('pz',))
    box(mb, (-44.0, 44.0, 22.0, 24.0, 0.0, 48.0), SL['stone_bricks'], faces=('ny',))
    box(mb, (-46.0, -44.0, -70.0, 22.0, 0.0, 48.0), SL['stone_bricks'], faces=('px',))
    box(mb, (44.0, 46.0, -70.0, 22.0, 0.0, 48.0), SL['stone_bricks'], faces=('nx',))
    for (lx, ly, lz) in LAMPS:
        box(mb, (lx - 2.0, lx + 2.0, ly - 0.1, ly + 2.0, lz - 2.0, lz + 2.0), SL['lamp'], faces=('ny', 'px', 'nx'))
    # a dark oak workbench at the back left and a floor stripe around the press
    box(mb, (-34.0, -22.0, 12.0, 20.0, 0.0, 8.0), {'pz': SL['dark_oak'], 'px': SL['dark_oak'], 'nx': SL['dark_oak'],
                                                    'ny': SL['dark_oak'], 'py': SL['dark_oak'], 'nz': SL['dark_oak']})
    box(mb, (22.0, 30.0, 12.0, 20.0, 0.0, 8.0), SL['iron_dark'])
    box(mb, (-12.0, 12.0, -9.0, -7.5, 0.0, 0.05), SL['hazard'], faces=('pz',))
    # the press: base with a hazard band, platen, columns, beam, cylinder, hoses, a gauge
    x0, x1, y0, y1, z0, z1 = BASE
    box(mb, (x0, x1, y0, y1, z0, 1.0), SL['hazard'], faces=('px', 'nx', 'py', 'ny'))
    box(mb, (x0, x1, y0, y1, 1.0, z1), SL['steel'])
    box(mb, PLATEN, SL['steel'])
    for (cx0, cx1) in COLS:
        box(mb, (cx0, cx1, COL_Y[0], COL_Y[1], 0.0, BEAM[4]), SL['iron'])
        box(mb, (cx0 - 0.5, cx1 + 0.5, COL_Y[0] - 0.5, COL_Y[1] + 0.5, 0.0, 2.0), SL['steel'])

    box(mb, BEAM, SL['iron'])
    box(mb, (BEAM[0], BEAM[1], BEAM[2], BEAM[3], BEAM[5], BEAM[5] + 0.5), SL['hazard'], faces=('ny', 'py', 'px', 'nx'))
    box(mb, CYL, {'pz': SL['steel'], 'nz': SL['steel'], 'px': SL['chrome'], 'nx': SL['chrome'], 'py': SL['chrome'],
                  'ny': SL['chrome']})
    box(mb, (CYL[0] - 0.5, CYL[1] + 0.5, CYL[2] - 0.5, CYL[3] + 0.5, CYL[4], CYL[4] + 1.0), SL['steel'])
    for (hx, hy, hz0, hz1) in HOSES:
        box(mb, (hx - 0.35, hx + 0.35, hy - 0.35, hy + 0.35, hz0, hz1), SL['rubber'])
        xa, xb = (hx, CYL[0]) if hx < 0 else (CYL[1], hx)
        box(mb, (min(xa, xb) - 0.35, max(xa, xb) + 0.35, hy - 0.35, hy + 0.35, hz1 - 0.7, hz1), SL['rubber'])
        box(mb, (hx - 0.6, hx + 0.6, hy - 0.6, hy + 0.6, hz0 - 1.2, hz0), SL['steel'])
    box(mb, (7.0, 10.0, -2.8, -2.5, 12.0, 15.0), SL['gauge'], faces=('ny',))
    v, i = mb.arrays()
    lo = v[:, :3].min(0)
    hi = v[:, :3].max(0)
    return v, i, [(0, len(i), lo, hi)]


def studio_sky(width=512, height=128):
    """A dark backdrop instead of a sky (seen only through gaps and in reflections)."""
    v = np.linspace(0, 1, height)[:, None]
    col = np.array([0.012, 0.012, 0.016]) * (1.0 - 0.5 * v[..., None])
    return np.broadcast_to(col, (height, width, 3)).astype(np.float32).copy()


def tint_noise(n=256):
    v = (np.arange(n) + 0.5) / n
    x, y = np.meshgrid(v, v)
    t = fbm2d(x * 4, y * 4, octaves=4, seed=77, period=4)
    t = (t - t.min()) / (t.max() - t.min())
    return t.astype(np.float32)


def make_renderer(width=1080, height=1920, ss=1.0, shadow_res=4096):
    RD.SUN_DIR = KEY_DIR
    r = RD.Renderer(width, height, ss=ss, shadow_res=shadow_res, near_half=34.0, far_half=80.0)
    layers, _ = PA.studio_textures()
    r.set_textures(layers, studio_sky(), tint_noise(), puff_textures())
    v, i, chunks = build_static()
    r.set_static(v, i, chunks, ground_half=0.01)
    # studio lighting: a warm key light, dim cool ambience, the room falls off into darkness
    r.sun_col = np.array([1.0, 0.93, 0.84]) * 2.6
    r.sky_amb = np.array([0.20, 0.22, 0.30]) * 0.35
    r.gnd_amb = np.array([0.16, 0.13, 0.10]) * 0.3
    r.bounce = (0.05, 0.045, 0.04)
    r.fog = 0.009
    r.fog_col = (0.006, 0.006, 0.009)
    r.exposure = 0.8
    r.contrast = 1.12
    r.vignette = 0.32
    r.bloom_thresh = 2.2
    r.bloom = 0.35
    r.sculk_r = 0.0
    r.glow = 4.0
    return r


def studio_lights(extra=()):
    """Point lights of the room: the two redstone lamps, a cool rim light behind the press, a front fill."""
    rows = []
    for (lx, ly, lz) in LAMPS:
        rows.append([lx, ly - 3.0, lz, 2.6, 30.0, 1.0, 0.55, 0.25])
    rows.append([0.0, 16.0, 30.0, 2.2, 40.0, 0.45, 0.6, 1.0])
    rows.append([2.0, -16.0, 9.0, 5.0, 40.0, 0.92, 0.96, 1.0])       # a soft front fill under the ram
    rows.extend(extra)
    return np.array(rows, np.float32)
