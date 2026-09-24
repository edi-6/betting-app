"""What moves in the race, for the renderer: the marbles (voxel balls with the racers' faces, rolling as the physics
says, sinking and glowing once they're in lava), and the machinery rebuilt as a mesh every frame: the stalls' floor,
the trapdoors, the pens' floors, the pistons, the TNT, and the animated lava surfaces. Plus the lava's light.
"""
import numpy as np

import blocks as B
import course as C
import heads as H
from mathutil import quat_rotate
from scene import MeshBuilder, box, oriented_box, DEPTH

L = B.L
VOXEL_DTYPE = np.dtype([('pos', 'f4', 3), ('quat', 'f4', 4), ('scale', 'f4'), ('cx', 'u1', 4),
                        ('cy', 'u1', 4), ('cz', 'u1', 4), ('inner', 'u1', 4)])
VS = C.R / 10.0                    # voxel size of a marble
SINK = 0.7                         # how fast an eliminated marble sinks (blocks / s)
BURN = 1.6                         # seconds it takes to burn up


def roll_quat(a):
    """The marble's rotation for the physics angle a (counter-clockwise as the camera sees it: about -y)."""
    return np.array([0.0, -np.sin(a / 2.0), 0.0, np.cos(a / 2.0)])


class Marbles:
    def __init__(self, names, seed=3):
        rng = np.random.default_rng(seed)
        n = int(np.ceil(C.R / VS))
        g = np.arange(-n, n + 1)
        I, J, K = np.meshgrid(g, g, g, indexing='ij')
        ijk = np.stack([I.ravel(), J.ravel(), K.ravel()], -1)
        c = ijk * VS
        d = np.linalg.norm(c, axis=1)
        keep = (d <= C.R) & (d > C.R - 2.2 * VS)
        self.rel = c[keep]
        dirs = self.rel / np.linalg.norm(self.rel, axis=1, keepdims=True)
        # face visibility in the ball's own frame
        occ = set(map(tuple, ijk[keep]))
        vis = np.zeros(keep.sum(), np.uint8)
        for bit, dd in enumerate(((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))):
            nb = ijk[keep] + np.array(dd)
            vis |= np.array([tuple(v) not in occ for v in nb], np.uint8) << bit
        self.vis = vis
        self.names = list(names)
        self.cols = {}
        self.glow = {}
        for name in self.names:
            col, gl = H.ball_colors(name, dirs, rng)
            self.cols[name] = col
            self.glow[name] = gl
        self.n = len(self.rel)

    def instances(self, state, t):
        out = []
        for m in state['marbles']:
            name = m['name']
            col = self.cols[name].astype(np.float32)
            glow = self.glow[name].copy()
            if m['out'] is None:
                p = np.array([m['x'], 0.0, m['z']])
                q = roll_quat(m['a'])
                scale = VS
            else:
                age = t - m['out']
                if age > BURN + 0.3:
                    continue
                x0, z0, vx, a0 = m['sink']
                p = np.array([x0 + vx * 0.08 * min(age, 0.5), 0.0, z0 - SINK * age - 0.4 * min(age, 0.3)])
                q = roll_quat(a0 + 1.5 * age)
                # heats up: the colours go to glowing orange from the bottom
                u = float(np.clip(age / BURN, 0.0, 1.0))
                hot = np.array([255.0, 110.0, 24.0])
                col = col * (1.0 - u) + hot * u
                glow = glow | (u > 0.35)
                scale = VS * (1.0 - 0.3 * u)
            inst = np.zeros(self.n, VOXEL_DTYPE)
            inst['pos'] = p + quat_rotate(np.tile(q, (self.n, 1)), self.rel)
            inst['quat'] = q
            inst['scale'] = scale
            c8 = np.clip(col, 0, 255).astype(np.uint8)
            for k in ('cx', 'cy', 'cz', 'inner'):
                inst[k][:, :3] = c8
            inst['cx'][:, 3] = 0b111111
            inst['cy'][:, 3] = self.vis
            inst['cz'][:, 3] = np.where(glow, 255, 0)
            out.append(inst)
        return np.concatenate(out) if out else np.zeros(0, VOXEL_DTYPE)


# ---------------------------------------------------------------------------------------------
# the machinery
# ---------------------------------------------------------------------------------------------
def _flap(mb, hinge, angle, length, layer, t=0.22):
    """A plate hinged at one end at `angle` (radians, in the x-z plane)."""
    u = np.array([np.cos(angle), np.sin(angle)])
    c = np.asarray(hinge, float) + u * length / 2.0
    oriented_box(mb, c, u, length / 2.0, t / 2.0, DEPTH[0] + 0.05, DEPTH[1], layer)


def dynamic_mesh(course, state, t, lava_t):
    mb = MeshBuilder()
    y0, y1 = DEPTH
    # the stalls' floor: eight little trapdoors that drop at GO
    x0, z0, x1, z1 = course.start['floor']
    go_t = state.get('go_t')
    ang = 0.0
    if go_t is not None:
        ang = -min(1.0, (t - go_t) / 0.12) * np.pi / 2.0
    for k in range(C.N):
        xa = -7.2 + 1.8 * k + 0.12
        _flap(mb, (xa, z1 - 0.12), ang, 1.56, L['trapdoor'])
    # the trapdoors in the course
    for tr, st in zip(course.traps, state['traps']):
        mat = 'gold' if tr['name'].startswith('FINAL') else 'trapdoor'
        _flap(mb, tr['hinge'], st['angle'], tr['length'], L[mat], t=0.5)
    # the pens: a floor that splits in two and drops open at the end of the round
    for pn, rel in zip(course.pens, state['pens']):
        a, b = np.array(pn['a']), np.array(pn['b'])
        half = np.linalg.norm(b - a) / 2.0
        base = np.arctan2(b[1] - a[1], b[0] - a[0])
        s = 0.0 if rel is None else min(1.0, (t - rel) / 0.14)
        lay = L.get(pn['mat'], L['planks'])
        _flap(mb, a, base - s * np.pi / 2.0, half, lay, 0.5)
        _flap(mb, b, base + np.pi + s * np.pi / 2.0, half, lay, 0.5)
    # pistons: a piston block in the wall and its head on an arm
    for p, e in zip(course.pistons, state['pistons']):
        d = p['dir']
        xw = p['x']
        head_x = xw + d * (0.5 + p['reach'] * e)
        box(mb, min(xw, xw - d * 1.0), max(xw, xw - d * 1.0), y0, y1, p['z'] - 0.5, p['z'] + 0.5,
            {'px': L['piston_head'], 'nx': L['piston_head'], 'ny': L['piston_side'], 'pz': L['piston_side'],
             'nz': L['piston_side']})
        hx0, hx1 = sorted((head_x - 0.5 * d, head_x + 0.5 * d))
        if e > 0.01:
            box(mb, hx0, hx1, y0, y1, p['z'] - 0.5, p['z'] + 0.5, L['piston_head'])
            ax0, ax1 = sorted((xw, head_x - 0.5 * d))
            box(mb, ax0, ax1, -0.25, 0.25, p['z'] - 0.2, p['z'] + 0.2, L['planks'])
    # TNT: flashes white once lit, gone when it blows
    for tn, st in zip(course.tnt, state['tnt']):
        if st['boom'] is not None:
            continue
        lay = {'ny': L['tnt_side'], 'px': L['tnt_side'], 'nx': L['tnt_side'], 'pz': L['tnt_top'], 'nz': L['tnt_top']}
        sw = 1.0
        if st['lit'] is not None:
            a = t - st['lit']
            if int(a / 0.07) % 2 == 0:
                lay = L['white']
            sw = 1.0 + 0.25 * min(1.0, a / 0.3)
        h = 0.5 * sw
        box(mb, tn['x'] - h, tn['x'] + h, -h, h, tn['z'] - h, tn['z'] + h, lay)
    # lava: the surfaces of the basins and of the lake, flowing slowly
    pools = list(course.lava) + [course.lake]
    for (px0, pz0, px1, pz1) in pools:
        top = pz1 - 0.25
        u_off = 0.35 * np.sin(lava_t * 0.6) + 0.12 * lava_t
        p = [(px0, y0, top), (px1, y0, top), (px1, y1, top), (px0, y1, top)]
        uv = [(px0 + u_off, -y0), (px1 + u_off, -y0), (px1 + u_off, -y1), (px0 + u_off, -y1)]
        mb.quad(p, (0, 0, 1), uv, L['lava'])
        p = [(px0, y0, pz0), (px1, y0, pz0), (px1, y0, top), (px0, y0, top)]
        uv = [(px0 - u_off * 0.5, -pz0 + lava_t * 0.3), (px1 - u_off * 0.5, -pz0 + lava_t * 0.3),
              (px1 - u_off * 0.5, -top + lava_t * 0.3), (px0 - u_off * 0.5, -top + lava_t * 0.3)]
        mb.quad(p, (0, -1, 0), uv, L['lava'])
    return mb.arrays()


def lava_lights(course, t):
    """Warm, flickering point lights over the lava (rows of x y z intensity radius r g b)."""
    rows = []
    for k, (px0, pz0, px1, pz1) in enumerate(course.lava):
        fl = 1.0 + 0.12 * np.sin(t * 7.3 + k) + 0.08 * np.sin(t * 13.1 + 2 * k)
        rows.append([(px0 + px1) / 2.0, -1.2, pz1 + 0.8, 4.0 * fl, 9.0, 1.0, 0.45, 0.12])
    lx0, lz0, lx1, lz1 = course.lake
    for x in (-10.0, 0.0, 10.0):
        rows.append([x, -2.0, lz1 + 1.0, 6.0, 16.0, 1.0, 0.45, 0.12])
    return rows
