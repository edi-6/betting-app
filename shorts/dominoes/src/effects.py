"""The creeper's blast and what it throws: a fireball and smoke (explosion_vfx), a flash and a light, grass and dirt
thrown out of the crater (small cubes), dust, and the dominoes lying closest to it flung into the air (rigid bodies
that spin, bounce and come to rest lying flat). Also the crater itself: the patch of ground around the creeper is
drawn separately so its blocks can go missing.

Stepped in simulation time (so the slow motion slows it too).
"""
import numpy as np

import creeper as CR
import layout as LY
import world as WD
from explosion_vfx import VFX
from mathutil import quat_rotate
from vfx import Dust

GRAV = 32.0
DIRT = np.array([(136, 96, 64), (118, 84, 56), (150, 108, 72)], float)
GRASS = np.array([(96, 150, 52), (84, 136, 46), (104, 160, 60)], float)
PATCH = WD.PATCH


class Bits:
    """Small cubes (dirt, grass, sparks)."""

    def __init__(self):
        self.p = np.zeros((0, 3))
        self.v = np.zeros((0, 3))
        self.col = np.zeros((0, 3))
        self.size = np.zeros(0)
        self.q = np.zeros((0, 4))
        self.w = np.zeros((0, 3))
        self.rest = np.zeros(0, bool)

    def add(self, p, v, col, size, rng):
        n = len(p)
        q = rng.normal(0, 1, (n, 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        self.p = np.concatenate([self.p, p])
        self.v = np.concatenate([self.v, v])
        self.col = np.concatenate([self.col, col])
        self.size = np.concatenate([self.size, size])
        self.q = np.concatenate([self.q, q])
        self.w = np.concatenate([self.w, rng.normal(0, 9.0, (n, 3))])
        self.rest = np.concatenate([self.rest, np.zeros(n, bool)])

    def step(self, dt, ground):
        if not len(self.p) or dt <= 0:
            return
        m = ~self.rest
        self.v[m, 2] -= GRAV * dt
        self.v[m] *= (1.0 - 0.3 * dt)
        self.p[m] += self.v[m] * dt
        # spin: integrate orientation with the angular velocity
        wq = np.concatenate([self.w[m] * dt * 0.5, np.zeros((m.sum(), 1))], 1)
        self.q[m] = _qnorm(self.q[m] + _qmul_arr(wq, self.q[m]))
        g = ground(self.p[:, 0], self.p[:, 1]) + self.size / 2
        hit = m & (self.p[:, 2] < g)
        self.p[hit, 2] = g[hit]
        self.v[hit, 2] *= -0.3
        self.v[hit, :2] *= 0.55
        self.w[hit] *= 0.5
        slow = hit & (np.abs(self.v[:, 2]) < 1.2)
        self.rest |= slow
        self.v[slow] = 0.0

    def instances(self):
        n = len(self.p)
        out = np.zeros(n, CR.VOXEL_DTYPE)
        if not n:
            return out
        out['pos'] = self.p
        out['quat'] = self.q
        out['scale'] = self.size
        c = np.clip(self.col, 0, 255).astype(np.uint8)
        for k in ('cx', 'cy', 'cz', 'inner'):
            out[k][:, :3] = c
        out['cx'][:, 3] = 0b111111
        out['cy'][:, 3] = 0b111111
        out['cz'][:, 3] = 0
        return out


def _qmul_arr(a, b):
    ax, ay, az, aw = a.T
    bx, by, bz, bw = b.T
    return np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)


def _qnorm(q):
    return q / np.linalg.norm(q, axis=1, keepdims=True)


class Flung:
    """Dominoes thrown by the blast: rigid bodies (a box T x W x H) that tumble, bounce and settle flat."""

    def __init__(self, idx, pos, quat, vel, spin):
        self.idx = np.asarray(idx)
        self.p = np.asarray(pos, float)
        self.q = np.asarray(quat, float)
        self.v = np.asarray(vel, float)
        self.w = np.asarray(spin, float)
        self.rest = np.zeros(len(self.idx), bool)

    def step(self, dt):
        m = ~self.rest
        if not m.any() or dt <= 0:
            return
        self.v[m, 2] -= GRAV * dt
        self.p[m] += self.v[m] * dt
        wq = np.concatenate([self.w[m] * dt * 0.5, np.zeros((m.sum(), 1))], 1)
        self.q[m] = _qnorm(self.q[m] + _qmul_arr(wq, self.q[m]))
        # lowest point of the box over the ground (z = 0)
        corners = np.array([(sx * LY.T / 2, sy * LY.W / 2, sz * LY.H / 2) for sx in (-1, 1) for sy in (-1, 1)
                            for sz in (-1, 1)])
        for i in np.nonzero(m)[0]:
            c = quat_rotate(np.tile(self.q[i], (8, 1)), corners)
            low = self.p[i, 2] + c[:, 2].min()
            if low < 0.0:
                self.p[i, 2] -= low
                if self.v[i, 2] < 0:
                    self.v[i, 2] *= -0.25
                self.v[i, :2] *= 0.6
                self.w[i] *= 0.55
                if abs(self.v[i, 2]) < 1.0 and np.linalg.norm(self.w[i]) < 3.0:
                    # settle: lie flat on its big face, keeping its heading
                    self.q[i] = _flat(self.q[i])
                    self.p[i, 2] = LY.T / 2
                    self.rest[i] = True


def _flat(q):
    """The orientation of a domino lying on one of its big faces, nearest to q."""
    x = quat_rotate(q[None], np.array([[1.0, 0.0, 0.0]]))[0]        # its thickness axis
    up = np.array([0.0, 0.0, 1.0]) * np.sign(x[2] if abs(x[2]) > 1e-6 else 1.0)
    # rotate so the thickness axis points straight up (or down)
    axis = np.cross(x, up)
    s = np.linalg.norm(axis)
    if s < 1e-6:
        return q
    ang = np.arcsin(np.clip(s, -1, 1))
    if np.dot(x, up) < 0:
        ang = np.pi - ang
    r = np.array([*(axis / s * np.sin(ang / 2)), np.cos(ang / 2)])
    return _qnorm(_qmul_arr(r[None], q[None]))[0]


class Effects:
    def __init__(self, seed=9):
        self.rng = np.random.default_rng(seed)
        self.vfx = VFX(seed + 1)
        self.dust = Dust(seed + 2)
        self.bits = Bits()
        self.flung = None
        self.lights = []                        # [x, y, z, intensity, radius, r, g, b, t0, t1]
        self.t = 0.0
        self.crater = None

    def blast(self, t, c, dominoes, poses, idx_near):
        """The creeper goes off at c (world). dominoes: props.Dominoes; poses: their poses now; idx_near: the
        indices of the dominoes it throws."""
        rng = self.rng
        c = np.asarray(c, float)
        self.vfx.step(np.array([c]), 0.0, [(c, 0.6)])
        self.lights.append([c[0], c[1], c[2] + 1.0, 16.0, 14.0, 1.0, 0.7, 0.35, t, t + 0.45])
        # the crater: blocks within reach of the blast are gone
        self.crater = (c[0], c[1], 1.7)
        n = 70
        d = rng.normal(0, 1, (n, 3))
        d[:, 2] = np.abs(d[:, 2]) * 1.6 + 0.6
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        p = np.array([c[0], c[1], -0.3]) + rng.normal(0, 0.5, (n, 3)) * np.array([1, 1, 0.3])
        grass = rng.random(n) < 0.35
        col = np.where(grass[:, None], GRASS[rng.integers(3, size=n)], DIRT[rng.integers(3, size=n)])
        self.bits.add(p, d * rng.uniform(6.0, 15.0, (n, 1)), col, rng.uniform(0.12, 0.3, n), rng)
        self.dust.step(np.array([c + rng.normal(0, 0.6, 3) * [1, 1, 0] for _ in range(10)]), 0.0)
        # fling the dominoes lying closest
        A = dominoes.A
        pos = np.stack([A['x'][idx_near] + dominoes.cos[idx_near] * poses[idx_near, 0],
                        A['y'][idx_near] + dominoes.sin[idx_near] * poses[idx_near, 0], poses[idx_near, 1]], -1)
        from props import quat_yaw_pitch
        q = quat_yaw_pitch(A['yaw'][idx_near].astype(float), poses[idx_near, 2])
        away = pos - c
        away[:, 2] = np.abs(away[:, 2]) + 0.8
        dist = np.linalg.norm(away, axis=1, keepdims=True)
        v = away / dist * rng.uniform(9.0, 14.0, (len(idx_near), 1)) * np.clip(2.2 / dist, 0.6, 1.4)
        w = rng.normal(0, 14.0, (len(idx_near), 3))
        self.flung = Flung(idx_near, pos, q, v, w)

    def step(self, dt, t):
        self.t = t
        self.bits.step(dt, lambda x, y: self.ground_height(x, y))
        if self.flung is not None:
            self.flung.step(dt)
        self.vfx.step(np.zeros((0, 3)), dt)
        self.dust.step(np.zeros((0, 3)), dt)
        self.lights = [l for l in self.lights if l[9] > t]

    def ground_height(self, x, y):
        h = np.zeros(np.shape(x))
        if self.crater is not None:
            cx, cy, r = self.crater
            d = np.hypot(np.asarray(x) - cx, np.asarray(y) - cy)
            h = np.where(d < r, -1.0, 0.0) + np.where(d < r * 0.5, -1.0, 0.0)
        return h

    def apply(self, dominoes):
        """Put the flung dominoes' poses into the domino instances."""
        dominoes.override = {}
        if self.flung is not None:
            for k, i in enumerate(self.flung.idx):
                dominoes.override[int(i)] = (self.flung.p[k].copy(), self.flung.q[k].copy())

    def render_data(self):
        lights = []
        for l in self.lights:
            u = float(np.clip((l[9] - self.t) / max(1e-6, l[9] - l[8]), 0.0, 1.0))
            lights.append(l[:3] + [l[3] * u * u] + l[4:8])
        for row in self.vfx.lights()[:3]:
            lights.append(list(row) + [1.0, 0.6, 0.25])
        puffs = np.concatenate([self.dust.puffs(), self.vfx.puffs()])
        return self.bits.instances(), puffs, self.vfx.flashes(), lights


def ground_patch(crater=None):
    """The patch of meadow around the creeper's spot as a mesh; with crater = (x, y, r): blocks within r of it are
    gone from the top layer, and within r/2 from the one under it too."""
    mb = WD.MeshBuilder()
    x0, y0, x1, y1 = PATCH
    L = WD.L

    def gone(ix, iy, level):
        if crater is None:
            return False
        cx, cy, r = crater
        d = np.hypot(ix + 0.5 - cx, iy + 0.5 - cy)
        return d < (r if level == 0 else r * 0.5)

    for ix in range(x0, x1):
        for iy in range(y0, y1):
            top = 0 if not gone(ix, iy, 0) else (-1 if not gone(ix, iy, 1) else -2)
            lay = L['grass_top'] if top == 0 else (L['crater'] if top == -1 else L['dirt'])
            WD.box_face(mb, ix, iy, top - 1, ix + 1, iy + 1, top, 'pz', lay)
            # walls of the hole
            if top < 0:
                for face, (dx, dy) in (('px', (1, 0)), ('nx', (-1, 0)), ('py', (0, 1)), ('ny', (0, -1))):
                    jx, jy = ix + dx, iy + dy
                    ntop = 0 if not gone(jx, jy, 0) else (-1 if not gone(jx, jy, 1) else -2)
                    if ntop > top:
                        # the neighbour's side, seen from inside the hole
                        opp = {'px': 'nx', 'nx': 'px', 'py': 'ny', 'ny': 'py'}[face]
                        for z in range(top, ntop):
                            lay2 = L['grass_side'] if z == -1 and ntop == 0 else L['dirt']
                            WD.box_face(mb, jx, jy, z, jx + 1, jy + 1, z + 1, opp, lay2)
    return mb.arrays()
