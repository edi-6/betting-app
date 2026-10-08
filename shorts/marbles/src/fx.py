"""Smoke, dust and light: the dust and stone chips where the goat hits the pedestal, the creeper's blast (fireball,
smoke, a flash and a light), and little puffs of dust where the marbles come down on the meadow.

Stepped in story time (slow motion slows it too).
"""
import numpy as np

from creeper import VOXEL_DTYPE
from explosion_vfx import VFX
from vfx import Dust

GRAV = 32.0
STONE = np.array([(124, 123, 126), (104, 104, 108), (140, 140, 142)], float)


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
        out = np.zeros(n, VOXEL_DTYPE)
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


class Effects:
    def __init__(self, seed=9):
        self.rng = np.random.default_rng(seed)
        self.vfx = VFX(seed + 1)
        self.dust = Dust(seed + 2)
        self.bits = Bits()
        self.lights = []                        # [x, y, z, intensity, radius, r, g, b, t0, t1]
        self.t = 0.0

    def ram(self, t, p):
        """The goat's head meets the pedestal at p."""
        rng = self.rng
        p = np.asarray(p, float)
        m = 9
        self.dust._cat(p=p + rng.normal(0, 0.3, (m, 3)) * np.array([1.2, 0.3, 0.6]),
                       v=rng.normal(0, 0.8, (m, 3)) + np.array([0.0, -1.2, 0.6]), age=np.zeros(m),
                       life=rng.uniform(0.9, 1.6, m), s0=rng.uniform(0.35, 0.6, m), s1=rng.uniform(1.2, 1.8, m),
                       var=rng.integers(0, 4, m).astype(float), opa=rng.uniform(0.35, 0.5, m))
        n = 22
        d = rng.normal(0, 1, (n, 3))
        d[:, 1] = -np.abs(d[:, 1]) - 0.6
        d[:, 2] = np.abs(d[:, 2]) + 0.4
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        self.bits.add(p + rng.normal(0, 0.2, (n, 3)), d * rng.uniform(2.5, 6.0, (n, 1)),
                      STONE[rng.integers(3, size=n)], rng.uniform(0.06, 0.14, n), rng)

    def boom(self, t, c):
        """The creeper goes off at c (world)."""
        c = np.asarray(c, float)
        rng = self.rng
        self.vfx.step(np.zeros((0, 3)), 0.0, [(c + np.array([0.0, -1.5, 0.0]), 4.2),
                                             (c + np.array([-5.0, -1.0, 6.0]), 2.6),
                                             (c + np.array([5.0, -1.0, -5.0]), 2.6)])
        self.lights.append([c[0], c[1] - 6.0, c[2], 60.0, 60.0, 1.0, 0.72, 0.38, t, t + 0.9])
        m = 26
        ang = rng.uniform(0, 2 * np.pi, m)
        self.dust._cat(p=c + np.stack([np.cos(ang) * 6, rng.uniform(-3, -1, m), np.sin(ang) * 6], -1),
                       v=np.stack([np.cos(ang) * 6, rng.uniform(-6, -2, m), np.sin(ang) * 6], -1),
                       age=np.zeros(m), life=rng.uniform(2.0, 3.4, m), s0=rng.uniform(2.0, 3.5, m),
                       s1=rng.uniform(6.0, 9.0, m), var=rng.integers(0, 4, m).astype(float),
                       opa=rng.uniform(0.35, 0.55, m))

    def landings(self, pts):
        """Marbles hitting the meadow: a few small puffs (the busiest moments are merged)."""
        if pts is None or not len(pts):
            return
        self.dust.step(np.asarray(pts, float)[:8], 0.0)

    def step(self, dt, t):
        self.t = t
        self.bits.step(dt, lambda x, y: np.zeros(np.shape(x)))
        self.vfx.step(np.zeros((0, 3)), dt)
        self.dust.step(np.zeros((0, 3)), dt)
        self.lights = [l for l in self.lights if l[9] > t]

    def render_data(self):
        lights = []
        for l in self.lights:
            u = float(np.clip((l[9] - self.t) / max(1e-6, l[9] - l[8]), 0.0, 1.0))
            lights.append(l[:3] + [l[3] * u * u] + l[4:8])
        for row in self.vfx.lights()[:3]:
            lights.append(list(row) + [1.0, 0.6, 0.25])
        puffs = np.concatenate([self.dust.puffs(), self.vfx.puffs()])
        return self.bits.instances(), puffs, self.vfx.flashes(), lights
