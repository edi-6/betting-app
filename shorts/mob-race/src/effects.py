"""Particles for the race, driven by its events and stepped in race time (so slow motion slows them too): lava
splashes, smoke and embers when a marble goes into the lava, the TNT blast, dust from the pistons, gold sparkles when
a marble gets over a trapdoor as it drops, and confetti and fireworks for the winner.
"""
import numpy as np

import props as PR
from explosion_vfx import VFX
from vfx import Dust

GRAV = 30.0
CONFETTI = np.array([(255, 70, 70), (255, 200, 40), (70, 200, 255), (120, 230, 90), (240, 110, 255),
                     (255, 255, 255)], float)


class Bits:
    """Small cubes: lava drops, embers, sparkles, confetti. Glowing ones light themselves."""

    def __init__(self):
        self.p = np.zeros((0, 3))
        self.v = np.zeros((0, 3))
        self.col = np.zeros((0, 3))
        self.size = np.zeros(0)
        self.life = np.zeros(0)
        self.age = np.zeros(0)
        self.glow = np.zeros(0, bool)
        self.grav = np.zeros(0)
        self.drag = np.zeros(0)
        self.spin = np.zeros((0, 4))

    def add(self, p, v, col, size, life, glow=False, grav=1.0, drag=0.5, rng=None):
        n = len(p)
        if n == 0:
            return
        rng = rng or np.random.default_rng(0)
        self.p = np.concatenate([self.p, p])
        self.v = np.concatenate([self.v, v])
        self.col = np.concatenate([self.col, np.broadcast_to(np.asarray(col, float), (n, 3))])
        self.size = np.concatenate([self.size, np.broadcast_to(size, (n,))])
        self.life = np.concatenate([self.life, np.broadcast_to(life, (n,))])
        self.age = np.concatenate([self.age, np.zeros(n)])
        self.glow = np.concatenate([self.glow, np.broadcast_to(glow, (n,))])
        self.grav = np.concatenate([self.grav, np.broadcast_to(grav, (n,))])
        self.drag = np.concatenate([self.drag, np.broadcast_to(drag, (n,))])
        q = rng.normal(0, 1, (n, 4))
        self.spin = np.concatenate([self.spin, q / np.linalg.norm(q, axis=1, keepdims=True)])

    def step(self, dt):
        if len(self.p) == 0 or dt <= 0:
            return
        self.age += dt
        keep = self.age < self.life
        for a in ('p', 'v', 'col', 'size', 'life', 'age', 'glow', 'grav', 'drag', 'spin'):
            setattr(self, a, getattr(self, a)[keep])
        self.v[:, 2] -= GRAV * self.grav * dt
        self.v *= (1.0 - self.drag[:, None] * dt)
        self.p += self.v * dt

    def instances(self):
        n = len(self.p)
        out = np.zeros(n, PR.VOXEL_DTYPE)
        if n == 0:
            return out
        u = self.age / self.life
        out['pos'] = self.p
        out['quat'] = self.spin
        out['scale'] = self.size * np.clip(1.0 - u, 0.0, 1.0) ** 0.5
        c = np.clip(self.col, 0, 255).astype(np.uint8)
        for k in ('cx', 'cy', 'cz', 'inner'):
            out[k][:, :3] = c
        out['cx'][:, 3] = 0b111111
        out['cy'][:, 3] = 0b111111
        out['cz'][:, 3] = np.where(self.glow, 255, 0)
        return out


class Effects:
    def __init__(self, course, seed=9):
        self.c = course
        self.rng = np.random.default_rng(seed)
        self.bits = Bits()
        self.vfx = VFX(seed + 1)
        self.dust = Dust(seed + 2)
        self.lights = []                # [x, y, z, intensity, radius, r, g, b, t0, t1]
        self.t = 0.0
        self.burning = {}               # name -> (x, z, t_in)

    def event(self, e, state):
        rng = self.rng
        t = state['t']
        if e[0] == 'lava':
            name, (x, z) = e[1], e[2]
            self.burning[name] = (x, z, t)
            n = 70
            d = rng.normal(0, 1, (n, 3))
            d[:, 2] = np.abs(d[:, 2]) * 2.2 + 0.8
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            p = np.array([x, -0.2, z + 0.3]) + rng.normal(0, 0.25, (n, 3)) * np.array([1.0, 0.4, 0.3])
            cols = np.where(rng.random((n, 1)) < 0.5, np.array([255, 150, 30]), np.array([255, 90, 10]))
            self.bits.add(p, d * rng.uniform(4.0, 10.0, (n, 1)), cols, rng.uniform(0.07, 0.16, n),
                          rng.uniform(0.5, 1.1, n), glow=True, grav=1.0, drag=0.4, rng=rng)
            self.vfx.step(np.zeros((0, 3)), 0.0, [(np.array([x, -0.3, z + 0.8]), 0.35)])
            self.lights.append([x, -1.5, z + 1.0, 7.0, 7.0, 1.0, 0.5, 0.15, t, t + 0.6])
        elif e[0] == 'boom':
            x, z = e[2]
            c = np.array([x, -0.2, z])
            self.vfx.step(np.array([c]), 0.0, [(c, 0.55)])
            n = 90
            d = rng.normal(0, 1, (n, 3))
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            cols = np.where(rng.random((n, 1)) < 0.55, np.array([220, 40, 30]), np.array([240, 236, 226]))
            self.bits.add(c + d * 0.4, d * rng.uniform(6.0, 16.0, (n, 1)), cols, rng.uniform(0.1, 0.2, n),
                          rng.uniform(0.6, 1.2, n), glow=False, grav=1.0, drag=0.6, rng=rng)
            self.lights.append([x, -2.0, z + 0.5, 16.0, 14.0, 1.0, 0.65, 0.3, t, t + 0.45])
        elif e[0] == 'piston':
            p = self.c.pistons[e[1]]
            self.dust.step(np.array([[p['x'] + p['dir'] * 1.2, -0.3, p['z'] - 0.4]]), 0.0)
        elif e[0] == 'saved':
            name = e[2]
            m = [m for m in state['marbles'] if m['name'] == name][0]
            self.sparkle(m['x'], m['z'], 60, (255, 220, 90))
        elif e[0] == 'release':
            pn = self.c.pens[e[1]]
            a, b = np.array(pn['a']), np.array(pn['b'])
            pts = [np.array([a[0] + (b[0] - a[0]) * u, -0.3, a[1] + (b[1] - a[1]) * u - 0.2])
                   for u in np.linspace(0.1, 0.9, 5)]
            self.dust.step(np.array(pts), 0.0)

    def sparkle(self, x, z, n, col):
        rng = self.rng
        d = rng.normal(0, 1, (n, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        self.bits.add(np.array([x, -0.8, z]) + d * 0.7, d * rng.uniform(2.0, 6.0, (n, 1)), col,
                      rng.uniform(0.06, 0.12, n), rng.uniform(0.5, 1.0, n), glow=True, grav=0.1, drag=1.5, rng=rng)

    def confetti(self, x, z, n=160):
        rng = self.rng
        d = rng.normal(0, 1, (n, 3))
        d[:, 2] = np.abs(d[:, 2]) * 1.5 + 1.0
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        cols = CONFETTI[rng.integers(0, len(CONFETTI), n)]
        self.bits.add(np.array([x, -1.0, z + 0.5]) + rng.normal(0, 0.3, (n, 3)), d * rng.uniform(6.0, 14.0, (n, 1)),
                      cols, rng.uniform(0.08, 0.14, n), rng.uniform(1.6, 2.6, n), glow=False, grav=0.35, drag=1.2,
                      rng=rng)

    def firework(self, x, z, col):
        rng = self.rng
        n = 120
        d = rng.normal(0, 1, (n, 3))
        d[:, 1] *= 0.3
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        self.bits.add(np.tile([x, -2.0, z], (n, 1)), d * rng.uniform(7.0, 9.0, (n, 1)), col,
                      rng.uniform(0.07, 0.11, n), rng.uniform(0.8, 1.3, n), glow=True, grav=0.25, drag=1.4, rng=rng)
        self.lights.append([x, -3.0, z, 5.0, 14.0, col[0] / 255.0, col[1] / 255.0, col[2] / 255.0, self.t,
                            self.t + 0.4])

    def step(self, dt, state):
        self.t = state['t']
        rng = self.rng
        # burning marbles: smoke and embers while they sink
        for name, (x, z, t_in) in list(self.burning.items()):
            age = self.t - t_in
            if age > PR.BURN + 0.4:
                del self.burning[name]
                continue
            k = int(rng.poisson(dt * 60.0))
            if k:
                p = np.array([x, -0.3, z + 0.6]) + rng.normal(0, 0.3, (k, 3)) * np.array([1.0, 0.3, 0.2])
                v = np.stack([rng.normal(0, 0.6, k), rng.normal(0, 0.2, k), rng.uniform(1.5, 4.0, k)], -1)
                self.bits.add(p, v, (255, 140, 30), rng.uniform(0.05, 0.1, k), rng.uniform(0.4, 0.9, k),
                              glow=True, grav=-0.05, drag=0.8, rng=rng)
            if rng.random() < dt * 8.0:
                self.dust.step(np.array([[x + rng.normal(0, 0.3), -0.3, z + 0.9]]), 0.0)
        self.bits.step(dt)
        self.vfx.step(np.zeros((0, 3)), dt)
        self.dust.step(np.zeros((0, 3)), dt)
        self.lights = [l for l in self.lights if l[9] > self.t]

    def render_data(self):
        lights = []
        for l in self.lights:
            u = float(np.clip((l[9] - self.t) / max(1e-6, l[9] - l[8]), 0.0, 1.0))
            lights.append(l[:3] + [l[3] * u * u] + l[4:8])
        for row in self.vfx.lights()[:3]:
            lights.append(list(row) + [1.0, 0.6, 0.25])
        puffs = np.concatenate([self.dust.puffs(), self.vfx.puffs()])
        return self.bits.instances(), puffs, self.vfx.flashes(), lights
