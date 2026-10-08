"""One round of the tsunami: the shallow-water flood (swe.py) over the coast, the village standing in its way block
by block, and what it carries off.

  * Every block of the village feels the water pushing on its exposed sides (depth times speed squared, plus the
    hydrostatic push) and breaks loose when that beats its strength (glass first, then planks, logs, cobblestone,
    stone bricks). Whatever is left hanging (a roof without walls, a tree without its trunk) comes down with it.
    The standing blocks are part of the bed the water flows around; as they go, the water pours through.
  * Loose blocks are debris: carried by the current, floating (wood) or sinking (stone), tumbling.
  * The villagers potter about, run for it when the wave comes, and get swept away; the iron golem stands its ground.
  * Spray: mist thrown up where the wave front rises and where it smashes into things.

Recorded at 30 frames per second of simulation time into ../cache/round{n}/.

python sim.py N        # simulate round N (1, 2, 3; 4 is the shallow-water version of the 1,000-block wave, which
                       # slumps into a long ramp: the video draws that one instead, see wall.py)
"""
import os
import sys
import time

import numpy as np
from scipy import ndimage

import blocks as BL
import swe
import village as VL
import water as WT
import world as WD

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, '..', 'cache')
VERSION = 1
REC_HZ = 30
G = 9.81

# per round: wave height, cell size, the simulated box, how the wave builds and holds, how long to run
ROUNDS = {
    1: dict(A=1.0, dx=0.5, box=(-56.0, 56.0, -44.0, 116.0), ramp=1.6, hold=3.5, fall=3.0, t_end=24.0, village=True),
    2: dict(A=10.0, dx=1.0, box=(-112.0, 112.0, -132.0, 228.0), ramp=2.0, hold=3.0, fall=6.0, t_end=19.0,
            village=True),
    3: dict(A=100.0, dx=2.0, box=(-320.0, 320.0, -420.0, 520.0), ramp=2.0, hold=30.0, fall=6.0, t_end=15.0,
            village=True),
    4: dict(A=1000.0, dx=24.0, box=(-4800.0, 4800.0, -5400.0, 4200.0), ramp=3.0, hold=60.0, fall=6.0, t_end=24.0,
            village=False),
}
LOAD_SCALE = 950.0          # a block of strength 1 breaks under this much load (m^3/s^2 per unit width)


def wave_fn(cfg):
    A, ramp, hold, fall = cfg['A'], cfg['ramp'], cfg['hold'], cfg['fall']

    def f(t):
        k = np.clip(t / ramp, 0.0, 1.0)
        up = k * k * (3 - 2 * k)
        d = np.clip((t - hold) / fall, 0.0, 1.0)
        down = 1.0 - d * d * (3 - 2 * d)
        return A * up * (0.15 + 0.85 * down) if t > hold else A * up
    return f


def column_runs(K, H):
    """For every village column: how many solid blocks stand in an unbroken run from the ground up."""
    nz, ny, nx = K.shape
    xs = np.arange(nx) + VL.ORIGIN[0]
    ys = np.arange(ny) + VL.ORIGIN[1]
    ix = np.clip(xs - WD.X0, 0, WD.NX - 1)
    iy = np.clip(ys - WD.Y0, 0, WD.NY - 1)
    ground = H[iy][:, ix]                                       # (ny, nx) terrain top under each column
    k0 = ground - VL.ORIGIN[2]                                  # index of the first block above the terrain
    solid = K > 0
    run = np.zeros((ny, nx), np.int32)
    alive = np.ones((ny, nx), bool)
    for k in range(nz):
        m = (k >= k0)
        cur = solid[k] & m
        alive &= ~m | cur
        run += (alive & m).astype(np.int32)
    return run, ground


class Debris:
    def __init__(self):
        self.p = np.zeros((0, 3), np.float32)
        self.v = np.zeros((0, 3), np.float32)
        self.q = np.zeros((0, 4), np.float32)
        self.w = np.zeros((0, 3), np.float32)
        self.kind = np.zeros(0, np.int32)
        self.dens = np.zeros(0, np.float32)
        self.size = np.zeros(0, np.float32)

    def add(self, p, v, kind, rng, size=1.0):
        n = len(p)
        if n == 0:
            return
        q = rng.normal(size=(n, 4)).astype(np.float32) * np.array([0.08, 0.08, 0.08, 1.0], np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        self.p = np.concatenate([self.p, p.astype(np.float32)])
        self.v = np.concatenate([self.v, v.astype(np.float32)])
        self.q = np.concatenate([self.q, q])
        self.w = np.concatenate([self.w, rng.normal(0, 2.0, (n, 3)).astype(np.float32)])
        self.kind = np.concatenate([self.kind, kind.astype(np.int32)])
        self.dens = np.concatenate([self.dens, BL.DENSITY[kind].astype(np.float32)])
        self.size = np.concatenate([self.size, np.full(n, size, np.float32)])

    def step(self, dt, sample, bed_at, rng):
        if len(self.p) == 0:
            return
        eta, h, u, v = sample(self.p[:, 0], self.p[:, 1])
        half = 0.5 * self.size
        sub = np.clip((eta - (self.p[:, 2] - half)) / self.size, 0.0, 1.0) * (h > 0.05)
        acc = np.zeros_like(self.p)
        acc[:, 2] = -G + G * sub / np.maximum(self.dens, 0.05) * 0.95
        flow = np.stack([u, v], 1)
        acc[:, :2] += (flow - self.v[:, :2]) * (3.2 * sub)[:, None]
        acc[:, 2] += -self.v[:, 2] * 2.2 * sub
        acc[:, :2] += -self.v[:, :2] * 0.03 * (1 - sub)[:, None]
        self.v += acc * dt
        self.p += self.v * dt
        floor = bed_at(self.p[:, 0], self.p[:, 1]) + half
        low = self.p[:, 2] < floor
        if np.any(low):
            self.p[low, 2] = floor[low]
            self.v[low, 2] = np.abs(self.v[low, 2]) * 0.25
            self.v[low, :2] *= 0.86
            self.w[low] *= 0.8
        # tumbling: the current twists them, the water damps the spin
        rel = np.linalg.norm(flow - self.v[:, :2], axis=1)
        self.w += rng.normal(0, 1.0, self.w.shape).astype(np.float32) * (rel * sub * 0.9 * dt)[:, None] * 6.0
        self.w *= np.exp(-dt * (0.4 + 1.2 * sub))[:, None]
        wq = np.concatenate([self.w, np.zeros((len(self.w), 1), np.float32)], 1)
        self.q += 0.5 * dt * qmul(wq, self.q)
        self.q /= np.linalg.norm(self.q, axis=1, keepdims=True)


def qmul(a, b):
    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], 1).astype(np.float32)


class Spray:
    """Mist puffs: position, velocity, size, age, life."""

    def __init__(self, cap=2600):
        self.cap = cap
        self.p = np.zeros((0, 3), np.float32)
        self.v = np.zeros((0, 3), np.float32)
        self.s = np.zeros(0, np.float32)
        self.age = np.zeros(0, np.float32)
        self.life = np.zeros(0, np.float32)
        self.var = np.zeros(0, np.float32)

    def add(self, p, v, size, life, rng):
        n = len(p)
        if n == 0:
            return
        room = self.cap - len(self.p)
        if room <= 0:
            return
        if n > room:
            sel = rng.choice(n, room, replace=False)
            p, v, size, life = p[sel], v[sel], size[sel], life[sel]
            n = room
        self.p = np.concatenate([self.p, p.astype(np.float32)])
        self.v = np.concatenate([self.v, v.astype(np.float32)])
        self.s = np.concatenate([self.s, size.astype(np.float32)])
        self.age = np.concatenate([self.age, np.zeros(n, np.float32)])
        self.life = np.concatenate([self.life, life.astype(np.float32)])
        self.var = np.concatenate([self.var, rng.integers(0, 4, n).astype(np.float32)])

    def step(self, dt):
        if len(self.p) == 0:
            return
        self.v[:, 2] -= 2.5 * dt
        self.v *= np.exp(-dt * 0.9)
        self.p += self.v * dt
        self.age += dt
        keep = self.age < self.life
        for k in ('p', 'v', 's', 'age', 'life', 'var'):
            setattr(self, k, getattr(self, k)[keep])

    def snapshot(self):
        """(n, 6): x, y, z, size, alpha, variant."""
        if len(self.p) == 0:
            return np.zeros((0, 6), np.float32)
        u = self.age / self.life
        size = self.s * (1.0 + 2.2 * u)
        alpha = np.clip(u / 0.12, 0, 1) * (1.0 - u) ** 1.5
        return np.concatenate([self.p, size[:, None], alpha[:, None], self.var[:, None]], 1).astype(np.float32)


class Agents:
    """The villagers and the iron golem: where they are and what they're doing."""
    IDLE, RUN, SWEPT = 0, 1, 2

    def __init__(self, vil, rng):
        spots = list(vil.villagers) + [vil.golem]
        self.n = len(spots)
        self.golem = self.n - 1
        self.p = np.array([s[:3] for s in spots], np.float32)
        self.yaw = np.array([s[3] for s in spots], np.float32)
        self.v = np.zeros((self.n, 3), np.float32)
        self.state = np.zeros(self.n, np.int32)
        self.t_state = np.zeros(self.n, np.float32)
        self.q = np.tile(np.array([0, 0, 0, 1], np.float32), (self.n, 1))
        self.w = np.zeros((self.n, 3), np.float32)
        self.dens = np.full(self.n, 0.92, np.float32)
        self.dens[self.golem] = 1.6
        self.speed = rng.uniform(3.6, 4.6, self.n).astype(np.float32)
        self.react = rng.uniform(0.0, 0.5, self.n).astype(np.float32)

    def step(self, dt, t, sample, ground_at, cfg, rng):
        A = cfg['A']
        eta, h, u, v = sample(self.p[:, 0], self.p[:, 1])
        # the wave is coming: look out to sea
        eta_sea, h_sea, _, _ = sample(self.p[:, 0], self.p[:, 1] + 30.0 + 0.4 * A)
        alarm = (eta_sea > 0.35 * A) & (h_sea > 0.2)
        for i in range(self.n):
            if self.state[i] == self.IDLE and alarm[i] and i != self.golem and A >= 5:
                self.state[i] = self.RUN
                self.t_state[i] = t
            deep = h[i] > (0.55 if i != self.golem else 1.4)
            fast = np.hypot(u[i], v[i]) > (2.0 if i != self.golem else 5.0)
            if self.state[i] != self.SWEPT and deep and fast:
                self.state[i] = self.SWEPT
                self.t_state[i] = t
                self.v[i] = (u[i] * 0.7, v[i] * 0.7, 1.5)
                self.w[i] = rng.normal(0, 3.0, 3)
            if self.state[i] == self.RUN and t - self.t_state[i] > self.react[i]:
                target_yaw = np.pi                                       # face inland (-y)
                self.yaw[i] += (target_yaw - self.yaw[i]) * min(1.0, dt * 6.0)
                self.p[i, 1] -= self.speed[i] * dt
                self.p[i, 0] += np.sin(t * 1.3 + i) * 0.4 * dt
                self.p[i, 2] = ground_at(self.p[i:i + 1, 0], self.p[i:i + 1, 1])[0]
        sw = self.state == self.SWEPT
        if np.any(sw):
            sub = np.clip((eta[sw] - (self.p[sw, 2])) / 1.8, 0.0, 1.0)
            acc = np.zeros((sw.sum(), 3), np.float32)
            acc[:, 2] = -G + G * sub / self.dens[sw] * 0.95
            acc[:, :2] += (np.stack([u[sw], v[sw]], 1) - self.v[sw, :2]) * (2.8 * sub)[:, None]
            acc[:, 2] += -self.v[sw, 2] * 2.0 * sub
            self.v[sw] += acc * dt
            self.p[sw] += self.v[sw] * dt
            floor = ground_at(self.p[sw, 0], self.p[sw, 1])
            pz = self.p[sw, 2]
            low = pz < floor
            pz[low] = floor[low]
            self.p[sw, 2] = pz
            ww = self.w[sw] * np.exp(-dt * 0.5)
            ww += rng.normal(0, 1.0, ww.shape) * dt * 4.0 * sub[:, None]
            self.w[sw] = ww
            wq = np.concatenate([ww, np.zeros((len(ww), 1), np.float32)], 1)
            qq = self.q[sw] + 0.5 * dt * qmul(wq, self.q[sw])
            self.q[sw] = qq / np.linalg.norm(qq, axis=1, keepdims=True)

    def snapshot(self, t):
        """(n, 10): x, y, z, yaw, state, time in state, quat."""
        return np.concatenate([self.p, self.yaw[:, None], self.state[:, None].astype(np.float32),
                               (t - self.t_state)[:, None], self.q], 1).astype(np.float32)


class Round:
    def __init__(self, n, H, vil, seed=7):
        self.n = n
        self.cfg = cfg = ROUNDS[n]
        self.H = H
        self.rng = np.random.default_rng(seed + n)
        x0, x1, y0, y1 = cfg['box']
        dx = cfg['dx']
        self.dx, self.x0, self.y0 = dx, x0, y0
        self.nx, self.ny = int(round((x1 - x0) / dx)), int(round((y1 - y0) / dx))
        xs = x0 + (np.arange(self.nx) + 0.5) * dx
        ys = y0 + (np.arange(self.ny) + 0.5) * dx
        X, Y = np.meshgrid(xs, ys)
        self.X, self.Y = X, Y
        inside = (np.floor(X) >= WD.X0) & (np.floor(X) < WD.X1) & (np.floor(Y) >= WD.Y0) & (np.floor(Y) < WD.Y1)
        ix = np.clip(np.floor(X).astype(int) - WD.X0, 0, WD.NX - 1)
        iy = np.clip(np.floor(Y).astype(int) - WD.Y0, 0, WD.NY - 1)
        if dx <= 1.0:
            terr = H[iy, ix].astype(np.float32)
        else:
            # coarse cells: the mean height of the blocks they cover
            k = int(dx)
            acc = np.zeros_like(X)
            for a in range(k):
                for b in range(k):
                    jx = np.clip(np.floor(X - dx / 2 + a + 0.5).astype(int) - WD.X0, 0, WD.NX - 1)
                    jy = np.clip(np.floor(Y - dx / 2 + b + 0.5).astype(int) - WD.Y0, 0, WD.NY - 1)
                    acc += H[jy, jx]
            terr = (acc / (k * k)).astype(np.float32)
        self.terrain = np.where(inside, terr, WD.far_heights(X, Y)).astype(np.float32)
        self.K = vil.K.copy() if cfg['village'] else np.zeros_like(vil.K)
        self.strength = (BL.STRENGTH[self.K] * self.rng.uniform(0.75, 1.3, self.K.shape)).astype(np.float32)
        self.events = []                 # (t, flat index into K)
        bed = self.terrain + self._structures()
        self.sim = swe.SWE(bed, dx, x0, y0, sea_level=0.0)
        self.sim.wave = wave_fn(cfg)
        self.sim.foam_scale = max(1.0, cfg['A'] / 10.0)
        self.sim.umax = 18.0 * np.sqrt(max(cfg['A'], 1.0))
        self.debris = Debris()
        self.spray = Spray()
        self.agents = Agents(vil, self.rng)
        self.t_check = 0.0

    # -- the village in the bed -----------------------------------------------------------------------------------
    def _structures(self):
        """Height of the standing structures over each cell (their unbroken run of blocks from the ground)."""
        if not self.cfg['village']:
            return np.zeros_like(self.terrain)
        run, _ = column_runs(self.K, self.H)
        vx = np.floor(self.X).astype(int) - VL.ORIGIN[0]
        vy = np.floor(self.Y).astype(int) - VL.ORIGIN[1]
        out = np.zeros_like(self.terrain)
        if self.dx <= 1.0:
            ok = (vx >= 0) & (vx < VL.NX) & (vy >= 0) & (vy < VL.NY)
            out[ok] = run[vy[ok], vx[ok]]
        else:
            k = int(self.dx)
            for a in range(k):
                for b in range(k):
                    jx = np.floor(self.X - self.dx / 2 + a + 0.5).astype(int) - VL.ORIGIN[0]
                    jy = np.floor(self.Y - self.dx / 2 + b + 0.5).astype(int) - VL.ORIGIN[1]
                    ok = (jx >= 0) & (jx < VL.NX) & (jy >= 0) & (jy < VL.NY)
                    vals = np.zeros_like(out)
                    vals[ok] = run[jy[ok], jx[ok]]
                    out = np.maximum(out, vals * 0.85)
        return out.astype(np.float32)

    # -- sampling the water ---------------------------------------------------------------------------------------
    def sample(self, x, y):
        s = self.sim
        fx = (np.asarray(x) - self.x0) / self.dx - 0.5
        fy = (np.asarray(y) - self.y0) / self.dx - 0.5
        fx = np.clip(fx, 0, self.nx - 1.001)
        fy = np.clip(fy, 0, self.ny - 1.001)
        i0, j0 = fx.astype(int), fy.astype(int)
        ax, ay = fx - i0, fy - j0
        u, v = s.velocities()

        def bil(a):
            return (a[j0, i0] * (1 - ax) * (1 - ay) + a[j0, i0 + 1] * ax * (1 - ay) + a[j0 + 1, i0] * (1 - ax) * ay +
                    a[j0 + 1, i0 + 1] * ax * ay)
        return bil(s.h + s.b), bil(s.h), bil(u), bil(v)

    def bed_at(self, x, y):
        s = self.sim
        i = np.clip(((np.asarray(x) - self.x0) / self.dx).astype(int), 0, self.nx - 1)
        j = np.clip(((np.asarray(y) - self.y0) / self.dx).astype(int), 0, self.ny - 1)
        return s.b[j, i]

    def ground_at(self, x, y):
        i = np.clip(((np.asarray(x) - self.x0) / self.dx).astype(int), 0, self.nx - 1)
        j = np.clip(((np.asarray(y) - self.y0) / self.dx).astype(int), 0, self.ny - 1)
        return self.terrain[j, i]

    # -- breaking ----------------------------------------------------------------------------------------------------
    def check_blocks(self):
        K = self.K
        idx = np.flatnonzero(K)
        if len(idx) == 0:
            return 0
        kz, ky, kx = np.unravel_index(idx, K.shape)
        bx = kx + VL.ORIGIN[0] + 0.5
        by = ky + VL.ORIGIN[1] + 0.5
        bz = kz + VL.ORIGIN[2]
        # only blocks the water can reach
        e0, h0, _, _ = self.sample(bx, by)
        near = (e0 + 1.5 > bz) | (np.abs(e0 - bz) < 3)
        cand = np.nonzero(near)[0]
        if len(cand) == 0:
            return 0
        load = np.zeros(len(cand), np.float32)
        solid = K > 0
        for (dx, dy) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx_, ny_ = kx[cand] + dx, ky[cand] + dy
            inb = (nx_ >= 0) & (nx_ < K.shape[2]) & (ny_ >= 0) & (ny_ < K.shape[1])
            exposed = np.ones(len(cand), bool)
            exposed[inb] = ~solid[kz[cand][inb], ny_[inb], nx_[inb]]
            e, hh, uu, vv = self.sample(bx[cand] + dx, by[cand] + dy)
            d = np.clip(e - bz[cand], 0.0, 5.0) * (hh > 0.05)
            l = d * (uu * uu + vv * vv) + 0.5 * G * d * d
            load = np.maximum(load, l * exposed)
        brk = cand[load > self.strength.ravel()[idx[cand]] * LOAD_SCALE]
        if len(brk) == 0:
            return 0
        self._detach(idx[brk], kick=True)
        # whatever is no longer held up by the ground comes down too
        lab, n = ndimage.label(K > 0)
        if n:
            run, ground = column_runs(K, self.H)
            k0 = ground - VL.ORIGIN[2]
            zz = np.arange(K.shape[0])[:, None, None]
            touching = (zz <= k0[None]) & (K > 0)
            grounded = np.unique(lab[touching])
            keep = np.zeros(n + 1, bool)
            keep[grounded] = True
            keep[0] = True
            loose = np.flatnonzero(~keep[lab] & (K > 0))
            if len(loose):
                self._detach(loose, kick=False)
        self.sim.b = (self.terrain + self._structures()).astype(np.float32)
        if hasattr(self.sim, '_gb'):
            del self.sim._gb
        return len(brk)

    def _detach(self, flat, kick):
        K = self.K
        kz, ky, kx = np.unravel_index(flat, K.shape)
        kinds = K.ravel()[flat].astype(np.int32)
        K.ravel()[flat] = 0
        p = np.stack([kx + VL.ORIGIN[0] + 0.5, ky + VL.ORIGIN[1] + 0.5, kz + VL.ORIGIN[2] + 0.5], 1).astype(np.float32)
        _, _, u, v = self.sample(p[:, 0], p[:, 1])
        vel = np.zeros_like(p)
        if kick:
            vel[:, 0] = u * self.rng.uniform(0.4, 0.9, len(p))
            vel[:, 1] = v * self.rng.uniform(0.4, 0.9, len(p))
            vel[:, 2] = self.rng.uniform(0.0, 2.5, len(p)) * min(3.0, 1.0 + self.cfg['A'] / 30.0)
        vel += self.rng.normal(0, 0.6, vel.shape).astype(np.float32)
        self.debris.add(p, vel, kinds, self.rng)
        for f in flat:
            self.events.append((self.sim.t, int(f)))
        if kick and self.cfg['A'] >= 5:
            # a splash where it broke
            m = min(len(p), 60)
            sel = self.rng.choice(len(p), m, replace=False) if len(p) > m else np.arange(len(p))
            sz = np.full(len(sel), 1.2 + self.cfg['A'] ** 0.5 * 0.5)
            sv = vel[sel] * 0.5 + np.array([0, 0, 2.0])
            self.spray.add(p[sel] + np.array([0, 0, 0.5]), sv, sz, self.rng.uniform(1.0, 2.2, len(sel)), self.rng)

    # -- spray from the wave front -------------------------------------------------------------------------------------
    def make_spray(self, rise):
        A = self.cfg['A']
        if A < 5:
            return
        thr = 0.9 * np.sqrt(A) * 2.0
        # where the front meets the shore and the land (out at sea the wave is still a smooth swell)
        near_land = self.terrain > -6.0 - 0.06 * A
        cand = np.flatnonzero((rise > thr) & (self.sim.h > 1.0) & near_land)
        if len(cand) == 0:
            return
        n = min(len(cand), int(60 + A))
        sel = self.rng.choice(cand, n, replace=False)
        j, i = np.unravel_index(sel, rise.shape)
        u, v = self.sim.velocities()
        eta = self.sim.h + self.sim.b
        p = np.stack([self.X[j, i] + self.rng.uniform(-0.5, 0.5, n) * self.dx,
                      self.Y[j, i] + self.rng.uniform(-0.5, 0.5, n) * self.dx,
                      eta[j, i] + self.rng.uniform(-0.1, 0.25, n) * A ** 0.8], 1)
        vel = np.stack([u[j, i] * self.rng.uniform(0.3, 0.8, n), v[j, i] * self.rng.uniform(0.3, 0.8, n),
                        self.rng.uniform(0.2, 1.0, n) * np.sqrt(G * A) * 0.45], 1)
        size = self.rng.uniform(0.8, 1.8, n) * max(2.0, A ** 0.75)
        self.spray.add(p, vel, size, self.rng.uniform(1.4, 3.2, n), self.rng)

    # -- run and record ---------------------------------------------------------------------------------------------
    def run(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        cfg = self.cfg
        n_rec = int(cfg['t_end'] * REC_HZ) + 1
        wpath = os.path.join(out_dir, 'water.npy')
        water = np.lib.format.open_memmap(wpath, mode='w+', dtype=np.float16, shape=(n_rec, self.ny, self.nx, 6))
        debris_frames, spray_frames, agent_frames = [], [], []
        t0 = time.time()
        dt_sub = 1.0 / 120.0
        eta_prev = self.sim.h + self.sim.b
        for k in range(n_rec):
            t_target = k / REC_HZ
            # integrate the water to this frame, with the particles and the people in step
            while self.sim.t < t_target - 1e-9:
                dt = min(self.sim.max_dt(0.45), 0.05, t_target - self.sim.t, dt_sub)
                self.sim.step(dt)
                self.debris.step(dt, self.sample, self.bed_at, self.rng)
                self.spray.step(dt)
                self.agents.step(dt, self.sim.t, self.sample, self.ground_at, cfg, self.rng)
                if cfg['village'] and self.sim.t - self.t_check >= 0.05:
                    self.check_blocks()
                    self.t_check = self.sim.t
            eta = self.sim.h + self.sim.b
            rise = (eta - eta_prev) * REC_HZ
            eta_prev = eta
            if k > 0:
                self.make_spray(rise)
            u, v = self.sim.velocities()
            st, vel = WT.surface(eta, self.sim.h, self.sim.b, self.sim.foam, u, v, self.dx)
            water[k, ..., :4] = st
            water[k, ..., 4:] = vel
            debris_frames.append(np.concatenate([self.debris.p, self.debris.q], 1).astype(np.float32))
            spray_frames.append(self.spray.snapshot())
            agent_frames.append(self.agents.snapshot(self.sim.t))
            if k % 30 == 0:
                print(f'[round {self.n}] t={self.sim.t:5.2f}  debris {len(self.debris.p):5d}  spray '
                      f'{len(self.spray.p):4d}  broken {len(self.events):5d}  {time.time() - t0:6.1f}s', flush=True)
        water.flush()
        del water
        nd = max(len(f) for f in debris_frames)
        D = np.full((n_rec, nd, 7), np.nan, np.float32)
        for k, f in enumerate(debris_frames):
            D[k, :len(f)] = f
        ns = max(1, max(len(f) for f in spray_frames))
        S = np.zeros((n_rec, ns, 6), np.float16)
        for k, f in enumerate(spray_frames):
            S[k, :len(f)] = f
        np.save(os.path.join(out_dir, 'debris.npy'), D)
        np.save(os.path.join(out_dir, 'spray.npy'), S)
        np.savez(os.path.join(out_dir, 'meta.npz'), kinds=self.debris.kind, agents=np.stack(agent_frames),
                 events=np.array(self.events, np.float64).reshape(-1, 2), grid=np.array(
                     [self.nx, self.ny, self.x0, self.y0, self.dx]), A=cfg['A'], ramp=cfg['ramp'],
                 hold=cfg['hold'], fall=cfg['fall'], t_end=cfg['t_end'], terrain=self.terrain,
                 version=VERSION)
        print(f'[round {self.n}] done in {time.time() - t0:.0f}s', flush=True)


def round_dir(n):
    return os.path.join(CACHE, f'round{n}_v{VERSION}')


def simulate(n):
    import scene as SC
    wd = SC.world_data()
    H = wd['H']
    vil = VL.build(H)
    Round(n, H, vil).run(round_dir(n))


if __name__ == '__main__':
    for a in sys.argv[1:]:
        simulate(int(a))
