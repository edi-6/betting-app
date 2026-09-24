"""Particle effects: dust puffs where arrows and anvils hit the ground, and the puff textures.

Deterministic (seeded) and stepped once per video frame with that frame's simulation time, so slow motion
slows the dust down too and a render is reproducible. Mass impacts are merged so thousands of arrows stay
cheap to draw.
"""
import numpy as np

MAX_DUST_SPAWN = 36       # dust puffs spawned per frame at most (mass impacts are merged into bigger puffs)
MAX_DUST = 1400


class Dust:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.p = np.zeros((0, 3))
        self.v = np.zeros((0, 3))
        self.age = np.zeros(0)
        self.life = np.zeros(0)
        self.s0 = np.zeros(0)
        self.s1 = np.zeros(0)
        self.var = np.zeros(0)
        self.opa = np.zeros(0)

    def _cat(self, **kw):
        for k, val in kw.items():
            setattr(self, k, np.concatenate([getattr(self, k), val]))

    def step(self, points, dt):
        """Advance by dt seconds of simulation time and add puffs for this frame's ground impacts."""
        rng = self.rng
        pts = np.asarray(points, float).reshape(-1, 3)
        n = len(pts)
        if n:
            if n > MAX_DUST_SPAWN:
                sel = rng.choice(n, MAX_DUST_SPAWN, replace=False)
                scale = min(2.2, np.sqrt(n / MAX_DUST_SPAWN))
                pts = pts[sel]
            else:
                scale = 1.0
            m = len(pts)
            off = rng.normal(0, 0.25, (m, 3)) * scale
            off[:, 2] = np.abs(off[:, 2]) * 0.4
            vel = rng.normal(0, 0.5, (m, 3)) * np.sqrt(scale)
            vel[:, 2] = rng.uniform(0.4, 1.4, m)
            s0 = rng.uniform(0.5, 0.9, m) * scale
            self._cat(p=pts + off, v=vel, age=np.zeros(m), life=rng.uniform(0.9, 1.6, m),
                      s0=s0, s1=s0 * rng.uniform(2.2, 3.2, m), var=rng.integers(0, 4, m).astype(float),
                      opa=rng.uniform(0.3, 0.5, m))
            if len(self.p) > MAX_DUST:
                keep = np.sort(np.argsort(self.age)[:MAX_DUST])
                for k in ('p', 'v', 'age', 'life', 's0', 's1', 'var', 'opa'):
                    setattr(self, k, getattr(self, k)[keep])
        if len(self.p) and dt > 0:
            self.age += dt
            self.v *= max(0.0, 1.0 - 1.5 * dt)
            self.v[:, 2] += 0.25 * dt
            self.p += self.v * dt
            alive = self.age < self.life
            for k in ('p', 'v', 'age', 'life', 's0', 's1', 'var', 'opa'):
                setattr(self, k, getattr(self, k)[alive])

    def puffs(self):
        """(N, 8) float32: pos3, size, alpha, fire(0), variant, age01 -- the layout the puff shader expects."""
        if len(self.p) == 0:
            return np.zeros((0, 8), np.float32)
        u = self.age / self.life
        out = np.zeros((len(self.p), 8), np.float32)
        out[:, 0:3] = self.p
        out[:, 3] = self.s0 + (self.s1 - self.s0) * (1.0 - (1.0 - u) ** 2.0)
        out[:, 4] = self.opa * np.clip(u / 0.08, 0, 1) * (1.0 - u) ** 1.4
        out[:, 6] = self.var
        out[:, 7] = u
        return out


def puff_textures(n=4, size=128, seed=3):
    """(n, size, size, 2) float32: R = smoke density (soft, billowy, round), G = inner brightness variation."""
    from noise import fbm2d
    v = (np.arange(size) + 0.5) / size * 2 - 1
    X, Y = np.meshgrid(v, v)
    r = np.hypot(X, Y)
    out = np.zeros((n, size, size, 2), np.float32)
    for i in range(n):
        warp = fbm2d(X * 2.2 + i * 7.1, Y * 2.2 - i * 3.3, octaves=3, seed=seed + 10 * i)
        billow = fbm2d(X * 4.0 + 11 * i, Y * 4.0 + 5 * i, octaves=4, seed=seed + 10 * i + 5)
        edge = 0.78 + 0.18 * warp
        d = np.clip(1.0 - r / edge, 0, 1)
        dens = d ** 0.7 * (0.72 + 0.28 * billow)
        dens = np.clip((dens - 0.05) / 0.8, 0, 1)
        out[i, :, :, 0] = dens
        out[i, :, :, 1] = np.clip(0.5 + 0.6 * billow, 0, 1) * np.clip(1.2 - r, 0, 1)
    return out
