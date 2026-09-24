"""Explosion visual effects: fireball/smoke puffs, bright flashes and short-lived point lights.

Deterministic (seeded) and stepped once per video frame from the simulation's explosion events, so a
render is reproducible. Mass detonations are merged so thousands of explosions stay cheap to draw.
"""
import numpy as np

FPS = 30
MAX_PUFF_EXPL = 26        # explosions per frame that spawn their own puffs (the rest are merged in)
MAX_FLASH = 30
MAX_LIGHTS = 16
MAX_PUFFS = 1500


class VFX:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        # puffs: pos(3) vel(3) age life size0 size1 variant spin opacity
        self.pp = np.zeros((0, 3))
        self.pv = np.zeros((0, 3))
        self.page = np.zeros(0)
        self.plife = np.zeros(0)
        self.ps0 = np.zeros(0)
        self.ps1 = np.zeros(0)
        self.pvar = np.zeros(0)
        self.popa = np.zeros(0)
        self.pfire = np.zeros(0)     # how much of the early life is fireball
        # flashes: pos age life size intensity
        self.fp = np.zeros((0, 3))
        self.fage = np.zeros(0)
        self.flife = np.zeros(0)
        self.fsize = np.zeros(0)
        self.fint = np.zeros(0)
        # lights: pos age intensity radius
        self.lp = np.zeros((0, 3))
        self.lage = np.zeros(0)
        self.lint = np.zeros(0)
        self.lrad = np.zeros(0)

    def _spawn(self, centers, scale):
        rng = self.rng
        n_e = len(centers)
        if n_e == 0:
            return
        # puffs: a fireball core plus a few smoke balls per explosion
        per = rng.integers(3, 6, n_e)
        idx = np.repeat(np.arange(n_e), per)
        m = len(idx)
        c = centers[idx]
        off = rng.normal(0, 0.55, (m, 3)) * scale
        dirv = off / np.maximum(np.linalg.norm(off, axis=1, keepdims=True), 1e-6)
        vel = dirv * rng.uniform(1.0, 3.2, (m, 1)) * np.sqrt(scale)
        vel[:, 2] += rng.uniform(1.2, 3.0, m)
        life = rng.uniform(0.9, 1.7, m) * (0.8 + 0.2 * scale)
        s0 = rng.uniform(1.3, 2.1, m) * scale
        s1 = s0 * rng.uniform(2.0, 2.8, m)
        self.pp = np.concatenate([self.pp, c + off])
        self.pv = np.concatenate([self.pv, vel])
        self.page = np.concatenate([self.page, np.zeros(m)])
        self.plife = np.concatenate([self.plife, life])
        self.ps0 = np.concatenate([self.ps0, s0])
        self.ps1 = np.concatenate([self.ps1, s1])
        self.pvar = np.concatenate([self.pvar, rng.integers(0, 4, m).astype(float)])
        self.popa = np.concatenate([self.popa, rng.uniform(0.55, 0.8, m)])
        self.pfire = np.concatenate([self.pfire, rng.uniform(0.18, 0.32, m)])
        if len(self.pp) > MAX_PUFFS:          # drop the oldest
            keep = np.argsort(self.page)[:MAX_PUFFS]
            keep.sort()
            for a in ('pp', 'pv', 'page', 'plife', 'ps0', 'ps1', 'pvar', 'popa', 'pfire'):
                setattr(self, a, getattr(self, a)[keep])

    def step(self, centers, dt=1.0 / FPS, big=None):
        """Advance by dt seconds of simulation time and add the explosions that happened during it.
        big: optional list of (centre, scale) for single huge blasts (a creeper going off)."""
        rng = self.rng
        centers = np.asarray(centers, float).reshape(-1, 3)
        n = len(centers)
        if n:
            # puffs
            if n > MAX_PUFF_EXPL:
                sel = rng.choice(n, MAX_PUFF_EXPL, replace=False)
                scale = min(2.0, np.sqrt(n / MAX_PUFF_EXPL))
                self._spawn(centers[sel], scale)
                self.popa[self.page == 0.0] *= 0.65  # a mass detonation must not hide everything behind smoke
            else:
                self._spawn(centers, 1.0)
            # flashes
            fsel = centers if n <= MAX_FLASH else centers[rng.choice(n, MAX_FLASH, replace=False)]
            fscale = 1.0 if n <= MAX_FLASH else min(1.5, np.sqrt(n / MAX_FLASH))
            k = len(fsel)
            self.fp = np.concatenate([self.fp, fsel + rng.normal(0, 0.2, (k, 3))])
            self.fage = np.concatenate([self.fage, np.zeros(k)])
            self.flife = np.concatenate([self.flife, rng.uniform(0.09, 0.14, k)])
            self.fsize = np.concatenate([self.fsize, rng.uniform(3.2, 4.4, k) * fscale])
            # a mass detonation should read as a firestorm, not a white-out
            self.fint = np.concatenate([self.fint, rng.uniform(0.8, 1.0, k) * (1.0 if n <= 6 else 0.6)])
            # lights: bucket explosions on a coarse grid so a mass detonation becomes a few big lights
            cell = np.floor(centers / 7.0).astype(np.int64)
            keys, inv, counts = np.unique(cell, axis=0, return_inverse=True, return_counts=True)
            inv = inv.ravel()
            for b in range(len(keys)):
                pts = centers[inv == b]
                self.lp = np.concatenate([self.lp, pts.mean(0)[None]])
                self.lage = np.concatenate([self.lage, [0.0]])
                self.lint = np.concatenate([self.lint, [min(4.0, np.sqrt(counts[b]))]])
                self.lrad = np.concatenate([self.lrad, [min(26.0, 11.0 * np.sqrt(np.sqrt(counts[b])))]])
        for c, sc in (big or []):
            c = np.asarray(c, float)[None]
            for _ in range(int(6 * sc)):
                self._spawn(c + self.rng.normal(0, 1.2 * sc, (1, 3)), sc)
            self.fp = np.concatenate([self.fp, c])
            self.fage = np.concatenate([self.fage, [0.0]])
            self.flife = np.concatenate([self.flife, [0.3]])
            self.fsize = np.concatenate([self.fsize, [5.0 * sc]])
            self.fint = np.concatenate([self.fint, [1.4]])
            self.lp = np.concatenate([self.lp, c])
            self.lage = np.concatenate([self.lage, [0.0]])
            self.lint = np.concatenate([self.lint, [1.8 * sc]])
            self.lrad = np.concatenate([self.lrad, [22.0 * sc]])
        # advance puffs
        if len(self.pp):
            self.page += dt
            self.pv *= max(0.0, 1.0 - 1.8 * dt)
            self.pv[:, 2] += 0.9 * dt                 # hot smoke keeps rising a little
            self.pp += self.pv * dt
            alive = self.page < self.plife
            for a in ('pp', 'pv', 'page', 'plife', 'ps0', 'ps1', 'pvar', 'popa', 'pfire'):
                setattr(self, a, getattr(self, a)[alive])
        if len(self.fp):
            self.fage += dt
            alive = self.fage < self.flife
            for a in ('fp', 'fage', 'flife', 'fsize', 'fint'):
                setattr(self, a, getattr(self, a)[alive])
        if len(self.lp):
            self.lage += dt
            alive = self.lage < 0.45
            for a in ('lp', 'lage', 'lint', 'lrad'):
                setattr(self, a, getattr(self, a)[alive])
        self.t += dt

    # -- render data ---------------------------------------------------------------------------
    def puffs(self):
        """(N, 8) float32: pos3, size, alpha, fire, variant, age01."""
        if len(self.pp) == 0:
            return np.zeros((0, 8), np.float32)
        u = self.page / self.plife
        size = self.ps0 + (self.ps1 - self.ps0) * (1.0 - (1.0 - u) ** 2.2)
        alpha = self.popa * np.clip(u / 0.06, 0, 1) * (1.0 - np.clip((u - 0.45) / 0.55, 0, 1)) ** 1.3
        fire = np.clip(1.0 - self.page / (self.pfire * self.plife + 1e-6), 0, 1) ** 1.5
        out = np.zeros((len(self.pp), 8), np.float32)
        out[:, 0:3] = self.pp
        out[:, 3] = size
        out[:, 4] = alpha
        out[:, 5] = fire
        out[:, 6] = self.pvar
        out[:, 7] = u
        return out

    def flashes(self):
        """(N, 5) float32: pos3, size, intensity."""
        if len(self.fp) == 0:
            return np.zeros((0, 5), np.float32)
        u = self.fage / self.flife
        out = np.zeros((len(self.fp), 5), np.float32)
        out[:, 0:3] = self.fp
        out[:, 3] = self.fsize * (0.55 + 0.45 * np.sqrt(u))
        out[:, 4] = self.fint * (1.0 - u) ** 1.6
        return out

    def lights(self):
        """Up to MAX_LIGHTS rows of (pos3, intensity, radius), strongest first."""
        if len(self.lp) == 0:
            return np.zeros((0, 5), np.float32)
        inten = self.lint * np.exp(-self.lage / 0.11)
        order = np.argsort(-inten)[:MAX_LIGHTS]
        out = np.zeros((len(order), 5), np.float32)
        out[:, 0:3] = self.lp[order]
        out[:, 3] = inten[order]
        out[:, 4] = self.lrad[order]
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
