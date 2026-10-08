"""The end: the picture's creeper goes off. Every marble is thrown out of the tank, away from the creeper's chest and
out through the glass towards you, and comes down on the meadow, bouncing and rolling to a stop. The glass goes
in hundreds of shards. No marble-on-marble collisions out here: they're spread out by then.
"""
import numpy as np

import machine as M
from creeper import VOXEL_DTYPE

G = 20.0
DT = 1.0 / 240.0
REC = 2
DUR = 6.0


def ground(x, y):
    """The top of whatever's under (x, y): the pedestal (two steps) or the meadow."""
    z = np.zeros_like(x)
    lo = (np.abs(x - M.MX) < M.HALF + 5) & (y > M.MY - 6) & (y < M.MY + 5)
    hi = (np.abs(x - M.MX) < M.HALF + 3) & (y > M.MY - 4) & (y < M.MY + 4)
    z = np.where(lo, M.MZ - 1.5, z)
    z = np.where(hi, M.MZ, z)
    return z


class Blast:
    def __init__(self, start_world, centre, seed=21):
        rng = np.random.default_rng(seed)
        p = np.asarray(start_world, float).copy()
        n = len(p)
        c = np.asarray(centre, float)
        d = p[:, [0, 2]] - c[[0, 2]]
        r = np.linalg.norm(d, axis=1) + 1e-6
        u = d / r[:, None]
        vr = 30.0 * np.exp(-r / 15.0) + 5.0 + rng.normal(0.0, 1.8, n)
        fwd = 14.0 * np.exp(-r / 20.0) + 6.0 + rng.uniform(0.0, 7.0, n)
        v = np.zeros((n, 3))
        v[:, 0] = u[:, 0] * vr + rng.normal(0.0, 1.2, n)
        v[:, 2] = u[:, 1] * vr + 3.0 + rng.normal(0.0, 1.2, n)
        v[:, 1] = -fwd
        p[:, 1] -= 0.05
        steps = int(DUR / DT)
        self.path = np.zeros((steps // REC + 1, n, 3), np.float32)
        self.path[0] = p
        self.hits = []                                          # (t, n hitting the ground) for the sound
        rest = np.zeros(n, bool)
        for s in range(1, steps + 1):
            mv = ~rest
            v[mv, 2] -= G * DT
            p[mv] += v[mv] * DT
            gz = ground(p[:, 0], p[:, 1]) + M.R
            under = (p[:, 2] < gz) & mv
            if under.any():
                fast = under & (v[:, 2] < -2.0)
                self.hits.append((s * DT, int(fast.sum()), p[fast][:24].copy() if fast.any() else None))
                p[under, 2] = gz[under]
                v[under, 2] = -v[under, 2] * 0.38
                v[under, 0] *= 0.8
                v[under, 1] *= 0.8
                slow = under & (np.abs(v[:, 2]) < 1.2)
                v[slow, 2] = 0.0
                v[slow, 0] *= 0.93
                v[slow, 1] *= 0.93
                rest |= slow & (np.hypot(v[:, 0], v[:, 1]) < 0.15)
            if s % REC == 0:
                self.path[s // REC] = p
        # the glass: shards from all over the pane, mostly from around the blast, flying out towards you
        m = 700
        sx = np.concatenate([rng.normal(c[0], 7.0, m // 2), rng.uniform(-M.HALF, M.HALF, m - m // 2)])
        sz = np.concatenate([rng.normal(c[2], 9.0, m // 2), rng.uniform(M.MZ, M.MZ + M.FRAME_TOP, m - m // 2)])
        sx = np.clip(sx, -M.HALF + 0.2, M.HALF - 0.2)
        sz = np.clip(sz, M.MZ + 0.2, M.MZ + M.FRAME_TOP - 0.2)
        sp = np.stack([sx, np.full(m, M.MY - M.SLOT - 0.05), sz], -1)
        dd = sp[:, [0, 2]] - c[[0, 2]]
        rr = np.linalg.norm(dd, axis=1) + 1e-6
        sv = np.zeros((m, 3))
        k = 20.0 * np.exp(-rr / 14.0) + 4.0
        sv[:, 0] = dd[:, 0] / rr * k + rng.normal(0, 2.0, m)
        sv[:, 2] = dd[:, 1] / rr * k + 2.0 + rng.normal(0, 2.0, m)
        sv[:, 1] = -(22.0 * np.exp(-rr / 18.0) + 7.0 + rng.uniform(0, 8.0, m))
        self.s0, self.sv = sp, sv
        self.s_size = rng.uniform(0.18, 0.55, m)
        self.s_axis = rng.normal(0, 1, (m, 3))
        self.s_axis /= np.linalg.norm(self.s_axis, axis=1, keepdims=True)
        self.s_spin = rng.uniform(4.0, 16.0, m) * rng.choice((-1, 1), m)
        self.s_tint = rng.uniform(0.9, 1.0, m)

    def marbles(self, t):
        """Positions t seconds after the blast."""
        f = np.clip(t / (DT * REC), 0, len(self.path) - 1)
        k = int(np.floor(f))
        k1 = min(k + 1, len(self.path) - 1)
        w = f - k
        return self.path[k] * (1 - w) + self.path[k1] * w

    def shards(self, t):
        if t < 0:
            return np.zeros(0, VOXEL_DTYPE)
        p = self.s0 + self.sv * t + np.array([0, 0, -0.5 * G]) * t * t
        gz = ground(p[:, 0], p[:, 1]) + 0.05
        landed = p[:, 2] < gz
        p[:, 2] = np.maximum(p[:, 2], gz)
        ang = self.s_spin * np.minimum(t, 1.2 + 0.0 * t)
        ang = np.where(landed, np.round(ang / np.pi) * np.pi, ang)
        q = np.concatenate([self.s_axis * np.sin(ang / 2)[:, None], np.cos(ang / 2)[:, None]], 1)
        n = len(p)
        out = np.zeros(n, VOXEL_DTYPE)
        out['pos'] = p
        out['quat'] = q
        out['scale'] = self.s_size
        col = np.clip(np.array([214, 236, 242])[None, :] * self.s_tint[:, None], 0, 255).astype(np.uint8)
        for key in ('cx', 'cy', 'cz'):
            out[key][:, :3] = col
        out['inner'][:, :3] = (200, 220, 226)
        out['cx'][:, 3] = 0b111111
        out['cy'][:, 3] = 0b111111
        out['cz'][:, 3] = 0
        return out
