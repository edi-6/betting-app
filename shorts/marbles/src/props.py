"""What the renderer draws of the marbles at a given moment: the sim's marbles (where they are, in the colour of
where they'll end up), the creeper's flashing before it goes, the blast, the hopper's surface and the golden ball.
"""
import numpy as np

import machine as M
import picture as PIC
import timeline as TL
from explosion import Blast


class Marbles:
    def __init__(self, sim, hopper):
        self.sim = sim
        self.T = np.asarray(sim['t'])
        self.pos = sim['pos']
        self.final = np.asarray(sim['final'])
        self.hero = M.N - 1
        # the last marble lands on the top of the picture: it's the bright star there
        self.col, self.role = PIC.colors(self.final, star=tuple(self.final[self.hero]))
        self.colf = self.col.astype(np.float32) / 255.0
        ev = sim['events']
        self.t_land = float(ev[5])
        self.count_t = np.asarray(sim['count_t']).copy()
        self.count_t[self.hero] = self.t_land                 # it counts when it lands
        self.hopper = hopper
        self.t_boom = self.t_land + TL.BOOM_AT
        self.t_fuse = self.t_land + TL.FUSE_AT
        fw = M.to_world(self.final[:, 0], self.final[:, 1])
        self.creeper_c = np.array([6.4, M.MY, M.MZ + 30.0])
        self.blast = Blast(fw, self.creeper_c)

    def xz(self, ts):
        """Sim positions (N, 2) at sim time ts (NaN: not out of the hopper yet)."""
        f = np.interp(ts, self.T, np.arange(len(self.T)))
        k = int(np.floor(f))
        k1 = min(k + 1, len(self.T) - 1)
        w = f - k
        a = np.asarray(self.pos[k])
        if w < 1e-6 or k1 == k:
            return a
        b = np.asarray(self.pos[k1])
        out = a * (1 - w) + b * w
        # a marble that first appears in the next sample: take it from there
        fresh = np.isnan(a[:, 0]) & ~np.isnan(b[:, 0])
        out[fresh] = b[fresh] if w > 0.5 else np.nan
        return out

    def hero_pos(self, ts):
        p = self.xz(min(ts, self.T[-1]))[self.hero]
        if np.isnan(p[0]):
            return np.array([M.MX, M.MY, M.MZ + M.THROAT_Z])
        return M.to_world(p[0], p[1])

    def counted(self, ts):
        return int((self.count_t <= ts).sum())

    def level(self, ts):
        """Roughly how high the pile is (sim z) at ts."""
        return 64.0 * min(self.counted(ts), M.N) / M.N

    def fuse(self, ts):
        """(swell, white) of the picture's creeper: like the game's, it pulses white, faster and faster, and swells.
        The flashes stay under 2.5 a second (it's a big part of the screen)."""
        if ts < self.t_fuse:
            return 1.0, 0.0
        d = ts - self.t_fuse
        a = float(np.clip(d / TL.FUSE_LEN, 0.0, 1.0))
        ph = 2 * np.pi * (1.0 * d + 0.7 * d * d / TL.FUSE_LEN)
        white = 0.5 * (0.5 + 0.5 * np.sin(ph - np.pi / 2)) ** 2 * min(1.0, a * 3)
        swell = 1.0 + 0.05 * (0.5 + 0.5 * np.sin(ph - np.pi / 2)) + 0.18 * a ** 3
        return swell, white

    def instances(self, ts, shake=0.0):
        """(M, 8) float32 marble instances at sim/story time ts."""
        out = []
        if ts < self.t_boom:
            p = self.xz(min(ts, self.T[-1]))
            ok = ~np.isnan(p[:, 0])
            idx = np.where(ok)[0]
            w = M.to_world(p[idx, 0], p[idx, 1])
            m = np.zeros((len(idx), 8), np.float32)
            m[:, :3] = w
            m[:, 3] = M.R
            m[:, 4:7] = self.colf[idx]
            if ts >= self.t_fuse:
                swell, white = self.fuse(ts)
                cr = self.role[idx] >= 1
                cr &= self.role[idx] <= 2
                m[cr, 7] = white
                if swell != 1.0:
                    c = self.creeper_c
                    m[cr, 0] = c[0] + (m[cr, 0] - c[0]) * (1 + (swell - 1) * 0.35)
                    m[cr, 2] = c[2] + (m[cr, 2] - c[2]) * (1 + (swell - 1) * 0.35)
            out.append(m)
        else:
            p = self.blast.marbles(ts - self.t_boom)
            m = np.zeros((len(p), 8), np.float32)
            m[:, :3] = p
            m[:, 3] = M.R
            m[:, 4:7] = self.colf
            out.append(m)
        if ts < self.t_boom:
            out.append(self.hopper.marbles(ts, shake))
        out.append(self.hopper.gold_instance(ts))
        return np.concatenate(out)
