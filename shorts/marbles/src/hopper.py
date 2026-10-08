"""What's in the hopper: the top of 10,000 marbles, draining, and the golden ball.

The marbles in the hopper aren't simulated one by one: only the surface you can see is drawn, from a model of how a
bin like this drains (funnel flow). A crater opens over the hole and deepens down to it, its sides at the angle of
repose, and then the whole surface sinks. Inside the crater the marbles slide down in rings towards the hole; on the
flat around it they just sink. How fast it goes is the sim's: every marble that comes out of the chute has left the
hopper.

The golden ball (too big for the hole) sits on the flat at the start. When the crater reaches it, it rolls in and
plugs the hole: that's the jam. When the goat rams the pedestal, the jolt pops it out over the back of the hopper,
and it falls behind the machine.
"""
import numpy as np

import machine as M

SLOPE = 0.62                        # the crater's sides (tan of the angle of repose)
SPACING = 2 * M.R * 1.02
IX, IY = M.HOP_X - 1.0, M.HOP_Y - 1.0          # the inside's half sizes
Z_HOLE = M.HOP_Z0 + 1.0             # the hole's rim (the lowest step's top)
FULL = M.HOP_Z1 - 0.2               # the surface when full
GOLD_HOME = (-5.5, 0.9)             # where the golden ball sits at the start (x, y - MY): against the wall
GOLD_SEAT = Z_HOLE + np.sqrt(M.GOLD_R ** 2 - M.HOLE ** 2)       # its centre, wedged in the hole
GOLD_COL = (1.0, 0.80, 0.30)
G = 20.0
# rainbow: the 16 dye colours, brightened a little
DYES = np.array([(233, 236, 236), (240, 118, 19), (189, 68, 179), (58, 175, 217), (248, 198, 39), (112, 185, 25),
                 (237, 141, 172), (90, 96, 100), (160, 160, 154), (21, 137, 145), (121, 42, 172), (53, 57, 157),
                 (114, 71, 40), (84, 109, 27), (161, 39, 34), (40, 40, 46)], float) / 255.0


def _ss(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


class Hopper:
    def __init__(self, sim_data, t_ram, seed=4):
        self.spawn_t = np.asarray(sim_data['spawn_t'])
        ev = sim_data['events']
        self.t_jam, self.t_save = float(ev[0]), float(ev[1])
        self.t_ram = float(t_ram)
        rng = np.random.default_rng(seed)
        # the flat: a hex grid over the inside, two layers (the second shows through the gaps)
        pts = []
        for layer in (0, 1):
            rows = int(2 * IY / (SPACING * 0.866)) + 1
            for r in range(rows):
                y = -IY + M.R + r * SPACING * 0.866 + layer * SPACING * 0.29
                off = (r % 2) * SPACING / 2 + layer * SPACING / 2
                for x in np.arange(-IX + M.R + off, IX - M.R, SPACING):
                    pts.append((x + rng.normal(0, 0.03), y + rng.normal(0, 0.03), layer))
        self.flat = np.array(pts)
        self.flat_col = DYES[rng.integers(len(DYES), size=len(pts))]
        self.flat_dz = rng.uniform(-0.04, 0.04, len(pts))
        # rings for the crater: ring m sits m spacings out from the hole; its marbles' angles and colours are fixed
        self.ring_rng = np.random.default_rng(seed + 1)
        self.n_ring = 40
        self.ring_off = self.ring_rng.random(400)
        self.ring_cols = DYES[self.ring_rng.integers(len(DYES), size=(400, 200))]
        # when the crater reaches the golden ball, it rolls in and is in the hole at the jam
        gx, gy = GOLD_HOME
        self.gold_r0 = float(np.hypot(gx, gy))
        ts = np.linspace(0.0, self.t_jam, 2000)
        rc = np.array([self.shape(t)[2] for t in ts])
        k = np.searchsorted(rc, self.gold_r0 - M.GOLD_R * 0.4)
        self.t_roll = float(ts[min(k, len(ts) - 1)])
        self.t_roll = min(self.t_roll, self.t_jam - 1.2)

    # -- the surface ---------------------------------------------------------------------------
    def drained(self, t):
        return int(np.searchsorted(self.spawn_t, t))

    def shape(self, t):
        """(plateau height, crater apex height, crater radius) at sim time t."""
        n_out = self.drained(t)
        frac = 1.0 - n_out / M.N
        plateau = Z_HOLE + (FULL - Z_HOLE) * frac ** 0.85
        depth = min(plateau - Z_HOLE, n_out * 0.0009)
        apex = plateau - depth
        return plateau, apex, (plateau - apex) / SLOPE

    def height(self, x, y, t):
        plateau, apex, _ = self.shape(t)
        r = np.hypot(x, y)
        return np.minimum(plateau, apex + SLOPE * np.maximum(r - M.HOLE * 0.7, 0.0))

    def phase(self, t):
        """How far the crater's rings have slid in (in spacings)."""
        return self.drained(t) / 70.0

    # -- the marbles you can see ---------------------------------------------------------------------
    def marbles(self, t, shake=0.0):
        """(K, 8) marble instances: centre, radius, colour, flag."""
        if self.drained(t) >= M.N - 30:
            return np.zeros((0, 8), np.float32)                 # empty
        plateau, apex, rc = self.shape(t)
        out = []
        fx, fy, lay = self.flat[:, 0], self.flat[:, 1], self.flat[:, 2]
        r = np.hypot(fx, fy)
        keep = r > rc + 0.35
        z = plateau + M.R * 0.9 - lay * 0.42 + self.flat_dz
        if shake:
            z = z + shake * np.sin(fx * 3.1 + fy * 2.3 + t * 90.0) * 0.08
        sel = np.where(keep)[0]
        flat = np.zeros((len(sel), 8), np.float32)
        flat[:, 0] = M.MX + fx[sel]
        flat[:, 1] = M.MY + fy[sel]
        flat[:, 2] = z[sel]
        flat[:, 3] = M.R
        flat[:, 4:7] = self.flat_col[sel]
        out.append(flat)
        if rc > 0.3:
            ph = self.phase(t)
            base = int(np.floor(ph))
            fr = ph - base
            rows = []
            for k in range(self.n_ring):
                m = k + base                                  # the ring's identity (keeps its marbles as it moves)
                rk = M.HOLE * 0.55 + (k + 1.0 - fr) * SPACING * 0.9
                if rk > rc + 0.2:
                    break
                n = max(3, int(2 * np.pi * rk / SPACING))
                j = np.arange(n)
                ang = 2 * np.pi * (j + self.ring_off[m % 400]) / n
                x = rk * np.cos(ang)
                y = rk * np.sin(ang)
                inside = (np.abs(x) < IX - M.R) & (np.abs(y) < IY - M.R)
                zz = apex + SLOPE * max(rk - M.HOLE * 0.7, 0.0) + M.R * 0.9
                c = self.ring_cols[m % 400, j % 200]
                blk = np.zeros((inside.sum(), 8), np.float32)
                blk[:, 0] = M.MX + x[inside]
                blk[:, 1] = M.MY + y[inside]
                blk[:, 2] = zz
                blk[:, 3] = M.R
                blk[:, 4:7] = c[inside]
                rows.append(blk)
            if rows:
                out.append(np.concatenate(rows))
        return np.concatenate(out)

    # -- the golden ball -----------------------------------------------------------------------------
    def gold(self, t):
        """Its centre at sim time t (None once it's gone for good)."""
        gx, gy = GOLD_HOME
        if t < self.t_roll:
            return np.array([M.MX + gx, M.MY + gy, self.shape(t)[0] + M.GOLD_R * 0.75])
        if t < self.t_jam:
            u = (t - self.t_roll) / (self.t_jam - self.t_roll)
            u = u * u * (3.0 - 2.0 * u) * 0.35 + u * u * 0.65              # rolling in, faster as it goes
            r = self.gold_r0 * (1.0 - u)
            x, y = gx * r / self.gold_r0, gy * r / self.gold_r0
            zs = float(self.height(np.array([x]), np.array([y]), t)[0]) + M.GOLD_R * 0.8
            z = zs * (1.0 - _ss(0.75, 1.0, u)) + GOLD_SEAT * _ss(0.75, 1.0, u)
            return np.array([M.MX + x, M.MY + y, max(z, GOLD_SEAT)])
        if t < self.t_ram:
            return np.array([M.MX, M.MY, GOLD_SEAT])
        # popped out by the jolt: up and over the back of the hopper, down behind the machine
        tt = t - self.t_ram
        v = np.array([1.2, 7.4, 21.0])
        p = np.array([M.MX, M.MY, GOLD_SEAT]) + v * tt + np.array([0.0, 0.0, -0.5 * G * tt * tt])
        if p[2] < M.GOLD_R:
            return None
        return p

    def gold_instance(self, t):
        p = self.gold(t)
        if p is None:
            return np.zeros((0, 8), np.float32)
        return np.array([[p[0], p[1], p[2], M.GOLD_R, *GOLD_COL, 2.0]], np.float32)

    def gold_spin(self, t):
        """(for the sound) is it rolling now?"""
        return self.t_roll <= t < self.t_jam
