"""Round 4: the 1,000-block wall of water. It is drawn rather than simulated: the shallow-water equations let a wave
that tall slump into a long ramp, and this one never reaches the shore anyway. A face a kilometre high and nearly
vertical, ragged along its length, foam pouring down it and mist blowing off the crest, rolling in across the bay;
the sea draining back off the beach ahead of it, as it does before a tsunami; and then the sponge, which drinks the
lot: the whole sea twists into a whirlpool around it and goes down, until the sea floor lies dry.

Wall.water(t, drink) is the renderer's water dict on a 12 m grid, Wall.mist(t, drink) the puffs off the crest and at
the foot of the wall. t is the wall's own clock: its foot is at y = front(t)."""
import os

import numpy as np

import water as WT
import world as WD
from noise import fbm2d

A = 1000.0
FACE = 190.0            # toe to lip, horizontally (m)
SPEED = 240.0           # how fast it comes (m/s)
Y_START = 4600.0        # where its foot is at t = 0
DRAW, DRAW_L = 14.0, 1100.0     # how far the sea pulls back ahead of it, and over what distance


def front(t):
    return Y_START - SPEED * t


def shade(t):
    """How much of the sky the wall has blotted out over the village (0 far away .. 1 right on top of it)."""
    return float(_sm((2600.0 - front(t)) / 1800.0))


def t_at(y):
    """When the wall's foot (at x = 0) is at y."""
    return (Y_START - y) / SPEED


def _sm(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _n1(x, seed, octaves=3):
    return fbm2d(np.asarray(x, float), np.full(np.shape(x), 0.37), octaves=octaves, seed=seed)


class Wall:
    def __init__(self, H=None, dx=12.0, box=(-4800.0, 4800.0, -600.0, 8400.0), sponge=(-3.5, 0.5, 1.0)):
        x0, x1, y0, y1 = box
        self.dx, self.x0, self.y0 = dx, x0, y0
        self.nx, self.ny = int(round((x1 - x0) / dx)), int(round((y1 - y0) / dx))
        xs = x0 + (np.arange(self.nx) + 0.5) * dx
        ys = y0 + (np.arange(self.ny) + 0.5) * dx
        X, Y = np.meshgrid(xs, ys)
        self.X, self.Y = X, Y
        bed = WD.far_heights(X, Y)
        if H is not None:
            inside = (X >= WD.X0) & (X < WD.X1) & (Y >= WD.Y0) & (Y < WD.Y1)
            ix = np.clip(np.floor(X).astype(int) - WD.X0, 0, WD.NX - 1)
            iy = np.clip(np.floor(Y).astype(int) - WD.Y0, 0, WD.NY - 1)
            bed = np.where(inside, H[iy, ix], bed)
        self.bed = bed.astype(np.float32)
        self.sponge = np.array(sponge, float)
        # fixed noise for the face and the plateau, looked up by position relative to the front
        self._streak_seed = 31
        rng = np.random.default_rng(5)
        n = 900
        self.m_x = rng.uniform(-4300.0, 4300.0, n)
        self.m_ph = rng.random(n)
        self.m_var = rng.integers(0, 4, n).astype(np.float32)
        self.m_toe = rng.random(n) < 0.3
        self.m_life = rng.uniform(3.0, 5.0, n)
        self.m_k = rng.random(n)
        self._tables()

    def _tables(self):
        """The wall's noise worked out once, in its own frame (x along it, d behind its foot), and looked up from
        there every frame: the foam and the swell ride along with it."""
        cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cache', 'wall_noise_v1.npz')
        self.tx0, self.tdx, self.td0, self.tdd = -7200.0, 6.0, -900.0, 3.0
        if os.path.exists(cache):
            self.T = np.load(cache)['T']
        else:
            xs = np.arange(self.tx0, 7200.0 + 1e-6, self.tdx)
            ds = np.arange(self.td0, 3000.0 + 1e-6, self.tdd)
            Xg, Dg = np.meshgrid(xs, ds)
            sg = Dg / FACE
            self.T = np.stack([fbm2d(Xg / 420.0, Dg / 420.0, octaves=3, seed=14),
                               fbm2d(Xg / 22.0, sg * 1.4, octaves=3, seed=15),
                               fbm2d(Xg / 40.0, Dg / 40.0, octaves=2, seed=16),
                               fbm2d(Xg / 60.0, Dg / 60.0, octaves=2, seed=17),
                               fbm2d(Xg / 90.0, Dg / 90.0, octaves=3, seed=18)], -1).astype(np.float32)
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            np.savez(cache, T=self.T)
        ox = np.arange(-30000.0, 30000.0 + 1e-6, 6.0)
        self.ox = ox
        self.off_t = (170.0 * _n1(ox / 1500.0, 11) + 45.0 * _n1(ox / 300.0, 12, 2) + 0.000018 * ox ** 2)
        self.amp_t = A * (1.0 + 0.07 * _n1(ox / 1800.0, 13))

    def _noise(self, X, D):
        """The five noise fields (back swell, streaks, foot, lip, whitecaps) at points along x and d behind the
        foot: bilinear lookups in the wall's own tables."""
        nd, nx = self.T.shape[:2]
        fx = np.clip((X - self.tx0) / self.tdx, 0.0, nx - 1.001)
        fd = np.clip((D - self.td0) / self.tdd, 0.0, nd - 1.001)
        i0 = fx.astype(np.int32)
        j0 = fd.astype(np.int32)
        ax = (fx - i0)[..., None].astype(np.float32)
        ay = (fd - j0)[..., None].astype(np.float32)
        T = self.T
        return ((T[j0, i0] * (1 - ax) + T[j0, i0 + 1] * ax) * (1 - ay) +
                (T[j0 + 1, i0] * (1 - ax) + T[j0 + 1, i0 + 1] * ax) * ay)

    def grid(self):
        return self.nx, self.ny, self.x0, self.y0, self.nx * self.dx, self.ny * self.dx

    # -- the wall itself ---------------------------------------------------------------------------------------------
    def offset(self, x):
        """How far the foot of the wall runs ahead (-) or behind (+) along its length."""
        return np.interp(x, self.ox, self.off_t)

    def amp(self, x):
        return np.interp(x, self.ox, self.amp_t)

    def _field(self, X, Y, t):
        """Surface height, foam, crest glow and the flow (v towards the shore) of the wall at points X, Y."""
        yf = front(t) + self.offset(X)
        D = Y - yf
        s = D / FACE                                # 0 at the foot, 1 at the lip
        a = self.amp(X)
        N = self._noise(X, D)
        # the face: concave, steepest just under the lip; a lip on top; a long swell behind
        face = a * np.clip(s, 0.0, 1.0) ** 2.3
        lip = a * 0.03 * np.exp(-((s - 1.15) / 0.35) ** 2)
        back = a * 0.012 * N[..., 0] * _sm((s - 1.2) / 2.0)
        # ahead of it the sea drains away towards it
        draw = -DRAW * np.exp(-np.maximum(yf - Y, 0.0) / DRAW_L) * (1.0 - _sm(s / 0.5))
        eta = face + lip + back + draw
        # foam: churning at the foot, pouring down the face in streaks, a band of white along the lip, whitecaps
        streak = 0.5 + 0.5 * N[..., 1]
        toe = np.exp(-((s - 0.04) / 0.10) ** 2) * (0.7 + 0.5 * N[..., 2])
        down = _sm((s - 0.30) / 0.55) * (1.0 - _sm((s - 1.0) / 0.1)) * np.clip(streak * 1.9 - 0.85, 0.0, 0.9)
        lipf = np.exp(-((s - 1.0) / 0.16) ** 2) * (0.9 + 0.3 * N[..., 3])
        caps = _sm((s - 1.4) / 0.5) * np.clip(N[..., 4] * 2.2 - 0.25, 0.0, 0.7)
        foam = np.clip(toe + down + lipf + caps, 0.0, 1.3)
        crest = _sm((s - 0.78) / 0.2) * (1.0 - _sm((s - 1.15) / 0.15))
        # the flow: on the face the water rides with the wall and pours down it; behind, it lags; ahead, the sea
        # runs out to meet it
        onface = _sm(s / 0.15) * (1.0 - _sm((s - 1.1) / 0.3))
        v = (-(SPEED + 15.0) * onface - 0.3 * SPEED * _sm((s - 1.1) / 0.3)
             + 3.0 * (s < 0) * np.exp(-np.maximum(yf - Y, 0.0) / DRAW_L))
        return eta, foam, crest, v

    def _twirl(self, X, Y, q):
        """Where the water now at X, Y was before the sponge started pulling it round: turned back about the sponge,
        more so the nearer it is."""
        cx, cy = self.sponge[:2]
        dx, dy = X - cx, Y - cy
        r = np.hypot(dx, dy)
        if q <= 0.0:
            return X, Y, r
        phi = 1.6 * q * np.exp(-r / 1500.0)
        c, s = np.cos(-phi), np.sin(-phi)
        return cx + c * dx - s * dy, cy + s * dx + c * dy, r

    @staticmethod
    def reach(q):
        """How far out from the sponge the sea is already gone."""
        return 40.0 + 7200.0 * q ** 2.2

    def water(self, t, drink=0.0):
        q = float(np.clip(drink, 0.0, 1.0))
        X, Y = self.X, self.Y
        Xs, Ys, r = self._twirl(X, Y, q)
        eta, foam, crest, v = self._field(Xs, Ys, t)
        u = np.zeros_like(v)
        b = self.bed
        far = A
        if q > 0.0:
            # the sponge drinks: the sea is gone out to a circle that races outwards from it, and everything
            # still outside it sinks, the wall too
            R = self.reach(q)
            sink = (1.0 - _sm(q)) ** 1.3
            ring = _sm((r - R + 30.0) / 180.0)
            keep = sink * ring
            eta = b + np.maximum(eta - b, 0.0) * keep - (1.0 - keep) * 0.5
            edge = np.exp(-((r - R - 90.0) / 120.0) ** 2) * (1.0 - _sm((q - 0.85) / 0.15))
            foam = np.clip(foam * sink ** 1.5 + edge * 0.75, 0.0, 1.3)
            cx, cy = self.sponge[:2]
            ux, uy = (X - cx) / np.maximum(r, 1.0), (Y - cy) / np.maximum(r, 1.0)
            inflow = 45.0 * np.exp(-((r - R) / 260.0) ** 2)
            u = -ux * inflow
            v = v * sink - uy * inflow
            crest = crest * sink
            far = A * sink
        h = np.maximum(eta - b, 0.0)
        st, vel = WT.surface(eta.astype(np.float32), h.astype(np.float32), b, foam.astype(np.float32),
                             u.astype(np.float32), v.astype(np.float32), self.dx)
        wet = h > 0.03
        st[..., 3] = (crest * wet).astype(np.float32)
        return {'state': st, 'vel': vel, 'sea': 0.0, 'wave': (far, -1.0e4, 1.0, 1.0), 'wave2': (0.0, 0.0),
                'time': float(t), 'face_foam': 0.85, 'glow_k': 1.0, 'wfog': 0.24, 'wall_shadow': far}

    def mist(self, t, drink=0.0, alpha=1.0):
        """Puffs blowing off the lip and churning at the foot: pos3 size alpha fire variant age."""
        q = float(np.clip(drink, 0.0, 1.0))
        age = np.mod(t / self.m_life + self.m_ph, 1.0)
        x = self.m_x + 40.0 * age
        yf = front(t) + self.offset(x)
        a = self.amp(x)
        toe = self.m_toe
        y = np.where(toe, yf + FACE * (0.05 + 0.25 * age), yf + FACE * (0.9 + 0.35 * age))
        k = 0.6 + 0.8 * self.m_k
        z = np.where(toe, (10.0 + 160.0 * age) * k, a * (1.01 + 0.08 * age))
        size = np.where(toe, (60.0 + 190.0 * age) * k, (140.0 + 320.0 * age) * k)
        al = np.sin(np.pi * age) ** 0.8 * np.where(toe, 0.45, 0.75) * alpha * (1.0 - q) ** 2
        if q > 0.0:
            z = z * (1.0 - _sm(q)) ** 1.3
        out = np.zeros((len(x), 8), np.float32)
        out[:, 0], out[:, 1], out[:, 2] = x, y, z
        out[:, 3] = size
        out[:, 4] = al
        out[:, 6] = self.m_var
        out[:, 7] = t
        return out[al > 0.01]
