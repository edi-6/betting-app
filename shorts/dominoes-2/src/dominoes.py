"""The chain reactions of part 2. Every line is simulated with pymunk in its own plane: its dominoes stand along the
line and fall forwards, so each line is a straight 2D chain however it curves on the ground. A line is set off at a
given time with a given spin on its first domino (a punch, the domino before it, the giant's slam, lightning), and
records how each domino moves until the whole line has settled.

The lines, in order: the run (a punch sets it off; its last ten dominoes are the growth, each 1.4 times the one
before, up to the giant) -> the four race lines (all set off when the giant slams down) -> red stops at its gap;
the tail is set off by whichever of the others reaches the merge domino first -> nothing: the tail's last domino
lands a block short of the feeder. Then lightning strikes the middle of each feeder segment, both halves of each
segment run outwards, each feeder domino sets off its field column, and the field falls. All field columns are the
same chain, so one simulation of a column serves them all, shifted in time.

Everything here is in simulation time (s); the edit maps video time onto it.
"""
import numpy as np
import pymunk

import layout as LY

G = 60.0                          # gravity (blocks / s^2): dominoes a block tall fall like real, smaller ones
G_FIELD = 135.0                   # the field falls faster: its 230 blocks would take too long otherwise
HZ = 600
REC_HZ = 240
FR_GROUND, FR_DOM, ELAST = 0.8, 0.35, 0.05
PUNCH_W = 7.5                     # the punch: the first domino's spin (rad/s)
SLAM_W = 9.0                      # the giant's slam on the race lines' first dominoes
STRIKE_W = 10.0                   # lightning on the feeder dominoes it hits
FEED_DELAY = 0.05                 # from a feeder domino starting to fall to its column's first domino
STRIKE_GAPS = (0.0, 0.45, 0.9)    # the strikes: the middle, then the next pair out, then the outer pair
SLAM_SKEW = 0.0025                # the giant comes down a hair twisted: its west end first (s per block east)


class Line:
    """Recorded motion of one line: for domino k, the pose (dx along the line, z of its centre, pitch forward) at
    t0[k] + i / REC_HZ for i < n[k]; upright before t0, the last pose after."""

    def __init__(self, t0, traj, n, scales):
        self.t0 = np.asarray(t0, float)
        self.traj = traj            # (N, M, 3) float32, padded with each domino's last pose
        self.n = np.asarray(n)
        self.t_settle = self.t0 + self.n / REC_HZ
        self.rest = np.zeros((len(self.t0), 3))
        self.rest[:, 1] = LY.H * np.asarray(scales, float) / 2

    def poses(self, t, shift=0.0):
        """Poses of all dominoes at time t (shift: add to every start time)."""
        u = (t - (self.t0 + shift)) * REC_HZ
        i0 = np.clip(np.floor(u).astype(int), 0, self.traj.shape[1] - 1)
        i1 = np.clip(i0 + 1, 0, self.traj.shape[1] - 1)
        f = np.clip(u - np.floor(u), 0.0, 1.0)[:, None]
        k = np.arange(len(self.t0))
        p = self.traj[k, i0] * (1 - f) + self.traj[k, i1] * f
        before = ~(u >= 0)
        p[before] = self.rest[before]
        return p

    def pose_at(self, k, t):
        return self.poses(t)[k]


def simulate(n, kicks, t_max=60.0, virtual=0, g=G, scales=None, xs=None):
    """n dominoes (+ `virtual` extra ones after the last, to see when and how hard it would hit the next line).
    kicks: {k: (t, spin)}: domino k gets the forward spin at time t (it's knocked over then).
    scales: each domino's size (1 = W x H x T); xs: their base centres along the line (default every SPACING).
    Returns (Line for the n real dominoes, [(t, spin) when each virtual domino got hit])."""
    N = n + virtual
    sc = np.ones(N) if scales is None else np.r_[np.asarray(scales, float), np.ones(virtual)][:N]
    if xs is None:
        x = np.arange(N) * LY.SPACING
    else:
        x = np.r_[np.asarray(xs, float), xs[-1] + LY.SPACING * (1 + np.arange(virtual))][:N]
    sp = pymunk.Space()
    sp.gravity = (0.0, -g)
    sp.iterations = 30
    ground = pymunk.Segment(sp.static_body, (-3.0, 0.0), (x[-1] + 50.0 * sc.max(), 0.0), 0.0)
    ground.friction = FR_GROUND
    ground.elasticity = 0.0
    sp.add(ground)
    bodies = []
    for k in range(N):
        s = sc[k]
        m = s ** 3
        b = pymunk.Body(m, pymunk.moment_for_box(m, (LY.T * s, LY.H * s)))
        b.position = (x[k], LY.H * s / 2)
        sh = pymunk.Poly.create_box(b, (LY.T * s, LY.H * s), 0.0)
        sh.friction = FR_DOM
        sh.elasticity = ELAST
        sp.add(b, sh)
        bodies.append(b)
    dt = 1.0 / HZ
    t = min(tk for tk, _ in kicks.values())
    t = np.floor(t * HZ) / HZ
    pending = dict(kicks)
    started = np.full(N, np.inf)
    rec = [[] for _ in range(N)]
    vhit = [None] * virtual
    rec_every = HZ / REC_HZ
    sub = 0
    next_rec = 0.0
    quiet = 0
    while t < t_max:
        for k, (tk, w) in list(pending.items()):
            if t >= tk - 1e-9:
                bodies[k].angular_velocity = -w
                del pending[k]
        sp.step(dt)
        t += dt
        sub += 1
        moving = False
        for k, b in enumerate(bodies):
            if started[k] == np.inf and (abs(b.angle) > 0.004 or abs(b.angular_velocity) > 0.3):
                started[k] = t
                if k >= n:
                    vhit[k - n] = (t, -b.angular_velocity)
            if started[k] < np.inf and (abs(b.angular_velocity) > 0.02 or b.velocity.length > 0.02):
                moving = True
        if sub >= next_rec:
            next_rec += rec_every
            for k in range(n):
                if started[k] < np.inf:
                    b = bodies[k]
                    rec[k].append((b.position.x - x[k], b.position.y, -b.angle))
        quiet = 0 if (moving or pending) else quiet + 1
        if quiet > HZ * 0.25 and np.all(started[:n] < np.inf):
            break
        if quiet > HZ * 1.0:
            break                     # the rest never got hit (the line stops)
    M = max(len(r) for r in rec[:n]) if n else 1
    traj = np.zeros((n, max(M, 1), 3), np.float32)
    cnt = np.zeros(n, int)
    t0 = np.full(n, np.inf)
    for k in range(n):
        r = rec[k]
        if r:
            traj[k, :len(r)] = r
            traj[k, len(r):] = r[-1]
            cnt[k] = len(r)
            t0[k] = started[k]
        else:
            traj[k, :] = (0.0, LY.H * sc[k] / 2, 0.0)
    return Line(t0, traj, cnt, sc[:n]), vhit


class Chains:
    """All the lines of the video, set off in turn."""

    def __init__(self, layout, storm_after=2.8, verbose=True):
        self.L = layout
        A = layout.all
        self.idx = {}                                   # line id -> indices into layout.all (in k order)
        for line in np.unique(A['line']):
            ii = np.nonzero(A['line'] == line)[0]
            self.idx[int(line)] = ii[np.argsort(A['k'][ii])]
        # the run and the growth: one chain, ordinary dominoes then bigger and bigger ones
        im = self.idx[LY.LINE_MAIN]
        n_open = layout.n_open
        xs = np.r_[np.arange(n_open) * LY.SPACING, (n_open - 1) * LY.SPACING + (LY.GROW_Y - LY.OPEN_END_Y)]
        self.main, _ = simulate(len(im), {0: (0.0, PUNCH_W)}, scales=A['s'][im], xs=xs)
        gi = len(im) - 1
        if not np.isfinite(self.main.t0[gi]):
            raise RuntimeError('the growth stopped: the giant never fell')
        self.t_grow = float(self.main.t0[n_open])
        self.t_giant = float(self.main.t0[gi])
        # the giant is down when it's lying (pitch near its final value)
        tr = self.main.traj[gi, :self.main.n[gi]]
        k_land = int(np.argmax(tr[:, 2] >= tr[-1, 2] - 0.03))
        self.t_land = self.t_giant + k_land / REC_HZ
        self.t_race = self.t_land + 0.02
        # the race: red to its gap; the others to the merge domino (a virtual one, to see who gets there first)
        self.race = []
        arrive = [np.inf] * 4
        hits = [None] * 4
        self.t_slam = [self.t_race + SLAM_SKEW * (LY.LANES[j] - LY.LANES[0]) for j in range(4)]
        for j in range(4):
            n = len(self.idx[1 + j])
            if j == 0:
                ln, _ = simulate(n, {0: (self.t_slam[j], SLAM_W)})
            else:
                ln, vh = simulate(n, {0: (self.t_slam[j], SLAM_W)}, virtual=1)
                if vh[0] is not None:
                    arrive[j], hits[j] = vh[0][0], vh[0]
            self.race.append(ln)
        self.arrivals = arrive
        self.winner = int(np.argmin(arrive))
        self.t_merge, w_m = hits[self.winner]
        # the losers' last dominoes fall onto nothing (the merge domino has gone): no virtual one for them
        for j in range(1, 4):
            if j != self.winner:
                self.race[j], _ = simulate(len(self.idx[1 + j]), {0: (self.t_slam[j], SLAM_W)})
        self.t_red_stop = float(self.race[0].t_settle[np.isfinite(self.race[0].t0)].max())
        n_tail = len(self.idx[LY.LINE_TAIL])
        self.tail, _ = simulate(n_tail, {0: (self.t_merge, w_m)})
        self.t_stop = float(self.tail.t_settle.max())
        self.w_steady = w_m
        # the storm: lightning down the middle of each feeder segment (the middle one first, by the gap)
        self.t_strike0 = self.t_stop + storm_after
        sx = layout.strikes()
        order = np.argsort(np.abs(sx), kind='stable')
        self.strikes = []                               # (time, x)
        for rank, s in enumerate(order):
            self.strikes.append((self.t_strike0 + STRIKE_GAPS[(rank + 1) // 2], float(sx[s])))
        self.feed = {}
        for seg in range(LY.NSEG):
            ts = self.strikes[int(np.nonzero(order == seg)[0][0])][0]
            for half in (0, 1):
                line = LY.FEED0 + 2 * seg + half
                self.feed[line], _ = simulate(len(self.idx[line]), {0: (ts + 0.01, STRIKE_W)})
        # the canonical field column, set off at t = 0, falling under G_FIELD
        self.column, _ = simulate(layout.nrow, {0: (0.0, self.w_steady * np.sqrt(G_FIELD / G))}, g=G_FIELD,
                                  t_max=200.0)
        # each column starts when the feeder domino in front of it starts to fall
        self.col_start = np.zeros(layout.ncol)
        fi = np.arange(len(layout.run), len(layout.run) + len(layout.feeder))
        for line, ln in self.feed.items():
            ii = self.idx[line]
            c = np.rint(A['x'][ii] / LY.PITCH + (layout.ncol - 1) / 2.0).astype(int)
            self.col_start[c] = ln.t0 + FEED_DELAY
        del fi
        self.t_field0 = float(self.col_start.min())
        self.t_end = float((self.col_start + self.column.t_settle.max()).max())
        if verbose:
            print(f'[chains] growth from {self.t_grow:.2f}s, giant tips {self.t_giant:.2f}s, lands {self.t_land:.2f}s; '
                  f'race arrivals {np.round(self.arrivals, 3)} (red stops {self.t_red_stop:.2f}s), winner '
                  f'{LY.RACE_NAMES[self.winner]}; tail settled {self.t_stop:.2f}s; strikes from {self.t_strike0:.2f}s; '
                  f'field {self.t_field0:.2f}s .. {self.t_end:.2f}s (a column takes '
                  f'{self.column.t_settle.max():.2f}s)', flush=True)

    def lines(self):
        """line id -> Line, for every simulated line (the field columns aside)."""
        d = {LY.LINE_MAIN: self.main, LY.LINE_TAIL: self.tail}
        for j in range(4):
            d[1 + j] = self.race[j]
        d.update(self.feed)
        return d


def load(layout, cache_dir=None):
    """The Chains for a layout, from a cache keyed on the code that makes them."""
    import hashlib
    import os
    import pickle
    here = os.path.dirname(os.path.abspath(__file__))
    cache_dir = cache_dir or os.path.join(here, '..', 'cache')
    h = hashlib.md5()
    for f in ('layout.py', 'dominoes.py'):
        with open(os.path.join(here, f), 'rb') as fh:
            h.update(fh.read())
    path = os.path.join(cache_dir, f'chains_{h.hexdigest()[:10]}.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as fh:
            C = pickle.load(fh)
        C.L = layout
        return C
    C = Chains(layout)
    os.makedirs(cache_dir, exist_ok=True)
    L, C.L = C.L, None
    with open(path, 'wb') as fh:
        pickle.dump(C, fh)
    C.L = L
    return C


if __name__ == '__main__':
    import time
    t = time.time()
    L = LY.Layout()
    C = Chains(L)
    print(f'{time.time() - t:.1f}s')
    m = C.main
    print('growth hit times', np.round(m.t0[L.n_open - 1:], 3))
    print('giant final pose', np.round(m.traj[-1, -1], 3))
    v = (L.n_open - 20) * LY.SPACING / (m.t0[L.n_open - 1] - m.t0[19])
    print('run wave speed %.2f b/s' % v)
    col = C.column
    print('field column wave speed %.2f b/s' % ((L.nrow - 40) * LY.SPACING / (col.t0[-1] - col.t0[39])))
    print('tail last domino final pose', np.round(C.tail.traj[-1, -1], 3))
