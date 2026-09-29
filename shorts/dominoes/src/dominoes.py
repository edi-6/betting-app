"""The chain reactions. Every line is simulated with pymunk in its own plane: its dominoes stand at s = k * SPACING
along the line and fall forwards, so each line is a straight 2D chain however it curves on the ground. A line is
set off at a given time with a given spin on its first domino (a punch, the domino before it, the blast), and
records how each domino moves until the whole line has settled.

The lines, in order: the main line (a punch sets it off) -> the three race lines (all set off by the main line's
last domino) -> the tail (set off by whichever race line arrives first) -> nothing: the tail's last domino lands a
block short of the feeder. Then the blast sets off both feeder lines (and knocks over what's close), each feeder
domino sets off its field column, and the field falls. All field columns are the same chain, so one simulation of
a column serves them all, shifted in time.

Everything here is in simulation time (s); the edit maps video time onto it.
"""
import numpy as np
import pymunk

import layout as LY

G = 60.0                          # gravity (blocks / s^2): dominoes a block tall fall like real, smaller ones
HZ = 600
REC_HZ = 240
FR_GROUND, FR_DOM, ELAST = 0.8, 0.35, 0.05
PUNCH_W = 7.5                     # the punch: the first domino's spin (rad/s)
FEED_DELAY = 0.05                 # from a feeder domino starting to fall to its column's first domino


class Line:
    """Recorded motion of one line: for domino k, the pose (dx along the line, z of its centre, pitch forward) at
    t0[k] + i / REC_HZ for i < n[k]; upright before t0, the last pose after."""

    def __init__(self, t0, traj, n):
        self.t0 = np.asarray(t0, float)
        self.traj = traj            # (N, M, 3) float32, padded with each domino's last pose
        self.n = np.asarray(n)
        self.t_settle = self.t0 + self.n / REC_HZ

    def poses(self, t, shift=0.0):
        """Poses of all dominoes at time t (shift: add to every start time)."""
        u = (t - (self.t0 + shift)) * REC_HZ
        i0 = np.clip(np.floor(u).astype(int), 0, self.traj.shape[1] - 1)
        i1 = np.clip(i0 + 1, 0, self.traj.shape[1] - 1)
        f = np.clip(u - np.floor(u), 0.0, 1.0)[:, None]
        k = np.arange(len(self.t0))
        p = self.traj[k, i0] * (1 - f) + self.traj[k, i1] * f
        rest = np.array([0.0, LY.H / 2, 0.0])
        p[u < 0] = rest
        return p


def simulate(n, kicks, t_max=40.0, virtual=0):
    """n dominoes (+ `virtual` extra ones after the last, to see when and how hard it would hit the next line).
    kicks: {k: (t, spin)}: domino k gets the forward spin at time t (it's knocked over then).
    Returns (Line for the n real dominoes, [(t, spin) when each virtual domino got hit])."""
    sp = pymunk.Space()
    sp.gravity = (0.0, -G)
    sp.iterations = 30
    N = n + virtual
    ground = pymunk.Segment(sp.static_body, (-3.0, 0.0), (N * LY.SPACING + 10.0, 0.0), 0.0)
    ground.friction = FR_GROUND
    ground.elasticity = 0.0
    sp.add(ground)
    bodies = []
    for k in range(N):
        b = pymunk.Body(1.0, pymunk.moment_for_box(1.0, (LY.T, LY.H)))
        b.position = (k * LY.SPACING, LY.H / 2)
        s = pymunk.Poly.create_box(b, (LY.T, LY.H), 0.0)
        s.friction = FR_DOM
        s.elasticity = ELAST
        sp.add(b, s)
        bodies.append(b)
    dt = 1.0 / HZ
    t = min(tk for tk, _ in kicks.values())
    t = np.floor(t * HZ) / HZ
    pending = dict(kicks)
    started = np.full(N, np.inf)
    rec = [[] for _ in range(N)]
    vhit = [None] * virtual
    step_rec = HZ // REC_HZ if HZ % REC_HZ == 0 else None
    sub = 0
    rec_every = HZ / REC_HZ
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
                    rec[k].append((b.position.x - k * LY.SPACING, b.position.y, -b.angle))
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
            traj[k, :] = (0.0, LY.H / 2, 0.0)
    del step_rec
    return Line(t0, traj, cnt), vhit


class Chains:
    """All the lines of the video, set off in turn."""

    def __init__(self, layout, t_blast_after=6.0, verbose=True):
        self.L = layout
        A = layout.all
        self.idx = {}                                   # line id -> indices into layout.all (in k order)
        for line in np.unique(A['line']):
            ii = np.nonzero(A['line'] == line)[0]
            self.idx[int(line)] = ii[np.argsort(A['k'][ii])]
        n_main = len(self.idx[LY.LINE_MAIN])
        self.main, vh = simulate(n_main, {0: (0.0, PUNCH_W)}, virtual=1)
        self.t_split, w = vh[0]
        self.w_steady = w
        self.branches = []
        arrive = []
        for b in range(3):
            nb = len(self.idx[1 + b])
            ln, vh = simulate(nb, {0: (self.t_split, w)}, virtual=1)
            self.branches.append(ln)
            arrive.append(vh[0])
        self.winner = int(np.argmin([a[0] for a in arrive]))
        self.arrivals = [a[0] for a in arrive]
        self.t_merge, w_m = arrive[self.winner]
        n_tail = len(self.idx[LY.LINE_TAIL])
        self.tail, _ = simulate(n_tail, {0: (self.t_merge, w_m)})
        self.t_stop = float(self.tail.t_settle.max())
        self.t_blast = self.t_stop + t_blast_after
        if verbose:
            print(f'[chains] split {self.t_split:.2f}s, arrivals {np.round(self.arrivals, 2)}, winner {self.winner}, '
                  f'tail settled {self.t_stop:.2f}s, blast {self.t_blast:.2f}s', flush=True)

    def field_lines(self, blast):
        """The blast (x, y, knock radius): feeders and field columns. blast is set by the creeper's position."""
        bx, by, R = blast
        A = self.L.all
        self.feed = {}
        for line in (LY.FEED_E, LY.FEED_W):
            ii = self.idx[line]
            x = A['x'][ii]
            kicks = {}
            for k, xi in enumerate(x):
                d = np.hypot(xi - bx, LY.FEEDER_Y - by)
                if d < R:
                    kicks[k] = (self.t_blast + d / 40.0, self.w_steady * (1.0 + 0.8 * (1 - d / R)))
            if not kicks:
                kicks = {0: (self.t_blast, self.w_steady)}
            self.feed[line], _ = simulate(len(ii), kicks)
        # the canonical field column, set off at t = 0
        self.column, _ = simulate(self.L.nrow, {0: (0.0, self.w_steady)})
        # each column's start: when the feeder domino in front of it starts to fall, or the blast if closer
        cx = (np.arange(LY.NCOL) - (LY.NCOL - 1) / 2.0) * LY.PITCH
        start = np.zeros(LY.NCOL)
        for c, x in enumerate(cx):
            line = LY.FEED_E if x > 0 else LY.FEED_W
            ii = self.idx[line]
            k = int(np.argmin(np.abs(A['x'][ii] - x)))
            start[c] = self.feed[line].t0[k] + FEED_DELAY
            d = np.hypot(x - bx, LY.FIELD_Y0 - by)
            if d < R:
                start[c] = min(start[c], self.t_blast + d / 40.0)
        self.col_start = start
        self.t_end = float((start + self.column.t_settle.max()).max())
        print(f'[chains] field from {start.min():.2f}s to {self.t_end:.2f}s '
              f'(column takes {self.column.t_settle.max():.2f}s)', flush=True)


if __name__ == '__main__':
    import time
    t = time.time()
    L = LY.Layout()
    C = Chains(L)
    C.field_lines((1.0, 17.8, 4.0))
    print(f'{time.time() - t:.1f}s')
    m = C.main
    v = (len(m.t0) - 20) * LY.SPACING / (m.t0[-1] - m.t0[19])
    print('main: first hit times', np.round(m.t0[:6], 3), 'wave speed %.2f b/s' % v)
    print('tail last domino final pose', np.round(C.tail.traj[-1, -1], 3), 'next-to-last', np.round(C.tail.traj[-2, -1], 3))
