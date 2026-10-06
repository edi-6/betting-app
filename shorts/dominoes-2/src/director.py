"""The camera: one continuous move, like a drone following the chain reaction.

POV at the start (the player's eyes, the arm punches the first domino), then it lifts off and chases the wave, slides
out to the side of the growth and pulls back as each domino outgrows the last, drops to the ground beside where the
giant will land and looks up as it comes down (it slams down right beside the lens), rises behind the four racing
lines, sits low at the finish for the photo finish, watches the run cross the bridge, follows it to the field, stops
low beside the gap where it comes up short and waits in the silence as the storm comes in; lightning strikes right in
front of it, and it swings up and round the field in a long arc, up onto the cliff, where it settles on the edge
looking south over the field as the words form. At the end it turns round.

Shots are functions of simulation time that often track the wave front; they're blended into each other over short
windows and the whole path is smoothed so the camera glides.
"""
import numpy as np

import layout as LY
import timeline as TL

EYE_H = 1.62


def _ss(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def final_view():
    eye = np.array(LY.FINAL_EYE, float)
    p = np.radians(LY.FINAL_PITCH)
    return eye, eye + np.array([0.0, -np.cos(p), np.sin(p)]) * 40.0, LY.FINAL_FOV


class Fronts:
    """Where the wave is: along the run, on each racing line, the growth, the giant's top, up the field."""

    def __init__(self, layout, chains):
        A = layout.all
        self.C = chains
        seq = []
        lines = [(LY.LINE_MAIN, chains.main), (1 + chains.winner, chains.race[chains.winner]),
                 (LY.LINE_TAIL, chains.tail)]
        for line, ln in lines:
            ii = chains.idx[line]
            for k, i in enumerate(ii):
                if np.isfinite(ln.t0[k]) and A['s'][i] < 1.01:
                    seq.append((ln.t0[k], A['x'][i], A['y'][i]))
        seq = np.array(sorted(seq))
        self.run_t, self.run_xy = seq[:, 0], seq[:, 1:]
        self.race = []
        for j in range(4):
            ii = chains.idx[1 + j]
            ln = chains.race[j]
            ok = np.isfinite(ln.t0)
            self.race.append((ln.t0[ok], np.stack([A['x'][ii][ok], A['y'][ii][ok]], -1)))
        im = chains.idx[LY.LINE_MAIN]
        g0 = layout.n_open - 1
        self.grow_t = chains.main.t0[g0:]
        self.grow_s = np.r_[1.0, LY.GROW_S]
        self.grow_y = np.r_[LY.OPEN_END_Y, LY.GROW_Y]
        gi = len(im) - 1
        self.giant = (chains.main.t0[gi], chains.main.traj[gi, :chains.main.n[gi]])
        c = layout.ncol // 2
        self.col_t = chains.column.t0 + chains.col_start[c]
        self.first = np.array([A['x'][im[0]], A['y'][im[0]]])

    def run(self, t):
        return np.array([np.interp(t, self.run_t, self.run_xy[:, 0]), np.interp(t, self.run_t, self.run_xy[:, 1])])

    def race_mean(self, t, lines=(0, 1, 2, 3)):
        ys = [np.interp(t, *(self.race[j][0], self.race[j][1][:, 1])) for j in lines]
        return float(np.mean(ys))

    def growth(self, t):
        """Size and y of the growth domino that's falling now (continuous, so the camera can follow smoothly)."""
        s = float(np.exp(np.interp(t, self.grow_t, np.log(self.grow_s))))
        y = float(np.interp(t, self.grow_t, self.grow_y))
        return s, y

    def giant_top(self, t):
        """The middle of the giant's top edge (y, z)."""
        t0, tr = self.giant
        k = int(np.clip((t - t0) * 240.0, 0, len(tr) - 1))
        dx, z, p = tr[k] if t >= t0 else (0.0, LY.H * LY.GIANT_S / 2, 0.0)
        h = LY.H * LY.GIANT_S / 2
        return LY.GIANT_Y + dx + h * np.sin(p), z + h * np.cos(p)

    def field_row_y(self, t):
        k = np.interp(t, self.col_t, np.arange(len(self.col_t)), left=-1.0)
        return LY.FIELD_Y0 + k * LY.SPACING


class Director:
    def __init__(self, layout, chains, ev):
        self.L = layout
        self.C = chains
        self.F = Fronts(layout, chains)
        self.ev = ev

    # -- the shots: each returns (eye, target, fov) ---------------------------------------------
    def pov(self, t):
        x0, y0 = self.F.first
        bob = 0.012 * np.sin(t * 5.0)
        return (np.array([x0, y0 - 1.25, EYE_H + bob]), np.array([x0 + 0.6, y0 + 6.0, 0.2]), 64.0)

    def chase(self, t, back=6.0, up=3.2, side=1.5, ahead=4.0, fov=55.0, tz=0.0):
        f = self.F.run(t)
        g = self.F.run(t + 0.35)
        d = g - f
        n = np.linalg.norm(d)
        d = d / n if n > 1e-6 else np.array([0.0, 1.0])
        s = np.array([d[1], -d[0]])
        eye = np.array([*(f - d * back + s * side), up])
        tgt = np.array([*(f + d * ahead), tz])
        return eye, tgt, fov

    def lift(self, t):
        u = _ss(0.35, 2.0, t)
        e0, t0, f0 = self.pov(t)
        e1, t1, f1 = self.chase(t, back=6.5, up=3.4, side=1.8, ahead=6.0, tz=2.2, fov=56.0)
        return e0 * (1 - u) + e1 * u, t0 * (1 - u) + t1 * u, f0 * (1 - u) + f1 * u

    def growth(self, t):
        """From the east side, pulling back and up as the dominoes grow (the one falling stays in frame)."""
        s, y = self.F.growth(t)
        D = 5.0 + 1.45 * s
        eye = np.array([D * 0.92, y - 0.42 * D, 0.9 + 0.36 * s])
        tgt = np.array([0.0, y + 0.45 * s, 0.42 * s])
        return eye, tgt, 58.0

    def giant(self, t):
        """On the ground beside where the giant's top comes down, looking up at it as it falls."""
        y_top, z_top = self.F.giant_top(t)
        eye = np.array([12.0, LY.RACE_Y0 + 3.0, 1.5])
        tgt = np.array([0.0, y_top - 2.0, max(z_top - 1.0, 1.0)])
        land = _ss(self.ev['land'] - 0.2, self.ev['land'] + 0.8, t)
        tgt = tgt * (1 - land) + np.array([-2.0, LY.RACE_Y0 + 14.0, 0.6]) * land
        return eye, tgt, 74.0 - 12.0 * land

    def race(self, t):
        fy = self.F.race_mean(t)
        return np.array([0.6, fy - 11.0, 6.0]), np.array([0.0, fy + 4.0, 0.0]), 62.0

    def finish(self, t):
        my = LY.MERGE_Y
        return np.array([3.6, my + 4.5, 4.0]), np.array([-0.9, my - 1.7, 0.3]), 54.0

    def bridge(self, t):
        f = self.F.run(t)
        fy = float(np.clip(f[1], LY.BRIDGE[0] - 4.0, LY.BRIDGE[1] + 3.0))
        return np.array([7.2, fy - 4.0, 2.3]), np.array([0.0, fy + 1.0, 0.1]), 50.0

    def gap(self, t):
        """Low beside the gap; in the silence it creeps in, then tilts up towards the cliff as the storm comes."""
        ev = self.ev
        u = _ss(ev['stop'] - 1.2, ev['strike'], t)
        eye = np.array([2.7 - 0.5 * u, 12.3 + 0.9 * u, 1.5 + 0.2 * u])
        up = _ss(ev['stop'] + 0.6, ev['strike'] - 0.2, t)
        tgt = np.array([0.35, 17.9, 0.45]) * (1 - up) + np.array([0.2, 120.0, 30.0]) * up
        return eye, tgt, 46.0 + 8.0 * up

    def orbit(self, t):
        """After the first strike: up and round the east side of the field in a long arc, onto the cliff's edge."""
        ev = self.ev
        t0, t1 = ev['strike'] + 0.35, ev['arrive']
        u = _ss(t0, t1, t)
        e_end, g_end, f_end = final_view()
        c = np.array([0.0, (LY.FIELD_Y0 + LY.CLIFF_Y) / 2])
        e0 = np.array([2.2, 13.2, 1.7])
        a0 = np.arctan2(e0[1] - c[1], e0[0] - c[0])            # about -pi/2 (south of the middle)
        a1 = np.arctan2(e_end[1] - c[1], e_end[0] - c[0])      # about +pi/2 (north of it)
        a = a0 + (a1 - a0) * u                                   # anticlockwise: round the east side
        r0, r1 = np.hypot(*(e0[:2] - c)), np.hypot(*(e_end[:2] - c))
        r = r0 + (r1 - r0) * u + 34.0 * np.sin(np.pi * u)
        z = e0[2] + (e_end[2] - e0[2]) * u ** 0.6 + 30.0 * np.sin(np.pi * u)      # up quickly, off the field
        eye = np.array([c[0] + r * np.cos(a), c[1] + r * np.sin(a), z])
        # look at the front of the wave early on, then at where the words are, then the last shot's target
        fy = float(np.clip(self.F.field_row_y(t), LY.FIELD_Y0, LY.FIELD_Y0 + 120.0))
        g_mid = np.array([0.0, fy + 10.0, 0.0])
        w = _ss(0.55, 1.0, u)
        tgt = g_mid * (1 - w) + g_end * w
        return eye, tgt, 60.0 * (1 - w) + f_end * w

    def final(self, t):
        return final_view()

    def turn(self, t):
        """He's behind you: the camera turns round to face north, and he's right there."""
        ev = self.ev
        u = _ss(ev['turn'], ev['turn'] + TL.TURN, t)
        e, g, f = final_view()
        hb = np.array(TL.HEROBRINE_END) + np.array([0.0, 0.0, 1.7])
        d0 = g - e
        d1 = hb - e
        a0 = np.arctan2(d0[1], d0[0])
        a1 = np.arctan2(d1[1], d1[0])
        if a1 - a0 > np.pi:
            a1 -= 2 * np.pi
        if a1 - a0 < -np.pi:
            a1 += 2 * np.pi
        # turn to the right (east first): through -90 degrees... whichever is the shorter way round
        a = a0 + (a1 - a0) * u
        p0 = np.arcsin(d0[2] / np.linalg.norm(d0))
        p1 = np.arcsin(d1[2] / np.linalg.norm(d1))
        p = p0 + (p1 - p0) * u
        d = np.array([np.cos(a) * np.cos(p), np.sin(a) * np.cos(p), np.sin(p)])
        return e, e + d * 3.0, f + (56.0 - f) * u

    # -- the whole path ---------------------------------------------------------------------------
    def shots(self):
        ev = self.ev
        return [
            (TL.T0, self.lift, 0.0),
            (ev['grow'] - 0.7, self.growth, 0.9),
            (ev['giant'] - 0.05, self.giant, 0.85),
            (ev['land'] + 0.55, self.race, 0.9),
            (ev['red_end'] - 0.45, self.finish, 0.35),
            (ev['merge'] + 0.45, self.bridge, 0.5),
            (ev['merge'] + 2.3, lambda t: self.chase(t, back=6.5, up=3.2, side=1.4, ahead=4.5), 0.9),
            (ev['stop'] - 1.3, self.gap, 0.8),
            (ev['strike'] + 0.35, self.orbit, 0.35),
            (ev['arrive'], self.final, 0.0),
            (ev['turn'], self.turn, 0.0),
        ]

    def raw(self, t):
        sh = self.shots()
        k = max(i for i, (ts, _, _) in enumerate(sh) if t >= ts) if t >= sh[0][0] else 0
        e, g, f = sh[k][1](t)
        if k > 0 and sh[k][2] > 0:
            u = _ss(sh[k][0], sh[k][0] + sh[k][2], t)
            if u < 1.0:
                e0, g0, f0 = sh[k - 1][1](t)
                e, g, f = e0 * (1 - u) + e * u, g0 * (1 - u) + g * u, f0 * (1 - u) + f * u
        return np.asarray(e, float), np.asarray(g, float), float(f)

    def cameras(self, times, smooth_s=0.16):
        raw = [self.raw(t) for t in times]
        E = np.array([r[0] for r in raw])
        G = np.array([r[1] for r in raw])
        Fv = np.array([r[2] for r in raw])
        # zero-phase smoothing so the path glides
        a = 1.0 / max(1.0, smooth_s * TL.FPS)
        for arr in (E, G, Fv):
            for i in range(1, len(arr)):
                arr[i] = arr[i - 1] + (arr[i] - arr[i - 1]) * a
            for i in range(len(arr) - 2, -1, -1):
                arr[i] = arr[i + 1] + (arr[i] - arr[i + 1]) * a
        # keep the POV exact (the arm is drawn relative to it), and the last shot and the turn
        exact = (times < 0.35) | (times >= self.ev['arrive'] + 0.6)
        for i in np.nonzero(exact)[0]:
            e, g, f = self.raw(times[i])
            E[i], G[i], Fv[i] = e, g, f
        cams = []
        ev = self.ev
        for i, t in enumerate(times):
            e, g = E[i].copy(), G[i].copy()
            # a touch of handheld drift, none in the POV, on the cliff, or in the turn
            hand = 0.0 if (t < 0.35 or t > ev['arrive']) else 1.0
            ph = t * np.array([0.9, 1.3, 1.1])
            e += hand * 0.04 * np.sin(ph + np.array([0.0, 1.7, 3.1]))
            # the slam and the strike shake it
            for te, amp, tau in ((ev['land'], 0.32, 0.30), (ev['strike'], 0.22, 0.22)):
                if t >= te:
                    k = np.exp(-(t - te) / tau)
                    e += k * amp * np.sin(np.array([47.0, 53.0, 61.0]) * (t - te))
                    g += k * amp * 1.5 * np.sin(np.array([41.0, 59.0, 37.0]) * (t - te) + 1.0)
            f = g - e
            down = np.degrees(np.arcsin(np.clip(-f[2] / np.linalg.norm(f), -1, 1)))
            w = _ss(60.0, 85.0, down)
            fh = f[:2] / max(np.linalg.norm(f[:2]), 1e-6)
            up = np.array([0.0, 0.0, 1.0]) * (1 - w) + np.array([fh[0], fh[1], 0.0]) * w
            cams.append({'eye': tuple(e), 'target': tuple(g), 'fov': float(Fv[i]), 'up': tuple(up / np.linalg.norm(up))})
        return cams
