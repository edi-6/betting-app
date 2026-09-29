"""The camera: one continuous move, like a drone following the chain reaction.

POV at the start (the player's eyes, the arm punches the first domino), then it lifts off and chases the wave,
climbs over the spiral and looks down on it, drops in behind the three racing lines, sits low at the finish for the
photo finish, watches the run cross the bridge from the bank, follows it to the field, stops low beside the gap
where the run comes up short, turns to the creeper as it walks in, pushes in as its fuse burns, recoils from the
blast, and cranes up over the field as it falls until it looks straight down on the whole picture.

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


class Fronts:
    """Where the wave is: along the run, on each racing line, and up the field's middle."""

    def __init__(self, layout, chains):
        A = layout.all
        self.C = chains
        seq = []
        for line, ln in ((LY.LINE_MAIN, chains.main), (1 + chains.winner, chains.branches[chains.winner]),
                         (LY.LINE_TAIL, chains.tail)):
            ii = chains.idx[line]
            for k, i in enumerate(ii):
                if np.isfinite(ln.t0[k]):
                    seq.append((ln.t0[k], A['x'][i], A['y'][i]))
        seq = np.array(sorted(seq))
        self.run_t, self.run_xy = seq[:, 0], seq[:, 1:]
        self.race = []
        for b in range(3):
            ii = chains.idx[1 + b]
            ln = chains.branches[b]
            self.race.append((ln.t0, np.stack([A['x'][ii], A['y'][ii]], -1)))
        c = LY.NCOL // 2
        self.col_t = chains.column.t0 + chains.col_start[c]
        self.first = np.array([A['x'][chains.idx[LY.LINE_MAIN][0]], A['y'][chains.idx[LY.LINE_MAIN][0]]])

    def run(self, t):
        return np.array([np.interp(t, self.run_t, self.run_xy[:, 0]), np.interp(t, self.run_t, self.run_xy[:, 1])])

    def race_mean(self, t):
        ys = [np.interp(t, t0, xy[:, 1]) for t0, xy in self.race]
        return float(np.mean(ys))

    def field_row_y(self, t):
        k = np.interp(t, self.col_t, np.arange(len(self.col_t)), left=-1.0)
        return LY.FIELD_Y0 + k * LY.SPACING


class Director:
    def __init__(self, layout, chains, ev):
        self.L = layout
        self.F = Fronts(layout, chains)
        self.ev = ev
        self.field_cy = LY.FIELD_Y0 + (layout.nrow - 1) * LY.SPACING / 2.0
        self.creeper = None                         # set by main: fn(t) -> feet position

    # -- the shots: each returns (eye, target, fov) ---------------------------------------------
    def pov(self, t):
        x0, y0 = self.F.first
        bob = 0.012 * np.sin(t * 5.0)
        return (np.array([x0, y0 - 1.25, EYE_H + bob]), np.array([x0, y0 + 5.0, -0.35]), 70.0)

    def chase(self, t, back=6.0, up=3.2, side=1.5, ahead=4.0, fov=55.0):
        f = self.F.run(t)
        g = self.F.run(t + 0.35)
        d = g - f
        n = np.linalg.norm(d)
        d = d / n if n > 1e-6 else np.array([0.0, 1.0])
        # stay behind the front along the local heading, a little to the side
        s = np.array([d[1], -d[0]])
        eye = np.array([*(f - d * back + s * side), up])
        tgt = np.array([*(f + d * ahead), 0.0])
        return eye, tgt, fov

    def lift(self, t):
        u = _ss(0.35, 2.2, t)
        e0, t0, f0 = self.pov(t)
        e1, t1, f1 = self.chase(t, back=6.5, up=4.2, side=1.8, ahead=5.0)
        return e0 * (1 - u) + e1 * u, t0 * (1 - u) + t1 * u, f0 * (1 - u) + f1 * u

    def spiral(self, t):
        c = np.array([*LY.SPIRAL_C, 0.0])
        u = _ss(2.3, 8.0, t)
        phi = -0.35 + 0.6 * u
        eye = c + np.array([7.5 * np.sin(phi), -7.5 * np.cos(phi), 21.0])
        return eye, c + np.array([0.0, 0.8, 0.0]), 52.0

    def race(self, t):
        fy = self.F.race_mean(t)
        return np.array([0.4, fy - 10.5, 5.2]), np.array([0.0, fy + 3.5, 0.0]), 62.0

    def finish(self, t):
        my = LY.MERGE[1]
        return np.array([2.3, my - 4.6, 1.35]), np.array([0.0, my - 0.4, 0.3]), 48.0

    def bridge(self, t):
        f = self.F.run(t)
        fy = float(np.clip(f[1], LY.BRIDGE[0] - 4.0, LY.BRIDGE[1] + 3.0))
        return np.array([7.2, fy - 4.0, 2.3]), np.array([0.0, fy + 1.0, 0.1]), 50.0

    def gap(self, t):
        u = _ss(self.ev['stop'] - 1.2, self.ev['walk'] + TL.WALK, t)
        eye = np.array([2.7 - 0.4 * u, 12.3 + 0.5 * u, 1.5])
        tgt = np.array([0.35, 17.9, 0.45])
        if self.creeper is not None:
            # look towards the creeper while it walks in
            w = _ss(self.ev['walk'] - 0.3, self.ev['walk'] + 0.6, t) * (1 - _ss(self.ev['turn'] - 0.4,
                                                                                     self.ev['turn'] + 0.2, t))
            cp = np.asarray(self.creeper(t), float) + np.array([0.0, 0.0, 0.9])
            tgt = tgt * (1 - 0.55 * w) + cp * (0.55 * w)
        return eye, tgt, 44.0

    def fuse(self, t):
        u = _ss(self.ev['turn'], self.ev['blast'], t)
        e0, t0, f0 = self.gap(t)
        c = np.asarray(self.creeper(t), float) if self.creeper is not None else np.array([1.1, 16.4, 0.0])
        e1 = np.array([2.05, 13.25, 1.3])
        t1 = c + np.array([0.0, 0.0, 1.0])
        return e0 * (1 - u) + e1 * u, t0 * (1 - u) + t1 * u, f0 * (1 - u) + 38.0 * u

    def blast(self, t):
        u = _ss(self.ev['blast'], self.ev['blast'] + 0.9, t)
        e0, t0, f0 = self.fuse(self.ev['blast'])
        e1 = np.array([2.6, 7.5, 4.2])
        t1 = np.array([0.5, 19.0, 0.6])
        return e0 * (1 - u) + e1 * u, t0 * (1 - u) + t1 * u, f0 * (1 - u) + 48.0 * u

    def field(self, t):
        b, e = self.ev['blast'] + 0.9, self.ev['end']
        u = _ss(b, e, t)
        fy = self.F.field_row_y(t)
        eye = np.array([0.0, 7.5 + 38.0 * u ** 1.3, 4.2 + 118.0 * u ** 1.6])
        tgt_y = np.clip(fy + 4.0, LY.FIELD_Y0 + 4.0, self.field_cy)
        tgt = np.array([0.0, tgt_y * (1 - u ** 2) + self.final()[1][1] * u ** 2, 0.0])
        return eye, tgt, 48.0 - 6.0 * u

    def final(self, t=None):
        # straight down on the whole picture, a little low in the frame (the counter sits on top)
        cy = self.field_cy + 1.2
        return np.array([0.0, cy - 0.01, 116.0]), np.array([0.0, cy, 0.0]), 40.0

    # -- the whole path ---------------------------------------------------------------------------
    def shots(self):
        ev = self.ev
        return [
            (TL.T0, self.lift),
            (2.3, self.spiral),
            (ev['split'] - 0.8, self.race),
            (ev['merge'] - 0.45, self.finish),
            (ev['merge'] + 0.35, self.bridge),
            (ev['merge'] + 2.3, lambda t: self.chase(t, back=6.5, up=3.2, side=1.4, ahead=4.5)),
            (ev['stop'] - 1.3, self.gap),
            (ev['turn'], self.fuse),
            (ev['blast'], self.blast),
            (ev['blast'] + 0.9, self.field),
            (ev['end'] + 0.2, self.final),
        ]

    def raw(self, t):
        sh = self.shots()
        blend = {1: 1.4, 2: 0.9, 3: 0.2, 4: 0.5, 5: 0.9, 6: 0.8, 7: 0.3, 8: 0.05, 9: 0.5, 10: 1.6}
        k = max(i for i, (ts, _) in enumerate(sh) if t >= ts) if t >= sh[0][0] else 0
        e, g, f = sh[k][1](t)
        if k > 0:
            w = blend.get(k, 0.5)
            u = _ss(sh[k][0], sh[k][0] + w, t)
            if u < 1.0:
                e0, g0, f0 = sh[k - 1][1](t)
                e, g, f = e0 * (1 - u) + e * u, g0 * (1 - u) + g * u, f0 * (1 - u) + f * u
        return np.asarray(e, float), np.asarray(g, float), float(f)

    def cameras(self, times, smooth_s=0.18):
        raw = [self.raw(t) for t in times]
        E = np.array([r[0] for r in raw])
        G = np.array([r[1] for r in raw])
        Fv = np.array([r[2] for r in raw])
        # zero-phase smoothing, lighter where the shots need to be snappy (the punch, the blast)
        a = 1.0 / max(1.0, smooth_s * TL.FPS)
        for arr in (E, G, Fv):
            for i in range(1, len(arr)):
                arr[i] = arr[i - 1] + (arr[i] - arr[i - 1]) * a
            for i in range(len(arr) - 2, -1, -1):
                arr[i] = arr[i + 1] + (arr[i] - arr[i + 1]) * a
        # keep the POV exact (the arm is drawn relative to it)
        pov = times < 0.35
        for i in np.nonzero(pov)[0]:
            e, g, f = self.pov(times[i])
            E[i], G[i], Fv[i] = e, g, f
        cams = []
        for i, t in enumerate(times):
            e, g = E[i].copy(), G[i].copy()
            # a touch of handheld drift, none in the POV and the top-down view
            hand = 0.0 if (t < 0.35 or t > self.ev['end'] - 0.5) else 1.0
            ph = t * np.array([0.9, 1.3, 1.1])
            e += hand * 0.04 * np.sin(ph + np.array([0.0, 1.7, 3.1]))
            # the blast shakes it
            if t >= self.ev['blast']:
                k = np.exp(-(t - self.ev['blast']) / 0.25)
                e += k * 0.35 * np.sin(np.array([47.0, 53.0, 61.0]) * (t - self.ev['blast']))
            f = g - e
            down = np.degrees(np.arcsin(np.clip(-f[2] / np.linalg.norm(f), -1, 1)))
            w = _ss(50.0, 80.0, down)
            up = np.array([0.0, 0.0, 1.0]) * (1 - w) + np.array([0.0, 1.0, 0.0]) * w
            cams.append({'eye': tuple(e), 'target': tuple(g), 'fov': float(Fv[i]), 'up': tuple(up / np.linalg.norm(up))})
        return cams
