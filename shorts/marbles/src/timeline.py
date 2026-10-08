"""The edit: a list of shots. Each shot runs for a while of video and maps its frames to simulation time (with speed
changes and slow motion inside a shot, and jumps at the cuts) and to a camera. After the last marble lands the
simulation is over and the story's own clock takes over (the fuse, the blast): times past the landing are 'after'.

Camera keys are (u, eye, target, fov) with u the fraction of the shot; between keys the camera eases (Catmull-Rom
through the keys, smoothstep in time). A camera can also follow something: key eye/target given as callables of
the sim time.
"""
import numpy as np

import machine as M

FPS = 60
MY, MZ = M.MY, M.MZ

FUSE_AT = 1.7          # seconds after the last marble lands: the creeper's fuse is lit (the hiss)
FUSE_LEN = 1.5         # like the game's
BOOM_AT = FUSE_AT + FUSE_LEN


def _ss(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _pchip(xs, ys, x):
    """Monotone cubic interpolation (for time maps: no overshoot, so time never runs backwards)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if len(xs) == 2:
        return float(np.interp(x, xs, ys))
    from scipy.interpolate import PchipInterpolator
    return float(PchipInterpolator(xs, ys)(np.clip(x, xs[0], xs[-1])))


def catmull(ps, u):
    """Position along a Catmull-Rom spline through ps at u in [0, 1] (uniform in the number of segments)."""
    ps = [np.asarray(p, float) for p in ps]
    if len(ps) == 1:
        return ps[0]
    n = len(ps) - 1
    s = np.clip(u, 0.0, 1.0) * n
    i = min(int(s), n - 1)
    f = s - i
    p0 = ps[max(i - 1, 0)]
    p1, p2 = ps[i], ps[i + 1]
    p3 = ps[min(i + 2, n)]
    return 0.5 * ((2 * p1) + (-p0 + p2) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f * f +
                  (-p0 + 3 * p1 - 3 * p2 + p3) * f * f * f)


class Shot:
    def __init__(self, name, dur, times, cams, shake=None, near=None):
        """times: [(u, ts)] (u in 0..1 of the shot); cams: [(u, eye, target, fov)]; shake: f(ts) -> amplitude."""
        self.name, self.dur = name, dur
        self.times = times
        self.cams = cams
        self.shake = shake
        self.near = near

    def ts(self, u):
        us = [a for a, _ in self.times]
        vs = [b for _, b in self.times]
        return _pchip(us, vs, u)

    def cam(self, u, ts, resolve):
        us = np.array([c[0] for c in self.cams])
        k = int(np.clip(np.searchsorted(us, u, side='right') - 1, 0, len(us) - 1))
        if k >= len(us) - 1:
            e, t, f = resolve(self.cams[-1][1], ts), resolve(self.cams[-1][2], ts), self.cams[-1][3]
            return np.asarray(e, float), np.asarray(t, float), float(f)
        u0, u1 = us[k], us[k + 1]
        w = _ss((u - u0) / max(u1 - u0, 1e-9))
        g = (k + w) / (len(us) - 1)
        eyes = [resolve(c[1], ts) for c in self.cams]
        tgts = [resolve(c[2], ts) for c in self.cams]
        fov = self.cams[k][3] * (1 - w) + self.cams[k + 1][3] * w
        return catmull(eyes, g), catmull(tgts, g), float(fov)


class Timeline:
    def __init__(self, sim, follow, level):
        """sim: the sim's data (events); follow: f(ts) -> the last marble's position (world); level: f(ts) -> how
        high the pile is (sim z)."""
        ev = sim['events']
        self.t_jam, self.t_save = float(ev[0]), float(ev[1])
        self.t_ram = self.t_save - 0.12
        self.t_end = float(ev[2])
        self.t_hero, self.t_land = float(ev[4]), float(ev[5])
        L = self.t_land
        R = self.t_ram
        cx, cz = 6.4, MZ + 30.0                               # the creeper's chest in the picture (world x, z)
        hero = follow

        def hero_eye(ts):
            return hero(ts) + np.array([1.2, -7.5, 1.6])

        def hero_tgt(ts):
            return hero(ts) + np.array([0.0, 0.0, -0.8])

        def lvl(dx, dy, dz):
            return lambda ts: np.array([dx, MY + dy, MZ + level(ts) + dz])

        gx = 6.5                                              # the goat's line (goat.Script)
        self.shots = [
            Shot('open', 2.8, [(0, 0.0), (1, 2.8)],
                 [(0.0, (-4, MY - 12, 143), (0, MY + 1, 114), 62),
                  (0.3, (-4, MY - 16, 139), (0, MY, 111), 62),
                  (0.7, (2, MY - 34, 94), (0, MY, 88), 60),
                  (1.0, (4, MY - 50, 76), (0, MY, 68), 60)], near=(0, MY, 84)),
            Shot('pegs', 2.2, [(0, 2.8), (1, 5.5)],
                 [(0.0, (7.5, MY - 10.5, MZ + 86.5), (2.0, MY, MZ + 85.0), 50),
                  (1.0, (4.5, MY - 9.5, MZ + 84.0), (0.5, MY, MZ + 83.0), 50)], near=(0, MY, MZ + 85)),
            Shot('rise', 2.3, [(0, 5.5), (1, 8.3)],
                 [(0.0, lvl(-6, -24, 5.0), lvl(0, 0, -2.0), 50),
                  (1.0, lvl(-3, -23, 5.5), lvl(0, 0, -1.5), 50)], near=(0, MY, MZ + 12)),
            Shot('gold', 3.1, [(0, 8.3), (1, 11.4)],
                 [(0.0, (-7, MY - 9, 134), (-2.5, MY + 0.5, 112), 54),
                  (1.0, (-3, MY - 8, 129), (0, MY, 110), 50)], near=(0, MY, 112)),
            Shot('stuck', 1.2, [(0, 11.4), (1, 12.6)],
                 [(0.0, (1, MY - 24, MZ + 93), (0, MY, MZ + 95), 52),
                  (1.0, (1, MY - 20, MZ + 93), (0, MY, MZ + 95), 52)], near=(0, MY, MZ + 92)),
            Shot('goat_a', 2.2, [(0, 12.6), (1, R - 1.25)],
                 [(0.0, (3.4, MY - 17.6, 1.35), (7.4, MY - 12.0, 1.45), 48),
                  (1.0, (3.9, MY - 16.9, 1.25), (6.8, MY - 12.2, 1.25), 46)], near=(gx, MY - 11, 1)),
            Shot('goat_b', 1.2, [(0, R - 1.25), (0.6, R - 0.42), (1.0, R - 0.13)],
                 [(0.0, (gx - 1.7, MY - 6.45, 0.75), (gx - 0.4, MY - 14.0, 1.05), 62),
                  (1.0, (gx - 1.7, MY - 6.45, 0.7), (gx - 0.2, MY - 12.0, 0.95), 64)], near=(gx, MY - 9, 1)),
            Shot('goat_c', 1.0, [(0, R - 0.13), (0.55, R + 0.0), (1.0, R + 0.14)],
                 [(0.0, (gx + 4.4, MY - 11.6, 1.6), (gx - 0.8, MY - 8.2, 1.0), 56),
                  (1.0, (gx + 4.0, MY - 11.0, 1.5), (gx - 0.5, MY - 7.2, 0.9), 56)],
                 shake=lambda ts: self.ram_shake(ts) * 1.4, near=(gx, MY - 8, 1)),
            Shot('pop', 1.1, [(0, R + 0.12), (1, R + 1.0)],
                 [(0.0, (3, MY - 24, 131), (0, MY + 2, 117), 58),
                  (1.0, (3, MY - 26, 133), (0, MY + 4, 120), 58)], shake=lambda ts: self.ram_shake(ts) * 0.6,
                 near=(0, MY, 115)),
            Shot('burst', 1.5, [(0, R + 1.0), (1, R + 2.5)],
                 [(0.0, (1, MY - 21, MZ + 93), (0, MY, MZ + 94), 52),
                  (1.0, (1, MY - 25, MZ + 92), (0, MY, MZ + 92), 54)], near=(0, MY, MZ + 92)),
            Shot('fill_a', 1.9, [(0, R + 2.5), (1, R + 5.4)],
                 [(0.0, lvl(-24, -40, 5.0), lvl(0, 0, 0.0), 46),
                  (1.0, lvl(-22, -40, 5.0), lvl(0, 0, 0.0), 46)], near=(0, MY, MZ + 30)),
            Shot('fill_b', 1.9, [(0, R + 5.4), (1, R + 8.3)],
                 [(0.0, lvl(3.0, -13, 2.6), lvl(0, 0, -0.6), 48),
                  (1.0, lvl(2.0, -12, 2.6), lvl(0, 0, -0.6), 48)], near=(0, MY, MZ + 45)),
            Shot('fill_c', 2.0, [(0, R + 8.3), (1, R + 11.3)],
                 [(0.0, (2, MY - 92, 40), (0, MY, 40), 50),
                  (1.0, (2, MY - 88, 41), (0, MY, 41), 50)], near=(0, MY, 40)),
            Shot('count', 0.9, [(0, R + 11.3), (0.8, self.t_hero - 0.15), (1, self.t_hero)],
                 [(0.0, (1, MY - 26, MZ + 92), (0, MY, MZ + 95), 52),
                  (1.0, (1, MY - 22, MZ + 93), (0, MY, MZ + 95), 52)], near=(0, MY, MZ + 92)),
            Shot('last', 3.7, [(0, self.t_hero), (0.75, L - 0.55), (1, L + 0.05)],
                 [(0.0, hero_eye, hero_tgt, 46), (1.0, hero_eye, hero_tgt, 46)], near=None),
            Shot('done', 2.6, [(0, L + 0.05), (1, L + 2.65)],
                 [(0.0, (0.8, MY - 14, MZ + 66), (0.2, MY, MZ + 63), 50),
                  (0.45, (1.5, MY - 70, MZ + 40), (1.0, MY, MZ + 36), 54),
                  (1.0, (2.0, MY - 80, MZ + 37), (1.2, MY, MZ + 34), 54)], near=(0, MY, 40)),
            Shot('fuse', 1.4, [(0, L + 2.65), (1, L + BOOM_AT)],
                 [(0.0, (2.5, MY - 72, MZ + 38), (2.0, MY, MZ + 35), 52),
                  (1.0, (cx * 0.6, MY - 44, MZ + 42), (cx, MY, MZ + 40), 50)],
                 shake=lambda ts: 0.05 * _ss((ts - L - FUSE_AT) / FUSE_LEN), near=(cx, MY, cz)),
            Shot('boom', 2.3, [(0, L + BOOM_AT), (0.3, L + BOOM_AT + 0.05), (1, L + BOOM_AT + 1.1)],
                 [(0.0, (cx * 0.6, MY - 44, MZ + 42), (cx, MY, MZ + 40), 50),
                  (1.0, (cx * 0.6, MY - 47, MZ + 41), (cx, MY, MZ + 38), 54)],
                 shake=lambda ts: self.boom_shake(ts), near=(cx, MY - 14, 30)),
            Shot('after', 2.1, [(0, L + BOOM_AT + 1.1), (1, L + BOOM_AT + 3.8)],
                 [(0.0, (32, MY - 90, 42), (-4, MY + 12, 25), 54),
                  (1.0, (28, MY - 94, 40), (-4, MY + 10, 22), 54)], near=(0, MY - 30, 0)),
        ]
        self.starts = np.cumsum([0.0] + [s.dur for s in self.shots])
        self.duration = float(self.starts[-1])
        self.n_frames = int(round(self.duration * FPS))
        self.cx, self.cz = cx, cz

    # -- shakes ------------------------------------------------------------------------------------
    def ram_shake(self, ts):
        d = ts - self.t_ram
        return 0.0 if d < 0 else 0.45 * np.exp(-d * 5.0)

    def boom_shake(self, ts):
        d = ts - (self.t_land + BOOM_AT)
        return 0.0 if d < 0 else 1.1 * np.exp(-d * 2.5)

    # -- per frame ---------------------------------------------------------------------------------
    def at(self, i):
        """(shot, u, ts) for frame i."""
        v = (i + 0.5) / FPS
        k = int(np.clip(np.searchsorted(self.starts, v, side='right') - 1, 0, len(self.shots) - 1))
        s = self.shots[k]
        u = (v - self.starts[k]) / s.dur
        return s, u, s.ts(u)

    def frames(self):
        out = []
        for i in range(self.n_frames):
            out.append(self.at(i))
        return out

    def camera(self, i):
        s, u, ts = self.at(i)

        def resolve(x, ts_):
            return x(ts_) if callable(x) else np.asarray(x, float)
        eye, tgt, fov = s.cam(u, ts, resolve)
        if s.shake is not None:
            a = s.shake(ts)
            if a > 0:
                v = (i + 0.5) / FPS
                off = np.array([np.sin(v * 61.0), np.sin(v * 47.0 + 1.3), np.sin(v * 53.0 + 2.1)]) * a
                eye = eye + off * 0.6
                tgt = tgt + off
        return {'eye': eye, 'target': tgt, 'fov': fov, 'shot': s.name, 'ts': ts, 'near': s.near}
