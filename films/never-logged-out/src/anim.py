"""Animation helpers: keyframes with easing, splines, smooth noise (handheld / mouse hands), the first-person
camera (walking with view bobbing, looking around) and walking characters."""
import numpy as np


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def smoother(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (x * 6 - 15) + 10)


EASE = {'linear': lambda x: np.clip(x, 0.0, 1.0), 'smooth': smooth, 'smoother': smoother,
        'in': lambda x: np.clip(x, 0.0, 1.0) ** 2, 'out': lambda x: 1 - (1 - np.clip(x, 0.0, 1.0)) ** 2,
        'step': lambda x: np.where(x >= 1.0, 1.0, 0.0)}


class Keys:
    """Piecewise keyframes: [(t, value, ease_into_this_key), ...] (value scalar or vector). Before the first key: the
    first value; after the last: the last value."""

    def __init__(self, keys, ease='smooth'):
        ks = []
        for k in keys:
            if len(k) == 2:
                ks.append((float(k[0]), np.asarray(k[1], float), ease))
            else:
                ks.append((float(k[0]), np.asarray(k[1], float), k[2]))
        ks.sort(key=lambda k: k[0])
        self.k = ks

    def __call__(self, t):
        k = self.k
        if t <= k[0][0]:
            return k[0][1].copy()
        for i in range(1, len(k)):
            if t <= k[i][0]:
                t0, v0, _ = k[i - 1]
                t1, v1, e = k[i]
                x = (t - t0) / max(t1 - t0, 1e-9)
                return v0 + (v1 - v0) * EASE[e](x)
        return k[-1][1].copy()


def catmull(points, u):
    """Centripetal-ish uniform Catmull-Rom through points (N, D) at parameter u in [0, N-1]."""
    P = np.asarray(points, float)
    n = len(P)
    u = float(np.clip(u, 0, n - 1))
    i = min(int(np.floor(u)), n - 2)
    s = u - i
    p0 = P[max(i - 1, 0)]
    p1 = P[i]
    p2 = P[i + 1]
    p3 = P[min(i + 2, n - 1)]
    s2, s3 = s * s, s * s * s
    return 0.5 * ((2 * p1) + (-p0 + p2) * s + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s2 + (-p0 + 3 * p1 - 3 * p2 + p3) * s3)


class Path:
    """A spline through points, walked at a speed profile: arc-length parametrised."""

    def __init__(self, points, samples=400):
        self.P = np.asarray(points, float)
        us = np.linspace(0, len(self.P) - 1, samples)
        pts = np.array([catmull(self.P, u) for u in us])
        d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        self.us, self.pts, self.d = us, pts, d
        self.length = float(d[-1])

    def at(self, s):
        """Point at arc length s (clamped)."""
        s = float(np.clip(s, 0, self.length))
        u = np.interp(s, self.d, self.us)
        return catmull(self.P, u)

    def tangent(self, s, h=0.05):
        a = self.at(s - h)
        b = self.at(s + h)
        v = b - a
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])[:len(v)]


def value_noise1(t, seed=0):
    """Smooth 1D noise in [-1, 1]."""
    rng = np.random.default_rng(seed)
    table = rng.uniform(-1, 1, 4096)
    i = int(np.floor(t)) % 4095
    f = t - np.floor(t)
    f = f * f * (3 - 2 * f)
    return table[i] * (1 - f) + table[i + 1] * f


def fbm1(t, seed=0, octaves=3):
    v, a, fr, norm = 0.0, 1.0, 1.0, 0.0
    for o in range(octaves):
        v += a * value_noise1(t * fr, seed + 17 * o)
        norm += a
        a *= 0.5
        fr *= 2.0
    return v / norm


def dir_from(yaw_deg, pitch_deg):
    """yaw 0 = looking +y (north), positive = turning left (counter-clockwise from above); pitch positive = up."""
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    return np.array([-np.sin(y) * np.cos(p), np.cos(y) * np.cos(p), np.sin(p)])


def yaw_of(v):
    """The yaw (degrees) that looks along the horizontal part of v."""
    return float(np.degrees(np.arctan2(-v[0], v[1])))


def look_angles(eye, target):
    d = np.asarray(target, float) - np.asarray(eye, float)
    yaw = yaw_of(d)
    pitch = float(np.degrees(np.arctan2(d[2], np.hypot(d[0], d[1]))))
    return yaw, pitch


def unwrap_keys(keys):
    """Unwrap yaw keyframes so they turn the short way."""
    out = []
    prev = None
    for k in keys:
        t, v = k[0], float(k[1])
        if prev is not None:
            while v - prev > 180:
                v -= 360
            while v - prev < -180:
                v += 360
        out.append((t, v) + tuple(k[2:]))
        prev = v
    return out


class POV:
    """The player's eyes: feet position keys (or a path + speed keys), yaw/pitch keys, view bobbing while walking and
    a little mouse-hand jitter. cam(t) -> dict for the renderer; bob(t) -> hand offsets."""

    EYE = 1.62

    def __init__(self, pos_keys=None, yaw_keys=None, pitch_keys=None, path=None, s_keys=None, jitter=0.18, seed=1,
                 fov=70.0, bob=1.0, eye=EYE):
        self.pos_keys = Keys(pos_keys) if pos_keys is not None else None
        self.path = path
        self.s_keys = Keys(s_keys, 'linear') if s_keys is not None else None
        self.yaw = Keys(unwrap_keys(yaw_keys)) if yaw_keys is not None else None
        self.pitch = Keys(pitch_keys) if pitch_keys is not None else Keys([(0, 0.0)])
        self.jitter = jitter
        self.seed = seed
        self.fov = fov
        self.bob_amt = bob
        self.eye_h = eye
        self.fov_keys = None

    def feet(self, t):
        if self.path is not None:
            return self.path.at(float(self.s_keys(t)))
        return self.pos_keys(t)

    def speed(self, t, h=0.05):
        return float(np.linalg.norm(self.feet(t + h)[:2] - self.feet(t - h)[:2]) / (2 * h))

    def dist(self, t):
        if self.path is not None:
            return float(self.s_keys(t))
        # integrate horizontal distance coarsely
        n = max(2, int(t * 12))
        ts = np.linspace(0, t, n)
        pts = np.array([self.pos_keys(x)[:2] for x in ts])
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    def angles(self, t):
        yaw = float(self.yaw(t)) if self.yaw is not None else 0.0
        pitch = float(self.pitch(t))
        j = self.jitter
        yaw += j * fbm1(t * 0.9, self.seed) + 0.3 * j * fbm1(t * 3.1, self.seed + 5)
        pitch += 0.7 * j * fbm1(t * 0.8, self.seed + 9)
        return yaw, pitch

    def bob(self, t):
        v = self.speed(t)
        amp = float(np.clip(v / 4.0, 0.0, 1.0)) * self.bob_amt
        ph = self.dist(t) * np.pi / 1.1
        return amp, ph

    def cam(self, t):
        f = self.feet(t)
        amp, ph = self.bob(t)
        eye = f + np.array([0.0, 0.0, self.eye_h])
        yaw, pitch = self.angles(t)
        d = dir_from(yaw, pitch)
        right = np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw)), 0.0])
        eye = eye + np.array([0.0, 0.0, -0.06 * amp * abs(np.cos(ph))]) + right * 0.03 * amp * np.sin(ph)
        fov = self.fov if self.fov_keys is None else float(self.fov_keys(t))
        return dict(eye=eye, target=eye + d, fov=fov, roll=0.5 * amp * np.sin(ph))

    def hand_bob(self, t):
        amp, ph = self.bob(t)
        return (0.03 * amp * np.sin(ph), 0.04 * amp * abs(np.cos(ph)))

    def steps(self, t0, t1, stride=1.1):
        """Times of footsteps between t0 and t1 (every `stride` blocks of walking)."""
        out = []
        n = int((t1 - t0) * 24)
        last = None
        for k in range(n + 1):
            t = t0 + k / 24.0
            s = self.dist(t) / stride
            if last is not None and np.floor(s) > np.floor(last):
                out.append(t)
            last = s
        return out


class Walker:
    """A character moving along a path with a speed profile (s_keys: arc length over time); faces the way it walks
    unless yaw_keys say otherwise."""

    def __init__(self, path, s_keys, yaw_keys=None, stride=1.3):
        self.path = path
        self.s = Keys(s_keys, 'linear') if not isinstance(s_keys, Keys) else s_keys
        self.yaw_keys = Keys(unwrap_keys(yaw_keys)) if yaw_keys is not None else None
        self.stride = stride

    def pos(self, t):
        return self.path.at(float(self.s(t)))

    def speed(self, t, h=0.06):
        return abs(float(self.s(t + h)) - float(self.s(t - h))) / (2 * h)

    def yaw(self, t):
        if self.yaw_keys is not None:
            return float(self.yaw_keys(t))
        # face the direction of travel (look a little ahead so turns are smooth)
        s = float(self.s(t))
        tan = self.path.tangent(min(s + 0.3, self.path.length))
        return yaw_of(tan)

    def pose(self, actor, t):
        actor.pos = np.asarray(self.pos(t), float)
        actor.yaw = np.radians(self.yaw(t))
        v = self.speed(t)
        actor.walk_amp = float(np.clip(v / 2.5, 0.0, 1.0))
        actor.walk = float(self.s(t)) / self.stride * np.pi
        actor.idle_t = t
        return actor
