"""The flight: one closed loop through the Sift on an elytra, timed to the music, and the rider's eye along it.

The player stands on the tip of a siftslate spire looking down into the valley, steps off, falls, opens the elytra,
dives, and flies a clockwise circuit of the valley: north through a grove of white trees, a firework, east through
the Carapace and the ribs of a colossal fossil, south low over the ichor lake past its falls, west climbing between
coral towers into the low sun, a second firework, and back north to the spire, landing where the flight began. The
video loops: its last frame is its first.

The path is a Catmull-Rom spline through the control points below, resampled by arc length. Its timing is authored:
the arc length reached at each beat of the music (128 bpm, eight bars = 15 s) is keyed, and a monotone curve through
the keys gives the speed. The view banks into the turns from the lateral acceleration, like a bird's (an elytra flyer
leans the way a plane does), and widens with speed.
"""
import numpy as np

BPM = 128.0
BAR = 240.0 / BPM            # 1.875 s
DURATION = 8 * BAR           # 15.0 s
FPS = 60
EYE = 1.62                   # the eye above the feet
SPIRE = (0.5, 0.5, 100.0)    # the middle of the spire's top face (the player stands at its north-east corner)
G = 14.0

# the eye's path (x east, y north, z up): from the spire's north-east corner round the valley and back
CONTROL = [
    (2.3, 2.3, 101.62),       # standing at the corner, looking north-east over the valley
    (3.4, 3.4, 101.25),       # stepping off
    (5.8, 5.8, 97.5),         # falling
    (10.0, 10.5, 87.0),       # the elytra opens
    (18.0, 20.0, 71.0),       # diving
    (27.0, 31.0, 58.0),
    (35.0, 44.0, 51.5),       # the bottom of the dive
    (42.0, 60.0, 51.0),       # into the grove
    (48.0, 78.0, 52.5),
    (55.0, 96.0, 51.0),
    (63.0, 113.0, 52.0),
    (76.0, 128.0, 52.5),      # first firework (bar 4)
    (94.0, 139.0, 51.5),      # into the Carapace: the fossil's ribs
    (116.0, 142.0, 50.5),
    (136.0, 133.0, 50.0),
    (149.0, 114.0, 49.0),
    (153.0, 92.0, 47.0),      # low over the lake, past the falls (bar 5)
    (152.0, 68.0, 46.5),
    (147.0, 44.0, 46.5),
    (136.0, 22.0, 48.0),
    (118.0, 4.0, 52.0),       # between the coral towers, climbing into the sun (bar 6)
    (96.0, -10.0, 58.0),
    (72.0, -20.0, 66.0),
    (48.0, -26.0, 74.0),      # second firework (bar 7)
    (24.0, -30.0, 82.0),
    (4.0, -31.6, 88.5),
    (-11.4, -28.4, 93.5),     # round behind the spire
    (-14.9, -19.9, 97.5),
    (-11.4, -11.4, 102.0),
    (-5.0, -5.0, 103.3),      # the approach
    (2.3, 2.3, 101.62),       # landed where it began
]
STAND_AZ = 45.0               # the standing look: north-east, down into the valley
STAND_PITCH = -24.0
FOV = 84.0                    # standing (it widens with speed)

# the speed (blocks/s) is keyed at these times; the arc length is its integral, scaled to land exactly at the end
T_LAND = 14.35
V_KEYS = [(0.0, 0.0), (0.47, 0.0), (0.78, 7.0), (1.1, 24.0), (1.5, 48.0), (1.875, 58.0), (2.3, 52.0), (3.0, 44.0),
          (4.2, 39.0), (5.5, 37.0), (5.625, 38.0), (5.95, 54.0), (7.0, 50.0), (8.5, 46.0), (9.6, 42.0),
          (10.6, 37.0), (11.25, 35.0), (11.6, 45.0), (12.4, 36.0), (13.125, 26.0), (13.8, 11.0), (T_LAND, 0.0),
          (DURATION, 0.0)]
BANK_MAX = 68.0
ROLL_MAX = 34.0             # the view rolls into a banked turn, but not all the way


def _catmull_rom(P, n=40):
    """Centripetal Catmull-Rom through the points (the ends extended), n samples per span."""
    P = np.asarray(P, float)
    ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        t0 = 0.0
        t1 = t0 + np.linalg.norm(p1 - p0) ** 0.5 + 1e-6
        t2 = t1 + np.linalg.norm(p2 - p1) ** 0.5 + 1e-6
        t3 = t2 + np.linalg.norm(p3 - p2) ** 0.5 + 1e-6
        for u in np.linspace(t1, t2, n, endpoint=False):
            a1 = (t1 - u) / (t1 - t0) * p0 + (u - t0) / (t1 - t0) * p1
            a2 = (t2 - u) / (t2 - t1) * p1 + (u - t1) / (t2 - t1) * p2
            a3 = (t3 - u) / (t3 - t2) * p2 + (u - t2) / (t3 - t2) * p3
            b1 = (t2 - u) / (t2 - t0) * a1 + (u - t0) / (t2 - t0) * a2
            b2 = (t3 - u) / (t3 - t1) * a2 + (u - t1) / (t3 - t1) * a3
            out.append((t2 - u) / (t2 - t1) * b1 + (u - t1) / (t2 - t1) * b2)
    out.append(P[-1])
    return np.array(out)


def _pchip(x, y, xq):
    """Monotone cubic interpolation (Fritsch-Carlson)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    h = np.diff(x)
    d = np.diff(y) / h
    m = np.zeros_like(y)
    for k in range(1, len(y) - 1):
        if d[k - 1] * d[k] > 0:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
    xq = np.atleast_1d(np.asarray(xq, float))
    i = np.clip(np.searchsorted(x, xq) - 1, 0, len(h) - 1)
    t = (xq - x[i]) / h[i]
    h00, h10 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t
    h01, h11 = -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
    return h00 * y[i] + h10 * h[i] * m[i] + h01 * y[i + 1] + h11 * h[i] * m[i + 1]


class Flight:
    STEP = 0.1

    def __init__(self):
        dense = _catmull_rom(CONTROL)
        seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
        sd = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(sd[-1])
        self.s = np.arange(0.0, self.length, self.STEP)
        self.P = np.stack([np.interp(self.s, sd, dense[:, k]) for k in range(3)], 1)
        T = np.gradient(self.P, axis=0)
        self.T = T / np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9)
        # arc length of each control point (the nearest sample to it)
        self.s_ctrl = [float(self.s[int(np.argmin(np.linalg.norm(self.P - np.asarray(c), axis=1)))])
                       for c in CONTROL]
        self.s_ctrl[-1] = self.s[-1]
        kt = np.array([k[0] for k in V_KEYS])
        kv = np.array([k[1] for k in V_KEYS])
        tt = np.arange(0.0, DURATION + 1e-9, 1.0 / 600)
        self.tt = tt
        vt = np.maximum(_pchip(kt, kv, tt), 0.0)
        st = np.concatenate([[0.0], np.cumsum(0.5 * (vt[1:] + vt[:-1]) * np.diff(tt))])
        scale = self.length / st[-1]
        self.st = np.minimum(st * scale, self.length)
        self.vt = vt * scale
        # banking: lean into the turn, roll = atan(v^2 k / g) from the horizontal curvature
        th = self.T.copy()
        th[:, 2] = 0.0
        th /= np.maximum(np.linalg.norm(th, axis=1, keepdims=True), 1e-9)
        yaw = np.unwrap(np.arctan2(th[:, 0], th[:, 1]))
        kappa = np.gradient(yaw) / self.STEP                       # + = turning right (clockwise)
        v_s = np.interp(self.s, np.interp(tt, tt, self.st), self.vt)
        want = np.degrees(np.arctan2(v_s ** 2 * kappa, G))
        want = np.clip(want, -BANK_MAX, BANK_MAX)
        sig = 5.0 / self.STEP
        k = np.arange(-int(3 * sig), int(3 * sig) + 1)
        w = np.exp(-0.5 * (k / sig) ** 2)
        w /= w.sum()
        self.bank = np.convolve(np.pad(want, len(k) // 2, mode='edge'), w, mode='valid')

    # -- sampling ------------------------------------------------------------------------------
    def s_at(self, t):
        return float(np.interp(t % DURATION, self.tt, self.st))

    def v_at(self, t):
        return float(np.interp(t % DURATION, self.tt, self.vt))

    def _lerp(self, A, s):
        f = np.clip(s / self.STEP, 0, len(self.s) - 1.000001)
        i = int(f)
        u = f - i
        return A[i] * (1 - u) + A[i + 1] * u

    def pos(self, s):
        return self._lerp(self.P, s)

    def tangent(self, s):
        T = self._lerp(self.T, s)
        return T / np.linalg.norm(T)

    def bank_at(self, s):
        return float(self._lerp(self.bank, s))

    def time_at(self, s):
        """The first time the flight reaches arc length s."""
        return float(np.interp(s, self.st[self.tt <= T_LAND], self.tt[self.tt <= T_LAND]))

    def summary(self):
        out = [f'flight: {self.length:.0f} blocks, {DURATION:.2f} s, top speed {self.vt.max():.1f} blocks/s, '
               f'bank {self.bank.min():.0f}..{self.bank.max():.0f} deg']
        for k, c in enumerate(CONTROL):
            s = self.s_ctrl[k]
            t = self.time_at(s)
            out.append(f'  ctrl {k:2d}  s {s:6.1f}  t {t:6.2f}  v {self.v_at(t):5.1f}  z {c[2]:6.1f}')
        return '\n'.join(out)


def soft_clip(x, lo, hi, k):
    """x kept within about lo..hi, easing into the limits over k."""
    if x > hi - k:
        x = hi - k + k * np.tanh((x - (hi - k)) / k)
    if x < lo + k:
        x = lo + k - k * np.tanh(((lo + k) - x) / k)
    return x


def smooth(a, b, x):
    u = np.clip((x - a) / (b - a), 0.0, 1.0)
    return u * u * (3 - 2 * u)


def view(fl, t):
    """The rider's eye at time t: (cam dict, eye, (F, R, U), speed). Standing on the spire at the start and the end
    (looking down the drop, breathing), flying in between."""
    t = t % DURATION
    s = fl.s_at(t)
    v = fl.v_at(t)
    eye = fl.pos(s)
    T = fl.tangent(s)
    Z = np.array([0.0, 0.0, 1.0])
    # the flying look: along the path and at a point ahead on it
    la = float(np.clip(6.0 + v * 0.32, 6.0, 18.0))
    to = fl.pos(min(s + la, fl.length - 0.01)) - eye
    to /= max(np.linalg.norm(to), 1e-6)
    F_fly = T * 0.45 + to * 0.55
    F_fly /= np.linalg.norm(F_fly)
    # the head doesn't tip right back on a climb or right down in a dive
    el = np.degrees(np.arcsin(np.clip(F_fly[2], -1, 1)))
    el = soft_clip(el, -52.0, 7.0, 7.0)
    hz = np.array([F_fly[0], F_fly[1], 0.0])
    hz /= max(np.linalg.norm(hz), 1e-6)
    F_fly = hz * np.cos(np.radians(el)) + Z * np.sin(np.radians(el))
    # the standing look: down over the valley
    pitch, az = np.radians(STAND_PITCH), np.radians(STAND_AZ)
    F_stand = np.array([np.cos(pitch) * np.cos(az), np.cos(pitch) * np.sin(az), np.sin(pitch)])
    # the approach: swinging round behind the spire the head turns to its top (where they'll land), then to the
    # view they'll land facing
    top = np.array([SPIRE[0], SPIRE[1], SPIRE[2] + 0.6])
    to_top = top - eye
    to_top /= max(np.linalg.norm(to_top), 1e-6)
    u = smooth(12.95, 13.9, t)
    F_app = to_top * (1 - u) + F_stand * u
    F_app /= np.linalg.norm(F_app)
    w_app = smooth(11.35, 12.6, t)
    F_fly = F_fly * (1 - w_app) + F_app * w_app
    F_fly /= np.linalg.norm(F_fly)
    fly = smooth(0.40, 0.95, t) * (1.0 - smooth(13.95, 14.75, t))
    F = F_stand * (1 - fly) + F_fly * fly
    F /= np.linalg.norm(F)
    # up: banked in flight (the head keeps a little of the world's up)
    bank = np.radians(fl.bank_at(s)) * fly
    Th = np.array([F[0], F[1], 0.0])
    Th /= max(np.linalg.norm(Th), 1e-6)
    Rh = np.array([Th[1], -Th[0], 0.0])
    up0 = Z - np.dot(Z, F) * F
    up0 /= np.linalg.norm(up0)
    roll = np.radians(ROLL_MAX) * np.tanh(bank * 0.85 / np.radians(ROLL_MAX))
    up = up0 * np.cos(roll) + Rh * np.sin(roll)
    up = up - np.dot(up, F) * F
    up /= np.linalg.norm(up)
    # breathing while standing; a buffet of wind at speed; a kick when a firework fires (every wobble has a whole
    # number of cycles in the video, so the loop has no seam)
    w0 = 2 * np.pi / DURATION
    stand = 1.0 - fly
    bob = np.array([0.0, 0.0, 0.035 * np.sin(t * w0 * 6)]) * stand
    amp = 0.0025 + 0.010 * smooth(20.0, 50.0, v)
    jx = np.sin(t * w0 * 55) * np.sin(t * w0 * 17) * amp
    jy = np.sin(t * w0 * 45 + 1.3) * np.sin(t * w0 * 13) * amp
    R = np.cross(F, up)
    F2 = F + R * jx + up * jy
    F2 /= np.linalg.norm(F2)
    eye2 = eye + bob
    fov = FOV + 22.0 * smooth(8.0, 50.0, v) + fw_kick(t) * 7.0
    cam = dict(eye=eye2, target=eye2 + F2, fov=float(fov), up=up)
    return cam, eye2, (F2, np.cross(F2, up), up), v


FIREWORKS = (5.625, 11.25)


def fw_kick(t):
    """How hard a firework is pushing right now (0..1): a quick rise, a slower fade."""
    k = 0.0
    for tf in FIREWORKS:
        d = t - tf
        if 0 <= d < 1.4:
            k = max(k, min(1.0, d / 0.06) * np.exp(-d / 0.45))
    return k


if __name__ == '__main__':
    fl = Flight()
    print(fl.summary())
