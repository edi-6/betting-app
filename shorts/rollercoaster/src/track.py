"""The coaster's track: paths designed like a ride, with the physics that sets the speed along them.

A path is built by steering a "turtle": each segment has a length, a total turn and an end pitch, eased over its
length, so drops and turns come out smooth. While it is built, the cart's speed is integrated along it (gravity,
rolling friction, air drag, powered-rail boosters and the lift chain), so a ballistic segment (the jump over the gap)
can use the real take-off speed and land where the physics puts it.

A Track is a path resampled by arc length s, with rotation-minimising frames, banking into the curves from the lateral
acceleration, the speed v(s) and the ride time t(s).

World units are blocks, z up. Yaw 0 = +y (north), positive = towards +x (east). Pitch positive = up.
"""
import numpy as np

G = 14.0            # gravity for the ride (blocks/s^2): the game's is harsher, this keeps the big drops readable
ROLL = 0.0025       # rolling friction (fraction of g)
DRAG = 0.0016       # air drag: d(v^2)/ds = -DRAG * v^2
STEP = 0.1


def direction(yaw_deg, pitch_deg):
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    return np.array([np.sin(y) * np.cos(p), np.cos(y) * np.cos(p), np.sin(p)])


def smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


class PathBuilder:
    """Steer a path and integrate the cart's speed along it."""

    def __init__(self, pos, yaw=0.0, pitch=0.0, v=0.0):
        self.p = np.array(pos, float)
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.v = float(v)
        self.pts = [self.p.copy()]
        self.vs = [self.v]
        self.marks = {}               # name -> index of the point where it was set
        self.powered = []             # (i0, i1, target speed, accel) ranges of powered rail
        self.lift = []                # (i0, i1, speed) ranges pulled by the lift chain
        self.gaps = []                # (i0, i1) ranges flown ballistically (no rails)
        self.boost = None             # (target, accel) while a powered section is being built
        self.chain = None

    @property
    def i(self):
        return len(self.pts) - 1

    def mark(self, name):
        self.marks[name] = self.i
        return self

    def _advance(self, d):
        """Move STEP along direction d and update the speed."""
        dz = d[2] * STEP
        v2 = self.v * self.v - 2 * G * dz - 2 * ROLL * G * STEP - DRAG * self.v * self.v * STEP
        if self.boost is not None:
            target, acc = self.boost
            if v2 < target * target:
                v2 = min(target * target, v2 + 2 * acc * STEP)
        if self.chain is not None:
            v2 = self.chain * self.chain
        self.v = float(np.sqrt(max(v2, 0.25)))
        self.p = self.p + d * STEP
        self.pts.append(self.p.copy())
        self.vs.append(self.v)

    def seg(self, length, turn=0.0, pitch=None, ease_pitch='smooth', ease_turn='smooth'):
        """length blocks; turn: total yaw change (degrees); pitch: end pitch (degrees, None = keep)."""
        n = max(int(round(length / STEP)), 1)
        y0, p0 = self.yaw, self.pitch
        p1 = p0 if pitch is None else float(pitch)
        e_p = smooth if ease_pitch == 'smooth' else (lambda u: u)
        # turn with a smooth rate profile: yaw(u) = y0 + turn * e(u)
        e_t = smooth if ease_turn == 'smooth' else (lambda u: u)
        for k in range(1, n + 1):
            u = k / n
            self.yaw = y0 + turn * e_t(u)
            self.pitch = p0 + (p1 - p0) * e_p(u)
            self._advance(direction(self.yaw, self.pitch))
        return self

    def powered_on(self, target, accel):
        self.boost = (target, accel)
        self._pw0 = self.i
        return self

    def powered_off(self):
        self.powered.append((self._pw0, self.i, self.boost[0], self.boost[1]))
        self.boost = None
        return self

    def chain_on(self, speed):
        self.chain = speed
        self._ch0 = self.i
        self.v = speed
        return self

    def chain_off(self):
        self.lift.append((self._ch0, self.i, self.chain))
        self.chain = None
        return self

    def ballistic(self, until_z=None, duration=None, name='gap'):
        """Fly off the end of the rails: a parabola from the current point with the current speed and direction,
        under G, until it comes back down to until_z (or for `duration` seconds). Returns self (the heading and
        pitch are those of the landing)."""
        d = direction(self.yaw, self.pitch)
        vel = d * self.v
        i0 = self.i
        p = self.p.copy()
        t = 0.0
        dt = 0.002
        while True:
            vel = vel + np.array([0, 0, -G]) * dt
            p = p + vel * dt
            t += dt
            if np.linalg.norm(p - self.pts[-1]) >= STEP:
                self.pts.append(p.copy())
                self.vs.append(float(np.linalg.norm(vel)))
            if duration is not None and t >= duration:
                break
            if until_z is not None and vel[2] < 0 and p[2] <= until_z:
                break
        self.p = self.pts[-1].copy()
        self.v = self.vs[-1]
        hv = np.hypot(vel[0], vel[1])
        self.yaw = float(np.degrees(np.arctan2(vel[0], vel[1])))
        self.pitch = float(np.degrees(np.arctan2(vel[2], hv)))
        self.gaps.append((i0, self.i))
        self.marks[name] = i0
        self.marks[name + '_end'] = self.i
        return self

    def build(self, bank_max=62.0, bank_smooth=6.0, name='track'):
        return Track(np.array(self.pts), np.array(self.vs), self, bank_max=bank_max, bank_smooth=bank_smooth,
                     name=name)


def rmf(P, T):
    """Rotation-minimising frames along a polyline (double reflection, Wang et al. 2008): returns the up vectors,
    starting from the world up made perpendicular to the first tangent."""
    n = len(P)
    U = np.zeros_like(P)
    up = np.array([0.0, 0.0, 1.0])
    u0 = up - np.dot(up, T[0]) * T[0]
    if np.linalg.norm(u0) < 1e-6:
        u0 = np.array([0.0, 1.0, 0.0]) - np.dot([0, 1.0, 0], T[0]) * T[0]
    U[0] = u0 / np.linalg.norm(u0)
    for i in range(n - 1):
        v1 = P[i + 1] - P[i]
        c1 = np.dot(v1, v1)
        if c1 < 1e-12:
            U[i + 1] = U[i]
            continue
        rL = U[i] - (2 / c1) * np.dot(v1, U[i]) * v1
        tL = T[i] - (2 / c1) * np.dot(v1, T[i]) * v1
        v2 = T[i + 1] - tL
        c2 = np.dot(v2, v2)
        U[i + 1] = rL - (2 / c2) * np.dot(v2, rL) * v2 if c2 > 1e-12 else rL
        U[i + 1] /= np.linalg.norm(U[i + 1])
    return U


class Track:
    """A built path resampled every STEP blocks of arc length."""

    def __init__(self, pts, vs, builder=None, bank_max=62.0, bank_smooth=6.0, name='track'):
        self.name = name
        d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(d)])
        self.length = float(s[-1])
        n = int(self.length / STEP) + 1
        self.s = np.linspace(0.0, self.length, n)
        self.P = np.stack([np.interp(self.s, s, pts[:, k]) for k in range(3)], 1)
        self.v = np.interp(self.s, s, vs)
        self.src_s = s
        T = np.gradient(self.P, axis=0)
        T /= np.linalg.norm(T, axis=1, keepdims=True)
        self.T = T
        U0 = rmf(self.P, T)
        # banking: lean into the curve so the lateral acceleration is (mostly) felt as down. The curvature is taken
        # in the plane perpendicular to world up (turns), measured against the frame's right vector.
        R0 = np.cross(T, U0)
        dT = np.gradient(T, axis=0) / STEP                           # curvature vector
        lat = np.einsum('ij,ij->i', dT, R0)                          # + = turning right
        want = np.degrees(np.arctan2(self.v ** 2 * lat, G))
        want = np.clip(want, -bank_max, bank_max)
        sig = bank_smooth / STEP
        k = np.arange(-int(3 * sig), int(3 * sig) + 1)
        w = np.exp(-0.5 * (k / sig) ** 2)
        w /= w.sum()
        bank = np.convolve(np.pad(want, len(k) // 2, mode='edge'), w, mode='valid')
        self.gap_mask = np.zeros(n, bool)
        self.gaps = []
        self.powered = []
        self.lift = []
        self.marks = {}
        if builder is not None:
            for (i0, i1) in builder.gaps:
                a, b = s[i0], s[i1]
                self.gaps.append((a, b))
                self.gap_mask |= (self.s >= a) & (self.s <= b)
            self.powered = [(s[i0], s[i1], tv, acc) for (i0, i1, tv, acc) in builder.powered]
            self.lift = [(s[i0], s[i1], sp) for (i0, i1, sp) in builder.lift]
            self.marks = {k: float(s[i]) for k, i in builder.marks.items()}
        bank = np.where(self.gap_mask, 0.0, bank)
        self.bank = bank
        br = np.radians(bank)
        c, sn = np.cos(br)[:, None], np.sin(br)[:, None]
        # roll U0/R0 about T by the bank angle: positive bank (a right turn) tips the up vector towards the right, into
        # the turn, so the rider's "down" follows gravity plus the centripetal pull
        self.U = U0 * c + R0 * sn
        self.R = np.cross(T, self.U)
        dt = STEP / np.maximum(self.v, 0.5)
        self.t = np.concatenate([[0.0], np.cumsum(0.5 * (dt[1:] + dt[:-1]))])
        self.duration = float(self.t[-1])

    # -- sampling ------------------------------------------------------------------------------
    def _idx(self, s):
        f = np.clip(s / STEP, 0, len(self.s) - 1.000001)
        i = int(f)
        return i, f - i

    def _lerp(self, A, s):
        i, u = self._idx(s)
        return A[i] * (1 - u) + A[i + 1] * u

    def pos(self, s):
        return self._lerp(self.P, s)

    def frame(self, s):
        """(T, R, U): forward, right, up of the rails at s (banked)."""
        T = self._lerp(self.T, s)
        U = self._lerp(self.U, s)
        T /= np.linalg.norm(T)
        U = U - np.dot(U, T) * T
        U /= np.linalg.norm(U)
        return T, np.cross(T, U), U

    def speed(self, s):
        return float(self._lerp(self.v, s))

    def time(self, s):
        return float(np.interp(s, self.s, self.t))

    def s_at(self, t):
        return float(np.interp(t, self.t, self.s))

    def in_gap(self, s):
        return any(a <= s <= b for (a, b) in self.gaps)

    def is_powered(self, s):
        return any(a <= s <= b for (a, b, _, _) in self.powered)

    def summary(self):
        out = [f'{self.name}: {self.length:.0f} blocks, {self.duration:.2f} s, v {self.v.min():.1f}..{self.v.max():.1f}'
               f' blocks/s, bank {self.bank.min():.0f}..{self.bank.max():.0f} deg, z {self.P[:, 2].min():.0f}..'
               f'{self.P[:, 2].max():.0f}']
        for k, sv in sorted(self.marks.items(), key=lambda kv: kv[1]):
            out.append(f'  {k:18s} s {sv:7.1f}  t {self.time(sv):6.2f}  v {self.speed(sv):5.1f}  z {self.pos(sv)[2]:6.1f}')
        return '\n'.join(out)
