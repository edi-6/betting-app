"""The edit, shot by shot: which round, which moment of it (slowed down or sped up), where the camera is, what the
villagers do. Screen time runs from 0 at the first frame; every shot maps its own 0..1 onto the round's clock.

  cold open   the 1,000-block wall over the village (a flash-forward)
  round 1     one block: a wave washes over the fisherman's feet. He looks down. Hmm.
  round 2     ten blocks: it rolls in over the beach, slams into the first houses and floods the village
  round 3     a hundred: the horizon rises, the bell rings, the wall comes down on the village, nothing is left
  round 4     a thousand: the wall blots out the sun... and you put down a sponge
"""
import numpy as np

import wall as WL

FPS = 60


def ease(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _interp_keys(keys, u):
    """Piecewise interpolation of (u, value) keys with eased segments."""
    us = [k[0] for k in keys]
    if u <= us[0]:
        return np.asarray(keys[0][1], float)
    if u >= us[-1]:
        return np.asarray(keys[-1][1], float)
    j = int(np.searchsorted(us, u, side='right')) - 1
    a, b = keys[j], keys[j + 1]
    w = ease((u - a[0]) / (b[0] - a[0]))
    return np.asarray(a[1], float) * (1 - w) + np.asarray(b[1], float) * w


def _clock(keys, u):
    """Monotone, smooth mapping from the shot's 0..1 to the round's clock: (u, t) keys, linear between them
    but with the speed changes rounded off (Catmull-Rom on the slopes, clamped)."""
    us = np.array([k[0] for k in keys], float)
    ts = np.array([k[1] for k in keys], float)
    if len(us) == 2:
        return float(ts[0] + (ts[1] - ts[0]) * (u - us[0]) / (us[1] - us[0]))
    # monotone cubic (Fritsch-Carlson)
    d = np.diff(ts) / np.diff(us)
    m = np.zeros_like(ts)
    m[0], m[-1] = d[0], d[-1]
    for i in range(1, len(ts) - 1):
        m[i] = 0.0 if d[i - 1] * d[i] <= 0 else 2.0 / (1.0 / d[i - 1] + 1.0 / d[i])
    u = float(np.clip(u, us[0], us[-1]))
    j = min(int(np.searchsorted(us, u, side='right')) - 1, len(us) - 2)
    h = us[j + 1] - us[j]
    s = (u - us[j]) / h
    h00, h10, h01, h11 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s, -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
    return float(h00 * ts[j] + h10 * h * m[j] + h01 * ts[j + 1] + h11 * h * m[j + 1])


class Shot:
    def __init__(self, name, dur, rnd, clock, cam, nc=None, shake=None, people=None, hide=(), extra=None,
                 spray=None, fog=None, far=1500.0, near=0.3):
        self.name, self.dur, self.rnd = name, float(dur), rnd
        self.clock_keys = clock          # [(u, round time)]
        self.cam_keys = cam              # [(u, eye, target, fov)]
        self.nc = nc
        self.shake = shake               # [(u0, u1, amplitude in degrees)]
        self.people = people             # f(u, t) -> {index: pose overrides}
        self.hide = hide
        self.extra = extra or {}
        self.spray = spray or {}
        self.fog = fog
        self.far, self.near = far, near
        self.start = 0.0

    def clock(self, u):
        return _clock(self.clock_keys, u)

    def camera(self, u):
        eye = _interp_keys([(k[0], k[1]) for k in self.cam_keys], u)
        tgt = _interp_keys([(k[0], k[2]) for k in self.cam_keys], u)
        fov = float(_interp_keys([(k[0], k[3]) for k in self.cam_keys], u))
        cam = {'eye': tuple(eye), 'target': tuple(tgt), 'fov': fov, 'far': self.far, 'near': self.near}
        if self.shake:
            amp = 0.0
            for (u0, u1, a) in self.shake:
                if u0 <= u <= u1:
                    amp = max(amp, a * np.sin(np.pi * (u - u0) / (u1 - u0)) ** 0.5)
            if amp > 0:
                t = (self.start + u * self.dur) * 23.0
                d = np.linalg.norm(tgt - eye)
                off = np.array([np.sin(t * 1.3) + 0.5 * np.sin(t * 3.1), np.cos(t * 1.7) + 0.5 * np.sin(t * 2.3),
                                0.6 * np.sin(t * 2.9)])
                cam['target'] = tuple(tgt + off * np.tan(np.radians(amp)) * d)
        return cam


# -- who does what ----------------------------------------------------------------------------------------------------
def _r1_people(u, t):
    """The fisherman, his back to the sea, until it's round his feet: he looks down, then at us."""
    look_down = ease((t - 16.05) / 0.35) * (1.0 - ease((t - 16.85) / 0.35))
    turn = ease((t - 16.85) / 0.3)
    return {0: {'pos': (-3.5, -3.6, 2.0), 'yaw': 0.0, 'state': 0,
                'head_pitch': -0.55 * look_down + 0.06 * turn,
                'head_yaw': 0.25 * np.sin(t * 0.8) * (1 - ease((t - 15.6) / 0.3)) + 0.0 * turn}}


def _r4_look_up(u, t):
    """Everybody in the street staring up at the wall."""
    out = {}
    for i in range(9):
        out[i] = {'state': 0, 'head_pitch': 0.55 if i != 8 else 0.5, 'head_yaw': 0.1 * np.sin(t * 0.6 + i),
                  'yaw': np.pi + 0.15 * np.sin(i * 2.1)}
    out[8]['arms'] = 0.0
    return out


def _r4_end_people(u, t):
    """The fisherman walks up to the wet sponge, has a look at it, then at us."""
    walk = np.clip(u / 0.40, 0.0, 1.0)
    walk = walk * (2.0 - walk)                          # slowing to a stop
    x = -8.8 + 4.1 * walk
    stop = u > 0.40
    turn = ease((u - 0.36) / 0.1)
    yaw = np.pi / 2 * (1 - turn) + 2.20 * turn
    hp = -0.55 * ease((u - 0.45) / 0.1)
    hy = 0.0
    if u > 0.78:
        k = ease((u - 0.78) / 0.08)
        hy = -0.78 * k                                # and then round at us
        hp = -0.55 * (1 - k) + 0.05 * k
    pose = {'pos': (x, -0.4, 1.0), 'yaw': yaw, 'state': 0, 'head_pitch': hp, 'head_yaw': hy}
    if not stop:
        pose['bob'] = abs(np.sin(u * 30.0)) * 0.05
    return {0: pose}


CROWD = {0: (-3.4, -27.2), 2: (-1.1, -27.8), 3: (1.3, -27.0), 4: (3.5, -27.6), 8: (0.2, -24.6)}


def _r4_crowd(u, t):
    """The villagers and the golem lined up in the street, staring up at it."""
    out = {}
    for i, (x, y) in CROWD.items():
        out[i] = {'pos': (x, y, 2.0), 'yaw': np.pi + 0.12 * np.sin(i * 2.3), 'state': 0,
                  'head_pitch': (0.62 if i != 8 else 0.5) + 0.03 * np.sin(t * 0.9 + i),
                  'head_yaw': 0.12 * np.sin(t * 0.5 + i * 1.3)}
    return out


# -- the shots ----------------------------------------------------------------------------------------------------------
SPONGE = (-3.5, 0.5, 1.5)        # where the sponge goes: on the sand, just above the water line
FP_EYE = (-3.5, -3.4, 3.62)      # standing on the grass at the top of the beach, looking out to sea
FP_TGT = (-3.5, 30.0, 6.8)

SHOTS = [
    # cold open: the wall over the village
    Shot('open', 1.5, 4, [(0, WL.t_at(1050)), (1, WL.t_at(1000))],
         [(0, (-9.0, -44.0, 3.4), (-2.0, 40.0, 32.0), 72), (1, (-9.0, -42.4, 3.6), (-2.0, 40.0, 33.5), 70)],
         nc=(-4.0, -25.0, 4.0), people=_r4_look_up, far=15000.0, near=0.5, fog=0.0007),
    # round 1
    Shot('r1', 3.8, 1, [(0, 13.3), (1, 17.1)],
         [(0, (-6.0, -14.5, 5.6), (-3.3, 6.0, 1.9), 40), (1, (-5.7, -13.3, 5.4), (-3.3, 6.0, 1.9), 38)],
         nc=(-3.5, 0.0, 2.0), people=_r1_people),
    # round 2
    Shot('r2_coming', 1.7, 2, [(0, 11.2), (1, 12.95)],
         [(0, (-16.0, -1.2, 3.2), (-6.0, 60.0, 9.0), 64), (1, (-16.0, -2.4, 3.4), (-6.0, 60.0, 10.0), 64)],
         nc=(-10.0, 15.0, 3.0)),
    Shot('r2_hit', 2.3, 2, [(0, 13.25), (1, 15.3)],
         [(0, (-25.0, -42.0, 17.0), (-3.0, -2.0, 3.0), 60), (1, (-23.0, -39.0, 17.5), (-3.0, -4.0, 3.0), 60)],
         nc=(-10.0, -18.0, 5.0), shake=[(0.05, 0.5, 0.35)], spray={'alpha': 0.5}),
    Shot('r2_flood', 2.2, 2, [(0, 16.4), (1, 18.95)],
         [(0, (45.0, -95.0, 30.0), (-5.0, -30.0, 4.0), 58), (1, (38.0, -100.0, 31.0), (-8.0, -32.0, 4.0), 58)],
         nc=(-5.0, -35.0, 5.0), spray={'alpha': 0.45}),
    # round 3
    Shot('r3_horizon', 1.7, 3, [(0, 7.6), (1, 9.9)],
         [(0, (-2.0, -96.0, 30.0), (0.0, 60.0, 40.0), 62), (1, (-2.0, -93.0, 30.5), (0.0, 60.0, 44.0), 62)],
         nc=(0.0, -60.0, 5.0), spray={'alpha': 0.35, 'size': 0.7}),
    Shot('r3_look_up', 1.6, 3, [(0, 10.0), (1, 10.78)],
         [(0, (-6.0, -31.0, 3.4), (0.0, 40.0, 28.0), 72), (1, (-6.0, -32.5, 3.2), (0.0, 40.0, 34.0), 74)],
         nc=(-4.0, -15.0, 4.0), shake=[(0.2, 1.0, 0.5)], spray={'alpha': 0.35, 'size': 0.7}),
    Shot('r3_impact', 2.8, 3, [(0, 10.5), (0.55, 10.95), (1, 11.26)],
         [(0, (62.0, -32.0, 20.0), (0.0, -20.0, 8.0), 60), (1, (58.0, -34.0, 23.0), (-2.0, -20.0, 10.0), 62)],
         nc=(25.0, -22.0, 6.0), shake=[(0.35, 1.0, 0.5)], spray={'alpha': 0.2, 'size': 0.5}),
    Shot('r3_gone', 2.1, 3, [(0, 12.6), (1, 14.4)],
         [(0, (0.0, 140.0, 230.0), (0.0, -160.0, 40.0), 60), (1, (0.0, 118.0, 224.0), (0.0, -172.0, 40.0), 60)],
         nc=(0.0, -60.0, 5.0), spray={'alpha': 0.15, 'size': 0.5}, far=3000.0),
    # round 4
    Shot('r4_wide', 2.0, 4, [(0, WL.t_at(2700)), (1, WL.t_at(2350))],
         [(0, (60.0, -330.0, 75.0), (0.0, 400.0, 190.0), 62), (1, (60.0, -312.0, 78.0), (0.0, 400.0, 200.0), 62)],
         nc=(0.0, -100.0, 5.0), far=15000.0, near=0.5, fog=0.0006),
    Shot('r4_street', 1.8, 4, [(0, WL.t_at(1600)), (1, WL.t_at(1450))],
         [(0, (0.5, -36.2, 2.2), (0.3, 40.0, 34.0), 78), (1, (0.5, -35.4, 2.25), (0.3, 40.0, 39.0), 78)],
         nc=(0.0, -26.0, 3.0), people=_r4_crowd, hide=(1, 5, 6, 7), far=15000.0, near=0.3, fog=0.0007),
    Shot('r4_sponge', 4.6, 4, [(0, WL.t_at(1330)), (0.25, WL.t_at(1290)), (1, WL.t_at(1270))],
         [(0, FP_EYE, FP_TGT, 74), (1, FP_EYE, FP_TGT, 74)],
         nc=(-3.5, 4.0, 2.0), hide=(0,), far=15000.0, near=0.2, fog=0.0005,
         extra={'arm': True, 'swing': 0.24, 'place': 0.27, 'drink': (0.37, 0.80), 'wet': 0.5}),
    Shot('r4_hmm', 2.8, 4, [(0, WL.t_at(1270)), (1, WL.t_at(1270) + 2.8)],
         [(0, (3.6, -1.55, 1.95), (-3.8, 0.3, 1.75), 40), (1, (3.3, -1.5, 1.92), (-3.8, 0.3, 1.75), 38)],
         nc=(-3.5, 0.0, 2.0), people=_r4_end_people, far=15000.0, near=0.2, fog=0.0005,
         extra={'drink_done': True, 'sponge_wet': True}),
    Shot('end', 1.0, 4, [(0, WL.t_at(1270) + 2.8), (1, WL.t_at(1270) + 3.8)],
         [(0, (3.3, -1.5, 1.92), (-3.8, 0.3, 1.75), 38), (1, (3.25, -1.5, 1.92), (-3.8, 0.3, 1.75), 38)],
         nc=(-3.5, 0.0, 2.0), people=lambda u, t: _r4_end_people(1.0, t), far=15000.0, near=0.2, fog=0.0005,
         extra={'drink_done': True, 'sponge_wet': True}),
]

t = 0.0
for _s in SHOTS:
    _s.start = t
    t += _s.dur
DURATION = t
N_FRAMES = int(round(DURATION * FPS))


def shot_at(ts):
    """The shot on screen at screen time ts, and how far through it is (0..1)."""
    for s in SHOTS:
        if ts < s.start + s.dur or s is SHOTS[-1]:
            return s, float(np.clip((ts - s.start) / s.dur, 0.0, 1.0))
    return SHOTS[-1], 1.0


def round_start(n):
    for s in SHOTS:
        if s.rnd == n and s.name != 'open':
            return s.start
    return None
