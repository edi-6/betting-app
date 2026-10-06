"""The zoom's schedule: how wide the view is at every frame, and what's on screen when.

The view is L metres wide. It starts on Steve's eye (6.4 of his skin's pixels across, L0 = 0.4 m) and widens by
RATIO = 960,000,000 to the whole world sitting in the pupil of his eye, which is that same first frame again: the
video loops. The speed (in e-folds, factors of e, a second) changes with the scale: steady through the pixels and
blocks, slower for the village, fast across the empty thousands of kilometres, slow again for the Earth, the world
border and the reveal. It's a periodic spline, so the loop's seam has no bump in speed either.

Phases: '3d' (the spawn area round Steve, up to HANDOVER), a short cross-fade, 'map' (the generated world from above),
and from the moment the world fills the frame's height 'eye': the 3D close-up of the eye again, with the whole world
drawn in its pupil.
"""
import numpy as np
from scipy.interpolate import CubicSpline

FPS = 60
L0 = 0.4                          # 6.4 skin pixels of 1/16 block
PX = 1.0 / 16.0
WORLD = 6.0e7                     # the world border, 30,000,000 blocks each way
RATIO = WORLD / PX                # 960,000,000: the world is one of Steve's pixels
U = float(np.log(RATIO))
HANDOVER = 150.0                  # 3D up to here, then the map
FADE_U = 0.1                      # cross-fade (in e-folds) before the hand-over
EYE_L = WORLD * 9.0 / 16.0        # from here (the world fills the frame's height) it's the eye with the world in it
WALK = 4.317                      # blocks a second

# zoom speed (e-folds a second) at a scale (metres across)
SPEED = [(0.4, 0.5), (1.0, 0.52), (4.0, 0.56), (16.0, 0.58), (60.0, 0.52), (130.0, 0.5), (400.0, 0.62),
         (1.5e3, 0.85), (1.0e4, 1.15), (1.0e5, 1.4), (1.0e6, 1.25), (5.0e6, 0.8), (1.6e7, 0.5), (3.5e7, 0.44),
         (6.0e7, 0.4), (1.5e8, 0.45)]


class Timeline:
    def __init__(self, fps=FPS):
        self.fps = fps
        u = np.log(np.array([k[0] for k in SPEED]) / L0)
        v = np.array([k[1] for k in SPEED])
        uu = np.concatenate([u, [U]])
        vv = np.concatenate([v, [v[0]]])
        self.spline = CubicSpline(uu, vv, bc_type='periodic')
        grid = np.linspace(0.0, U, 200001)
        dt = np.diff(grid) / self.spline(0.5 * (grid[1:] + grid[:-1]))
        tt = np.concatenate([[0.0], np.cumsum(dt)])
        T = tt[-1]
        self.n = int(round(T * fps))
        self.T = self.n / fps
        self._t = tt * (self.T / T)           # stretched a hair so the loop is a whole number of frames
        self._u = grid
        self.k = T / self.T                   # speed factor that stretch implies

    def u_at(self, t):
        return float(np.interp(t % self.T, self._t, self._u))

    def L_at(self, t):
        return L0 * float(np.exp(self.u_at(t)))

    def speed_at(self, t):
        """e-folds a second at time t."""
        return float(self.spline(self.u_at(t))) * self.k

    def t_at_L(self, L):
        return float(np.interp(np.log(L / L0), self._u, self._t))

    def frame(self, i):
        t = i / self.fps
        L = self.L_at(t)
        return t, L


def phase(L):
    """('3d', w3d), ('map', 0) or ('eye', 0): which renderer draws a view L wide; w3d: weight of the 3D picture in
    the cross-fade."""
    if L >= EYE_L:
        return 'eye', 1.0
    if L >= HANDOVER:
        return 'map', 0.0
    u = np.log(HANDOVER / L)
    if u < FADE_U:
        w = u / FADE_U
        return 'fade', float(w * w * (3 - 2 * w))
    return '3d', 1.0


def counter(L):
    """The width of the view as the counter shows it: (number text, unit)."""
    if L < 1.0 - 1e-9:
        n = L / PX
        return (f'{n:.1f}' if n < 9.95 else f'{n:.0f}'), 'PIXELS'
    if L <= WORLD * 1.0000001:
        b = int(round(L))
        return f'{b:,}', 'BLOCK' if b == 1 else 'BLOCKS'
    n = L / WORLD
    txt = f'{n:.1f}' if n < 9.95 else f'{n:.0f}'
    return txt, 'PIXEL' if txt == '1.0' else 'PIXELS'


def walk_text(L):
    """How long it takes to walk across the view."""
    s = L / WALK
    if s < 60:
        return f'{s:.0f} SECONDS' if s >= 2 else f'{s:.1f} SECONDS'
    m = s / 60
    if m < 60:
        return f'{m:.0f} MINUTES'
    h = m / 60
    if h < 48:
        return f'{h:.0f} HOURS' if h >= 10 else f'{h:.1f} HOURS'
    d = h / 24
    if d < 365:
        return f'{d:.0f} DAYS'
    return f'{d / 365:.1f} YEARS'


def ramp(L, a, b, c, d):
    """1 between L = b and c, fading in from a and out to d (on a log scale)."""
    x = np.log(L)
    up = np.clip((x - np.log(a)) / max(np.log(b) - np.log(a), 1e-9), 0, 1)
    dn = np.clip((np.log(d) - x) / max(np.log(d) - np.log(c), 1e-9), 0, 1)
    v = min(up, dn)
    return float(v * v * (3 - 2 * v))


if __name__ == '__main__':
    tl = Timeline()
    print('loop: %d frames, %.2f s' % (tl.n, tl.T))
    for L in (0.4, 1, 4, 16, 60, 150, 400, 1e3, 1e4, 1e5, 1e6, 1e7, EYE_L, WORLD, 1.5e8, RATIO * L0):
        t = tl.t_at_L(L) if L < RATIO * L0 else tl.T
        print(f'L {L:12.4g}  t {t:6.2f}  speed {tl.speed_at(min(t, tl.T - 1e-6)):.2f}  {counter(L)}  walk {walk_text(L)}')
