"""Where every domino stands in part 2: the run through the plains and the field of almost 100,000 it was meant to
set off.

World axes: x east, y north, z up; one unit is one Minecraft block; the grass is at z = 0. A domino is W wide, H tall
and T thick (times its scale) and stands on its base with its big faces across its line (it falls forwards, along
its line).

The run starts at the player's feet in the south (a punch knocks the first one) and S-bends north into the growth:
ten dominoes, each 1.4 times the size of the one before, up to a giant 29 blocks tall. The giant's fall slams down
across the start of the race: four lines (red, gold, green and blue) racing north to one merge domino. Red gets
there first, but its line ends a block short; the first of the others to arrive carries on, over the river on the bridge, towards the field, where
it comes up short: its last domino falls a block before the feeder.

The field is a grid of NCOL x NROW dominoes, all falling north; the colour of each is on its south face, the one that
ends up facing the sky once it's down, so the picture only appears as the field falls. A feeder line in front of it
is cut into five segments; lightning strikes the middle of each, and each half of a segment runs outwards and sets
off the columns behind it.

The total is exactly 100,000: the field's columns and rows, and the start of the run, are chosen to make it so.
"""
import colorsys

import numpy as np

W, H, T = 0.5, 1.0, 0.15          # domino width, height, thickness (blocks), at scale 1
SPACING = 0.6                     # base to base along a line
PITCH = 0.6                       # field column pitch
FIELD_Y0 = 20.0                   # the field's first row (its south edge)
FEEDER_Y = FIELD_Y0 - 0.75
TOTAL = 100000
NSEG = 5                          # feeder segments (one lightning strike each)

RED = (204, 38, 36)
GOLD = (242, 184, 24)
GREEN = (62, 178, 52)
BLUE = (44, 86, 212)
RACE_COLS = [RED, GOLD, GREEN, BLUE]
RACE_NAMES = ['RED', 'GOLD', 'GREEN', 'BLUE']
FEED_COL = (232, 232, 226)
RIVER_Y = (-30.0, -22.0)          # the river's banks (south, north), running east-west
BRIDGE = (-33.0, -19.0)           # the bridge deck's ends

GROW = 1.4                        # each growth domino is this much bigger than the one before
NGROW = 10
GIANT_Y = -128.0                  # the giant's base centre (it stands on x = 0)
LANES = (-5.4, -1.8, 1.8, 4.6)    # the race lines' x at the start
MERGE_Y = -46.0
END_Y = FEEDER_Y - 2.05           # where the run's last domino stands: it lands a block short
CLIFF_Y = 258.0                   # the cliff's face, north of the field: the last shot is from its edge
CLIFF_Z = 72                      # its flat top
FINAL_EYE = (0.0, CLIFF_Y + 0.35, CLIFF_Z + 1.62)   # at the very edge
FINAL_PITCH = -38.0               # looking south, down over the field
FINAL_FOV = 64.0
RACE_WIGGLE = ((0.0, 0.0), (1.4, 2.0), (1.25, 5.0), (0.3, 1.5))     # (amplitude, periods) of each line's wiggle


# ---------------------------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------------------------
class Path:
    """A polyline sampled densely, addressed by arc length."""

    def __init__(self, pts):
        pts = np.asarray(pts, float)
        keep = np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-9]
        self.p = pts[keep]
        seg = np.linalg.norm(np.diff(self.p, axis=0), axis=1)
        self.s = np.r_[0.0, np.cumsum(seg)]
        self.length = float(self.s[-1])

    def point(self, s):
        s = np.clip(s, 0.0, self.length)
        return np.stack([np.interp(s, self.s, self.p[:, 0]), np.interp(s, self.s, self.p[:, 1])], -1)

    def tangent(self, s, ds=0.05):
        a = self.point(np.asarray(s) - ds)
        b = self.point(np.asarray(s) + ds)
        d = b - a
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def curvature_radius(self):
        """Smallest radius of curvature along the path (checks that the dominoes can follow it)."""
        s = np.arange(0.5, self.length - 0.5, 0.25)
        t0 = self.tangent(s - 0.25)
        t1 = self.tangent(s + 0.25)
        ang = np.arccos(np.clip((t0 * t1).sum(-1), -1, 1))
        return float(0.5 / max(ang.max(), 1e-9))


def hermite(p0, h0, p1, h1, k=None, n=200):
    """Cubic Hermite from p0 heading h0 (degrees, 90 = north) to p1 heading h1."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = np.linalg.norm(p1 - p0)
    k = L * 1.0 if k is None else k
    d0 = k * np.array([np.cos(np.radians(h0)), np.sin(np.radians(h0))])
    d1 = k * np.array([np.cos(np.radians(h1)), np.sin(np.radians(h1))])
    u = np.linspace(0, 1, n)[:, None]
    return (2 * u ** 3 - 3 * u ** 2 + 1) * p0 + (u ** 3 - 2 * u ** 2 + u) * d0 + (-2 * u ** 3 + 3 * u ** 2) * p1 \
        + (u ** 3 - u ** 2) * d1


def cat(*parts):
    out = [np.asarray(parts[0], float)]
    for p in parts[1:]:
        p = np.asarray(p, float)
        out.append(p[1:] if np.linalg.norm(p[0] - out[-1][-1]) < 1e-6 else p)
    return np.concatenate(out)


# ---------------------------------------------------------------------------------------------
# the growth: ten dominoes, each GROW times the one before
# ---------------------------------------------------------------------------------------------
def growth():
    """(scales, base y) of the growth dominoes, and the y of the run's last ordinary domino before them. Each stands
    so that the gap in front of the smaller one is 0.45 of its height, as in the run (it hits the next one at about
    0.64 of that one's height)."""
    s = GROW ** np.arange(1, NGROW + 1)
    sp = np.r_[1.0, s[:-1]]
    d = (0.075 + 0.45) * sp + 0.075 * s                 # centre to centre from the one before
    c = np.cumsum(d)
    y_last = GIANT_Y - c[-1]
    return s, y_last + c, y_last


GROW_S, GROW_Y, OPEN_END_Y = growth()
GIANT_S = float(GROW_S[-1])
RACE_Y0 = GIANT_Y + T * GIANT_S / 2 + H * GIANT_S + 0.9   # just past where the giant's top comes down


# ---------------------------------------------------------------------------------------------
# the run's paths
# ---------------------------------------------------------------------------------------------
OPEN_X0 = -6.0                    # the start is a little west of the growth line


def opening_path(length):
    """From the start, `length` blocks long, S-bending east onto the growth line and ending at its last ordinary
    domino (heading north)."""
    straight = 7.0
    y0 = OPEN_END_Y - length              # roughly: the S adds a little
    for _ in range(30):
        pts = cat([(OPEN_X0, y0), (OPEN_X0, y0 + 4.0)],
                  hermite((OPEN_X0, y0 + 4.0), 90, (0.0, OPEN_END_Y - straight), 90, k=16.0),
                  [(0.0, OPEN_END_Y - straight), (0.0, OPEN_END_Y)])
        p = Path(pts)
        y0 += p.length - length
    return Path(pts)


def race_path(j):
    """Lane j from the giant's landing to the merge domino. The lines leave the start straight, wiggle (each its
    own way: the lengths decide the race) and come in onto the merge domino: gold from the left, green straight
    on, blue from the right (as in part 1). Red goes straight for it, the shortest way, but its line
    ends beside the merge domino: it gets there first, and its last domino falls short."""
    x = LANES[j]
    amp, periods = RACE_WIGGLE[j]
    (ex, ey), h_in = (((-2.6, -1.9), 62.0), ((-0.75, -0.9), 55.0), ((0.0, -SPACING), 90.0),
                      ((0.75, -0.9), 125.0))[j]
    y_bend = MERGE_Y - 7.5
    u = np.linspace(0, 1, 900)
    y = RACE_Y0 + (y_bend - RACE_Y0) * u
    xw = x + amp * np.sin(u * 2 * np.pi * periods) * np.sin(np.pi * u) ** 2
    mid = np.stack([xw, y], -1)
    p3 = (ex, MERGE_Y + ey)
    return Path(cat(mid, hermite((x, y_bend), 90, p3, h_in, k=6.5)))


def tail_path():
    return Path(cat([(0.0, MERGE_Y), (0.0, BRIDGE[0] - 2.0)], [(0.0, BRIDGE[0] - 2.0), (0.0, BRIDGE[1] + 2.0)],
                    hermite((0.0, BRIDGE[1] + 2.0), 90, (-3.0, -4.0), 90, k=12.0, n=150),
                    hermite((-3.0, -4.0), 90, (0.0, END_Y - 6.0), 90, k=12.0, n=150),
                    [(0.0, END_Y - 6.0), (0.0, END_Y + 0.001)]))


# ---------------------------------------------------------------------------------------------
# the dominoes
# ---------------------------------------------------------------------------------------------
DTYPE = np.dtype([('x', 'f4'), ('y', 'f4'), ('yaw', 'f4'), ('line', 'i2'), ('k', 'i2'), ('s', 'f4'),
                  ('col', 'u1', 3), ('face', 'u1', 3)])
# line ids: 0 opening run + growth, 1..4 race lines (red, gold, green, blue), 5 tail, 20 + 2 * seg (+1): feeder half-segments running east (west), 1000 + c: field column c
LINE_MAIN, LINE_TAIL = 0, 5
FEED0, FIELD = 20, 1000


def rainbow(s, period=36.0, sat=0.72):
    """The run's colours: the hue turns as the line goes, one full turn every `period` blocks."""
    h = (s / period) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, sat, 0.96)
    return (int(r * 255), int(g * 255), int(b * 255))


def _line(path, line, s0=0.0, s1=None, col=None, hue0=0.0):
    s1 = path.length if s1 is None else s1
    s = np.arange(s0, s1 + 1e-6, SPACING)
    p = path.point(s)
    t = path.tangent(s)
    out = np.zeros(len(s), DTYPE)
    out['x'], out['y'] = p[:, 0], p[:, 1]
    out['yaw'] = np.arctan2(t[:, 1], t[:, 0])
    out['line'] = line
    out['k'] = np.arange(len(s))
    out['s'] = 1.0
    for i in range(len(s)):
        c = col if col is not None else rainbow(s[i] + hue0)
        out['col'][i] = c
        out['face'][i] = c
    return out


class Layout:
    def __init__(self, picture=None, n_open=60):
        self.race = [race_path(j) for j in range(4)]
        self.tail = tail_path()
        race = []
        for j, path in enumerate(self.race):
            d = _line(path, 1 + j, col=RACE_COLS[j])
            race.append(d)
        race = np.concatenate(race)
        tail = _line(self.tail, LINE_TAIL, SPACING, hue0=150.0)
        n_grow = NGROW
        fixed = len(race) + len(tail) + n_grow + n_open
        # the field: an odd number of columns (one on the middle line) and as many rows as fit, with the rest
        # made up at the start of the run; pick the width (near 257) that leaves the fewest to make up
        best = None
        for ncol in range(249, 267, 2):
            nrow = (TOTAL - fixed - ncol) // ncol
            extra = TOTAL - fixed - ncol - ncol * nrow
            if best is None or extra < best[2]:
                best = (ncol, nrow, extra)
        self.ncol, self.nrow, extra = best
        self.n_open = n_open + extra
        self.opening = opening_path((self.n_open - 1) * SPACING)
        run0 = _line(self.opening, LINE_MAIN, 0.0, self.opening.length + 1e-3)
        run0 = run0[len(run0) - self.n_open:]
        run0['k'] = np.arange(len(run0))
        grow = np.zeros(n_grow, DTYPE)
        grow['x'] = 0.0
        grow['y'] = GROW_Y
        grow['yaw'] = np.pi / 2
        grow['line'] = LINE_MAIN
        grow['k'] = len(run0) + np.arange(n_grow)
        grow['s'] = GROW_S
        for i in range(n_grow):
            c = rainbow(i * 3.6, period=36.0, sat=0.82)
            grow['col'][i] = grow['face'][i] = c
        self.run = np.concatenate([run0, grow, race, tail])
        self.feeder = self._feeder()
        self.field = self._field(picture)
        self.all = np.concatenate([self.run, self.feeder, self.field])
        assert len(self.all) == TOTAL, len(self.all)

    # -- the field and its feeder ------------------------------------------------------------------
    def col_x(self, c):
        return (np.asarray(c) - (self.ncol - 1) / 2.0) * PITCH

    def strikes(self):
        """x of the lightning strikes: the middle of each feeder segment (the first one is the middle)."""
        w = self.ncol * PITCH / NSEG
        return (np.arange(NSEG) - (NSEG - 1) / 2.0) * w

    def _feeder(self):
        """One feeder domino in front of each column. Segment k runs from its strike point out both ways: the half
        east of it falls east, the half west of it falls west."""
        xs = self.col_x(np.arange(self.ncol))
        sx = self.strikes()
        seg = np.argmin(np.abs(xs[:, None] - sx[None, :]), axis=1)
        out = np.zeros(self.ncol, DTYPE)
        out['x'] = xs
        out['y'] = FEEDER_Y
        out['s'] = 1.0
        east = xs >= sx[seg] - 1e-6
        out['yaw'] = np.where(east, 0.0, np.pi)
        out['line'] = FEED0 + 2 * seg + (~east)
        for line in np.unique(out['line']):
            ii = np.nonzero(out['line'] == line)[0]
            order = np.argsort(np.abs(xs[ii] - sx[(line - FEED0) // 2]))
            out['k'][ii[order]] = np.arange(len(ii))
        out['col'] = FEED_COL
        out['face'] = FEED_COL
        return out

    def _field(self, picture):
        out = np.zeros(self.ncol * self.nrow, DTYPE)
        c, r = np.meshgrid(np.arange(self.ncol), np.arange(self.nrow), indexing='ij')
        out['x'] = self.col_x(c).ravel()
        out['y'] = (FIELD_Y0 + r * SPACING).ravel()
        out['yaw'] = np.pi / 2
        out['line'] = (FIELD + c).ravel()
        out['k'] = r.ravel()
        out['s'] = 1.0
        out['col'] = FEED_COL
        out['face'] = (200, 200, 200) if picture is None else picture(out['x'], out['y'])
        return out

    @property
    def field_y1(self):
        return FIELD_Y0 + (self.nrow - 1) * SPACING

    def counts(self):
        run = self.run
        return {'opening': self.n_open, 'growth': NGROW,
                'race': [int((run['line'] == 1 + j).sum()) for j in range(4)],
                'tail': int((run['line'] == LINE_TAIL).sum()), 'feeder': len(self.feeder),
                'field': len(self.field), 'cols': self.ncol, 'rows': self.nrow, 'total': len(self.all)}


if __name__ == '__main__':
    L = Layout()
    print(L.counts())
    print('growth scales', np.round(GROW_S, 2), 'y', np.round(GROW_Y, 1), 'opening ends', round(OPEN_END_Y, 1),
          'race starts', round(RACE_Y0, 1))
    print('opening from y %.1f, length %.1f; race lengths %s; tail %.1f; field y %.1f..%.1f, x +-%.1f' % (
        L.opening.p[0, 1], L.opening.length, [round(p.length, 2) for p in L.race], L.tail.length, FIELD_Y0,
        L.field_y1, L.col_x(L.ncol - 1)))
    print('min curvature radius: opening %.2f, race %s, tail %.2f' % (
        L.opening.curvature_radius(), [round(p.curvature_radius(), 2) for p in L.race], L.tail.curvature_radius()))
    # the race lines must keep clear of each other
    from scipy.spatial import cKDTree
    for a in range(4):
        for b in range(a + 1, 4):
            pa = L.race[a].point(np.arange(0, L.race[a].length - 1.0, 0.3))
            pb = L.race[b].point(np.arange(0, L.race[b].length - 1.0, 0.3))
            print('lanes', a, b, 'closest %.2f' % cKDTree(pb).query(pa)[0].min())
