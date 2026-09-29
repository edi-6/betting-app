"""Where every domino stands: the run through the plains and the picture field it was meant to set off.

World axes: x east, y north, z up; one unit is one Minecraft block; the grass is at z = 0. A domino is W wide, H tall
and T thick, and stands on its base with its big faces across its line (it falls forward, along its line).

The run starts at a player's feet in the south (a punch knocks the first one), comes north into a double spiral
(in clockwise, an S through the middle, out anticlockwise), splits three ways into a red, a gold and a blue line
that race each other and join again (the first to arrive carries on), crosses the river on a bridge and heads for
the field, where it comes up short: its last domino falls a block before the one it was meant to hit.

The field is a grid of NCOL x NROW dominoes, all falling north. The colour of each one is on its south face, the one
that ends up facing the sky once it's down, so the picture only appears as the field falls. A feeder line in front
of the field (two lines running east and west from the middle) sets off every column in turn.

The total is exactly 10,000: the field's height in rows is chosen to make it so.
"""
import numpy as np

W, H, T = 0.5, 1.0, 0.15          # domino width, height, thickness (blocks)
SPACING = 0.6                     # base to base along a line
NCOL = 72                         # field columns (picture width in pixels)
PITCH = 0.6                       # field column pitch
FIELD_Y0 = 20.0                   # the field's first row (its south edge)
FEEDER_Y = FIELD_Y0 - 0.75
TOTAL = 10000

RED = (196, 36, 36)
GOLD = (236, 176, 24)
BLUE = (44, 82, 206)
RUN_COLS = [(232, 232, 226), (206, 206, 200)]          # the run: white dominoes, a little variation
RIVER_Y = (-30.0, -22.0)          # the river's banks (south, north), running east-west
BRIDGE = (-33.0, -19.0)           # the bridge deck's ends


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


def double_spiral(c, b, th_max, th_c, n=1200):
    """In along the Archimedean spiral r = b th (clockwise, th from th_max down to th_c), an S of two
    semicircles through the centre, out along the other arm (anticlockwise). th_max an odd multiple of pi: in on the
    west side and out on the east side, both heading (nearly) north."""
    e = lambda t: np.stack([np.cos(t), np.sin(t)], -1)
    th = np.linspace(th_max, th_c, n)
    arm1 = (b * th)[:, None] * e(th)
    rc = b * th_c
    ec = e(np.array(th_c))
    phi0 = np.arctan2(ec[1], ec[0])
    s1 = (rc / 2) * ec + (rc / 2) * e(np.linspace(phi0, phi0 - np.pi, 200))
    s2 = -(rc / 2) * ec + (rc / 2) * e(np.linspace(phi0, phi0 + np.pi, 200))
    arm2 = -arm1[::-1]
    return np.asarray(c, float) + np.concatenate([arm1, s1[1:], s2[1:], arm2[1:]])


def cat(*parts):
    out = [np.asarray(parts[0], float)]
    for p in parts[1:]:
        p = np.asarray(p, float)
        out.append(p[1:] if np.linalg.norm(p[0] - out[-1][-1]) < 1e-6 else p)
    return np.concatenate(out)


SPIRAL_C = (0.0, -90.0)
SPIRAL_B = 0.35                   # arms 1.1 apart
SPIRAL_TH = 5 * np.pi             # 1.2 turns in, 1.2 out
SPIRAL_THC = 2.55 * np.pi         # where the S starts (its semicircles have a radius of 1.4)
R_SPIRAL = SPIRAL_B * SPIRAL_TH
START = (-R_SPIRAL, -112.0)
SPLIT = (0.0, -73.0)
MERGE = (0.0, -41.0)
END_Y = 17.2                      # where the run's last domino stands
# the race: (side, how far out, wiggle amplitude, wiggle periods); the lines' lengths differ by a block or two
BRANCH_SHAPES = ((-1, 3.2, 2.0, 1.5), (0, 0.0, 1.5, 2.5), (1, 3.2, 0.0, 0.0))


def branch_path(side, out, amp, periods):
    """From just past the split to just before the merge point: the side lines leave at 35 degrees, go out to
    `out` blocks off the middle, and come back in at 35 degrees onto the merge domino."""
    y0, y1 = SPLIT[1], MERGE[1]
    if side == 0:
        u = np.linspace(0, 1, 800)
        y = y0 + SPACING + (y1 - y0 - 2 * SPACING) * u
        x = amp * np.sin(u * 2 * np.pi * periods) * np.sin(np.pi * u) ** 2
        return np.stack([x, y], -1)
    h_out = 90 - side * 35
    p0 = (side * 0.75, y0 + 0.9)
    p1 = (side * out, y0 + 6.5)
    p2 = (side * out, y1 - 6.5)
    p3 = (side * 0.75, y1 - 0.9)
    a = hermite(p0, h_out, p1, 90, k=7.0)
    u = np.linspace(0, 1, 800)[:, None]
    mid = np.asarray(p1) + (np.asarray(p2) - np.asarray(p1)) * u
    mid[:, 0] += amp * np.sin(u[:, 0] * 2 * np.pi * periods) * np.sin(np.pi * u[:, 0]) ** 2
    c = hermite(p2, 90, p3, 90 + side * 35, k=7.0)
    return cat(a, mid, c)


def run_paths():
    """The run's lines: main (start -> split), three branches (split -> merge), tail (merge -> field)."""
    c = np.asarray(SPIRAL_C)
    spiral = double_spiral(c, SPIRAL_B, SPIRAL_TH, SPIRAL_THC)
    d0 = spiral[1] - spiral[0]
    d1 = spiral[-1] - spiral[-2]
    h_in = np.degrees(np.arctan2(d0[1], d0[0]))
    h_out = np.degrees(np.arctan2(d1[1], d1[0]))
    entry, exit_ = spiral[0], spiral[-1]
    main = cat([START, (START[0], entry[1] - 6.0)],
               hermite((START[0], entry[1] - 6.0), 90, entry, h_in, k=6.0, n=60),
               spiral,
               hermite(exit_, h_out, (exit_[0], exit_[1] + 6.0), 90, k=6.0, n=60),
               hermite((exit_[0], exit_[1] + 6.0), 90, SPLIT, 90, k=14.0))
    branches = [branch_path(*b) for b in BRANCH_SHAPES]
    tail = cat([MERGE, (0.0, BRIDGE[0] - 2.0)], [(0.0, BRIDGE[0] - 2.0), (0.0, BRIDGE[1] + 2.0)],
               hermite((0.0, BRIDGE[1] + 2.0), 90, (-3.0, -4.0), 90, k=12.0, n=150),
               hermite((-3.0, -4.0), 90, (0.0, END_Y - 6.0), 90, k=12.0, n=150),
               [(0.0, END_Y - 6.0), (0.0, END_Y + 0.001)])
    return Path(main), [Path(b) for b in branches], Path(tail)


# ---------------------------------------------------------------------------------------------
# the dominoes
# ---------------------------------------------------------------------------------------------
DTYPE = np.dtype([('x', 'f4'), ('y', 'f4'), ('yaw', 'f4'), ('line', 'i2'), ('k', 'i2'), ('col', 'u1', 3),
                  ('face', 'u1', 3)])
# line ids: 0 main, 1..3 branches (red, gold, blue), 4 tail, 5 feeder east, 6 feeder west, 100 + c: field column c
LINE_MAIN, LINE_TAIL, FEED_E, FEED_W, FIELD = 0, 4, 5, 6, 100
BRANCH_COLS = [RED, GOLD, BLUE]


def _line_dominoes(path, line, s0=0.0, s1=None, cols=None, hue0=0.0):
    s1 = path.length if s1 is None else s1
    s = np.arange(s0, s1 + 1e-6, SPACING)
    p = path.point(s)
    t = path.tangent(s)
    out = np.zeros(len(s), DTYPE)
    out['x'], out['y'] = p[:, 0], p[:, 1]
    out['yaw'] = np.arctan2(t[:, 1], t[:, 0])
    out['line'] = line
    out['k'] = np.arange(len(s))
    for i in range(len(s)):
        c = cols[i] if cols is not None else rainbow(s[i] + hue0 * 36.0)
        out['col'][i] = c
        out['face'][i] = c
    return out


def rainbow(s, period=36.0):
    """The run's colours: the hue turns as the line goes, one full turn every `period` blocks."""
    import colorsys
    h = (s / period) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.72, 0.96)
    return (int(r * 255), int(g * 255), int(b * 255))


class Layout:
    def __init__(self, picture=None):
        self.main, self.branches, self.tail = run_paths()
        parts = [_line_dominoes(self.main, LINE_MAIN)]
        for b, (path, col) in enumerate(zip(self.branches, BRANCH_COLS)):
            parts.append(_line_dominoes(path, 1 + b, 0.0, path.length, cols=[col] * 400))
        parts.append(_line_dominoes(self.tail, LINE_TAIL, SPACING, hue0=self.main.length / 36.0))
        run = np.concatenate(parts)
        feeder = self._feeder()
        n_avail = TOTAL - len(run) - len(feeder)
        self.nrow = -(-n_avail // NCOL)
        drop = self.nrow * NCOL - n_avail            # start the run a few dominoes later so the total is exact
        main = run[run['line'] == LINE_MAIN][drop:]
        main['k'] -= main['k'][0]
        self.run = np.concatenate([main, run[run['line'] != LINE_MAIN]])
        self.feeder = feeder
        self.field = self._field(picture)
        self.all = np.concatenate([self.run, self.feeder, self.field])
        assert len(self.all) == TOTAL, len(self.all)

    def _feeder(self):
        n = NCOL // 2
        out = np.zeros(2 * n, DTYPE)
        xs = (np.arange(n) + 0.5) * PITCH
        out['x'][:n], out['x'][n:] = xs, -xs
        out['y'] = FEEDER_Y
        out['yaw'][:n], out['yaw'][n:] = 0.0, np.pi
        out['line'][:n], out['line'][n:] = FEED_E, FEED_W
        out['k'][:n], out['k'][n:] = np.arange(n), np.arange(n)
        for i in range(2 * n):
            out['col'][i] = out['face'][i] = RUN_COLS[0]
        return out

    def _field(self, picture):
        out = np.zeros(NCOL * self.nrow, DTYPE)
        c, r = np.meshgrid(np.arange(NCOL), np.arange(self.nrow), indexing='ij')
        out['x'] = ((c - (NCOL - 1) / 2.0) * PITCH).ravel()
        out['y'] = (FIELD_Y0 + r * SPACING).ravel()
        out['yaw'] = np.pi / 2
        out['line'] = (FIELD + c).ravel()
        out['k'] = r.ravel()
        out['col'] = RUN_COLS[0]
        if picture is not None:
            # picture[row, col], row 0 = the top of the picture = the field's north end
            img = np.asarray(picture)
            assert img.shape[:2] == (self.nrow, NCOL), (img.shape, self.nrow)
            out['face'] = img[::-1].transpose(1, 0, 2).reshape(-1, 3)
        else:
            out['face'] = (200, 200, 200)
        return out

    def counts(self):
        run = self.run
        return {'main': int((run['line'] == LINE_MAIN).sum()),
                'branches': [int((run['line'] == 1 + b).sum()) for b in range(3)],
                'tail': int((run['line'] == LINE_TAIL).sum()), 'feeder': len(self.feeder),
                'field': len(self.field), 'rows': self.nrow, 'total': len(self.all)}


if __name__ == '__main__':
    L = Layout()
    print(L.counts())
    print('lengths: main %.1f, branches %s, tail %.1f' % (L.main.length, [round(b.length, 2) for b in L.branches],
                                                           L.tail.length))
    print('min curvature radius: main %.2f, branches %s, tail %.2f' % (
        L.main.curvature_radius(), [round(b.curvature_radius(), 2) for b in L.branches], L.tail.curvature_radius()))
