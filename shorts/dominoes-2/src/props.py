"""The dominoes' poses at a moment, as instance arrays for the renderer (pos3 quat4 scale var fade flash).

A domino's recorded pose is (dx forwards along its line, z of its centre, pitch forwards); its heading on the ground
is its yaw. Its orientation is the yaw about z, then the pitch about its own y axis (which tips the top forwards).
The growth dominoes and the giant are the same domino scaled up.
"""
import numpy as np

import blocks as BL
import layout as LY

BLANK = (236, 232, 222)             # a standing field domino's face: the same off-white as its back


def quat_yaw_pitch(yaw, pitch):
    sz, cz = np.sin(yaw / 2), np.cos(yaw / 2)
    sp, cp = np.sin(pitch / 2), np.cos(pitch / 2)
    return np.stack([-sz * sp, cz * sp, sz * cp, cz * cp], -1)


class Dominoes:
    def __init__(self, layout, chains, kinds=None):
        self.L = layout
        self.C = chains
        A = layout.all
        self.A = A
        self.paint = BL.pack_colour(A['face'])
        self.cos = np.cos(A['yaw'])
        self.sin = np.sin(A['yaw'])
        self.f0 = len(layout.run) + len(layout.feeder)
        # which prop kind each domino is drawn as
        self.kind = np.array(['domino'] * self.f0 + ['tile'] * len(layout.field), dtype=object)
        if kinds is not None:
            self.kind[self.f0:][kinds == 1] = 'text'
            self.kind[self.f0:][kinds == 2] = 'eye'
        self.sel = {k: np.nonzero(self.kind == k)[0] for k in ('domino', 'tile', 'text', 'eye')}
        self.lines = chains.lines()
        f = layout.field
        self.fk = f['k'].astype(int)
        self.fc = (f['line'] - LY.FIELD).astype(int)

    def poses(self, t):
        """(N, 3): dx, z, pitch for every domino, in the order of layout.all."""
        out = np.zeros((len(self.A), 3))
        out[:, 1] = LY.H * self.A['s'] / 2
        C = self.C
        for line, ln in self.lines.items():
            out[C.idx[line]] = ln.poses(t)
        col = C.column
        # every field column is the canonical column, shifted in time
        k, c = self.fk, self.fc
        u = (t - (col.t0[k] + C.col_start[c])) * 240.0
        M = col.traj.shape[1]
        i0 = np.clip(np.floor(u).astype(int), 0, M - 1)
        i1 = np.clip(i0 + 1, 0, M - 1)
        fr = np.clip(u - np.floor(u), 0, 1)[:, None]
        p = col.traj[k, i0] * (1 - fr) + col.traj[k, i1] * fr
        p[~(u >= 0)] = (0.0, LY.H / 2, 0.0)
        out[self.f0:] = p
        return out

    def positions(self, P):
        A = self.A
        return np.stack([A['x'] + self.cos * P[:, 0], A['y'] + self.sin * P[:, 0], P[:, 1]], -1)

    def instances(self, t, fade=None):
        """dict kind -> float32 (M, 11), and the poses."""
        P = self.poses(t)
        A = self.A
        arr = np.zeros((len(A), 11), np.float32)
        arr[:, 0:3] = self.positions(P)
        arr[:, 3:7] = quat_yaw_pitch(A['yaw'].astype(float), P[:, 2])
        arr[:, 7] = A['s']
        # the field's picture faces show their colour as they tip over (a standing field is a blank canvas)
        f0 = self.f0
        u = np.clip((P[f0:, 2] - 0.12) / 0.3, 0.0, 1.0)[:, None]
        col = A['face'][f0:].astype(float) * u + np.array(BLANK) * (1 - u)
        paint = self.paint.copy()
        paint[f0:] = BL.pack_colour(np.rint(col))
        arr[:, 8] = paint
        arr[:, 9] = 1.0 if fade is None else fade
        out = {}
        for kind, sel in self.sel.items():
            if len(sel):
                out[kind] = arr[sel]
        return out, P

    def fallen(self, t):
        """How many dominoes have started to fall by time t (the counter)."""
        n = 0
        for ln in self.lines.values():
            n += int((ln.t0 <= t).sum())
        C = self.C
        n += int(((C.column.t0[None, :] + C.col_start[:, None]) <= t).sum())
        return n
