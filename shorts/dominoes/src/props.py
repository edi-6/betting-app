"""The dominoes' poses at a moment, as instance arrays for the renderer (pos3 quat4 scale var fade flash).

A domino's recorded pose is (dx forwards along its line, z of its centre, pitch forwards); its heading on the ground
is its yaw. Its orientation is the yaw about z, then the pitch about its own y axis (which tips the top forwards).
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
    def __init__(self, layout, chains, eyes=None):
        self.L = layout
        self.C = chains
        A = layout.all
        self.A = A
        self.paint = BL.pack_colour(A['face'])
        self.cos = np.cos(A['yaw'])
        self.sin = np.sin(A['yaw'])
        # which prop kind each domino is drawn as
        n_run = len(layout.run) + len(layout.feeder)
        self.kind = np.array(['domino'] * n_run + ['tile'] * len(layout.field), dtype=object)
        if eyes is not None:
            # eyes[row, col] with row 0 = the top of the picture = the field's last row
            f = layout.field
            col = np.rint(f['x'] / LY.PITCH + (LY.NCOL - 1) / 2.0).astype(int)
            row = layout.nrow - 1 - f['k'].astype(int)
            is_eye = eyes[row, col]
            self.kind[n_run:][is_eye] = 'eye'
        self.override = {}                         # index -> (pos3, quat4) for dominoes the blast threw

    def poses(self, t):
        """(N, 3): dx, z, pitch for every domino, in the order of layout.all."""
        out = np.zeros((len(self.A), 3))
        out[:, 1] = LY.H / 2
        C = self.C
        lines = {LY.LINE_MAIN: C.main, LY.LINE_TAIL: C.tail, 1: C.branches[0], 2: C.branches[1],
                 3: C.branches[2]}
        if hasattr(C, 'feed'):
            lines.update(C.feed)
        for line, ln in lines.items():
            out[C.idx[line]] = ln.poses(t)
        if hasattr(C, 'column'):
            col = C.column
            # every field column is the canonical column, shifted in time
            f0 = len(self.L.run) + len(self.L.feeder)
            k = self.A['k'][f0:].astype(int)
            c = self.A['line'][f0:].astype(int) - LY.FIELD
            u = (t - (col.t0[k] + C.col_start[c])) * 240.0
            M = col.traj.shape[1]
            i0 = np.clip(np.floor(u).astype(int), 0, M - 1)
            i1 = np.clip(i0 + 1, 0, M - 1)
            fr = np.clip(u - np.floor(u), 0, 1)[:, None]
            p = col.traj[k, i0] * (1 - fr) + col.traj[k, i1] * fr
            p[u < 0] = (0.0, LY.H / 2, 0.0)
            out[f0:] = p
        return out

    def instances(self, t, fade=None):
        """dict kind -> float32 (M, 11)."""
        P = self.poses(t)
        A = self.A
        pos = np.stack([A['x'] + self.cos * P[:, 0], A['y'] + self.sin * P[:, 0], P[:, 1]], -1)
        q = quat_yaw_pitch(A['yaw'].astype(float), P[:, 2])
        for i, (p_, q_) in self.override.items():
            pos[i] = p_
            q[i] = q_
        arr = np.zeros((len(A), 11), np.float32)
        arr[:, 0:3] = pos
        arr[:, 3:7] = q
        arr[:, 7] = 1.0
        # the field's picture faces show their colour as they tip over (a standing field is a blank canvas)
        f0 = len(self.L.run) + len(self.L.feeder)
        u = np.clip((P[f0:, 2] - 0.12) / 0.3, 0.0, 1.0)[:, None]
        col = A['face'][f0:].astype(float) * u + np.array(BLANK) * (1 - u)
        paint = self.paint.copy()
        paint[f0:] = BL.pack_colour(np.rint(col))
        arr[:, 8] = paint
        arr[:, 9] = 1.0 if fade is None else fade
        out = {}
        for kind in ('domino', 'tile', 'eye'):
            sel = self.kind == kind
            if sel.any():
                out[kind] = arr[sel]
        return out, P

    def fallen(self, t):
        """How many dominoes have started to fall by time t (the counter)."""
        C = self.C
        n = 0
        for ln in [C.main, C.tail] + C.branches + list(getattr(C, 'feed', {}).values()):
            n += int((ln.t0 <= t).sum())
        if hasattr(C, 'column'):
            n += int(((C.column.t0[None, :] + C.col_start[:, None]) <= t).sum())
        return n
