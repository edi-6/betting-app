"""The camera: it frames the marbles still racing in the current round (and whoever is going into the lava), from
straight in front and a little above, zooming to fit, looking a little ahead down the course; smoothed with a
zero-phase filter so it glides. During the countdown it pans along the eight racers in their stalls, and it ends
close on the winner on the podium.
"""
import numpy as np

import course as C

FOV = 40.0
TAN = np.tan(np.radians(FOV / 2.0))
ASPECT = 9.0 / 16.0
D_MIN, D_MAX = 15.0, 32.0


def fit_distance(w, h, margin=2.2):
    """Camera distance that fits a w x h box (blocks) in the 9:16 frame."""
    need = max(h + 2 * margin, (w + 2 * margin) / ASPECT)
    return float(np.clip(need / 2.0 / TAN, D_MIN, D_MAX))


def raw_track(rec, course, winner):
    """Per frame: (x, z, distance) before smoothing."""
    out = []
    n_trap = len(course.traps)
    go = next((i for i, fr in enumerate(rec) if fr['go']), 1)
    for i, fr in enumerate(rec):
        ms = fr['marbles']
        t = fr['t']
        alive = [m for m in ms if m['out'] is None]
        if not fr['go']:
            # the countdown: a slow pan along the eight stalls, close
            u = i / max(1, go - 1)
            x = -4.6 + 9.2 * (u * u * (3 - 2 * u))
            out.append((x, 0.2, 16.5))
            continue
        k = min([m['stage'] for m in alive]) if alive else n_trap
        racing = [m for m in alive if m['stage'] == k]
        pts = [(m['x'], m['z']) for m in racing]
        # whoever just went into the lava
        for m in ms:
            if m['out'] is not None and 0.0 <= t - m['out'] < 1.0:
                pts.append((m['sink'][0], m['sink'][1] - 0.8))
        if k < n_trap:
            tr = course.traps[k]
            pen = course.pens[k]
            near = [p for p in pts if abs(p[1] - tr['hinge'][1]) < 7.0]
            if near or not pts:
                # the round's end is in play: keep the trapdoor and the pen in the picture
                pts.append((tr['hinge'][0], tr['hinge'][1]))
                pts.append((pen['b'][0] * 0.7, pen['b'][1]))
        if k >= n_trap:
            # the end: the winner on the podium
            w = [m for m in ms if m['name'] == winner][0]
            out.append((w['x'], w['z'] + 1.8, D_MIN * 0.9))
            continue
        P = np.array(pts)
        lo, hi = P.min(0), P.max(0)
        d = fit_distance(hi[0] - lo[0], hi[1] - lo[1])
        cx = (lo[0] + hi[0]) / 2.0
        cz = (lo[1] + hi[1]) / 2.0 - 1.4                     # look a little ahead (down)
        half_w = d * TAN * ASPECT
        lim = max(0.0, C.HALF + 1.2 - half_w)
        cx = float(np.clip(cx, -lim, lim))
        out.append((cx, cz, d))
    return np.array(out)


def smooth(track, tau_frames):
    """Zero-phase exponential smoothing (forward, then backward) of every column."""
    a = 1.0 / max(1.0, tau_frames)
    y = np.asarray(track, float).copy()
    for i in range(1, len(y)):
        y[i] = y[i - 1] + (y[i] - y[i - 1]) * a
    for i in range(len(y) - 2, -1, -1):
        y[i] = y[i + 1] + (y[i] - y[i + 1]) * a
    return y


def cameras(rec, course, winner, fps=30):
    raw = raw_track(rec, course, winner)
    tr = smooth(raw, 0.32 * fps)
    tr = smooth(tr, 0.12 * fps)
    cams = []
    for i, (x, z, d) in enumerate(tr):
        eye = (x, -d, z + 0.14 * d)
        cams.append({'eye': eye, 'target': (x, 0.0, z), 'fov': FOV})
    return cams, tr
