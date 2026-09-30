"""The cue sheet the soundtrack is built from, all measured off the flight and the valley: per frame, the rider's
speed, eye and right-hand direction and their height over the ground; and the events - the step off the spire, the
elytra opening, the fireworks, each tree, rib, tower and fall the flight passes (when, how close, on which side), the
lake run, the arch, the show's rockets, the touchdown.

python cues.py OUT.json
"""
import json
import sys

import numpy as np

import director as DR
import flight as FL
import fx
import world_valley as WV


def _passes(fl, pts, lo_hi=None, max_d=14.0):
    """For each point (x, y) (with an optional vertical extent), when the flight passes closest to it, how close
    (in 3-D, to its nearest part) and on which side (+1 right, -1 left)."""
    tt = np.arange(0.0, FL.T_LAND, 1.0 / 240)
    E = np.array([FL.view(fl, t)[1] for t in tt])
    out = []
    for k, p in enumerate(pts):
        d2 = np.hypot(E[:, 0] - p[0], E[:, 1] - p[1])
        if lo_hi is not None:
            z0, z1 = lo_hi[k]
            dz = np.where(E[:, 2] < z0, z0 - E[:, 2], np.where(E[:, 2] > z1, E[:, 2] - z1, 0.0))
            d = np.hypot(d2, dz)
        else:
            d = d2
        i = int(np.argmin(d))
        if d[i] > max_d:
            continue
        t = float(tt[i])
        _, eye, (F, R, U), v = FL.view(fl, t)
        side = float(np.sign(np.dot(np.array([p[0], p[1], eye[2]]) - eye, R)) or 1.0)
        out.append(dict(t=t, d=float(d[i]), side=side, v=float(v)))
    return sorted(out, key=lambda e: e['t'])


def build():
    fl = FL.Flight()
    meta = DR.load_world(None, fl)
    fx.setup(fl, meta)
    n = int(round(FL.DURATION * FL.FPS))
    H, org = meta['H'], meta['origin']
    frames = []
    for f in range(n):
        t = f / FL.FPS
        _, eye, (F, R, U), v = FL.view(fl, t)
        i, j = int(np.floor(eye[0])) - org[0], int(np.floor(eye[1])) - org[1]
        g = float(H[np.clip(i, 0, H.shape[0] - 1), np.clip(j, 0, H.shape[1] - 1)])
        frames.append(dict(t=t, v=round(float(v), 3), eye=[round(float(x), 3) for x in eye],
                           R=[round(float(x), 4) for x in R], agl=round(float(eye[2] - g), 2)))
    ev = {}
    ev['step'] = 0.47
    ev['elytra'] = FL.BAR / 2
    ev['drop'] = FL.BAR
    ev['fireworks'] = list(FL.FIREWORKS)
    # touchdown: the feet reach the spire's top
    for f in range(int(13.0 * FL.FPS), n):
        if frames[f]['eye'][2] < FL.CONTROL[-1][2] + 0.12:
            ev['touchdown'] = f / FL.FPS
            break
    ev.setdefault('touchdown', FL.T_LAND - 0.2)
    trees = meta.get('trees', [])
    ev['trees'] = _passes(fl, [(x, y) for x, y, _, _ in trees], [(a, b) for _, _, a, b in trees], 12.0)
    towers = meta.get('towers', [])
    ev['towers'] = _passes(fl, [(x, y) for x, y, _, _ in towers], [(a, b) for _, _, a, b in towers], 12.0)
    s0, s1 = meta['fossil']
    ev['ribs'] = [dict(t=fl.time_at(s), d=2.0, side=0.0) for s in np.arange(s0, s1, 5.0)]
    ev['falls'] = _passes(fl, meta['falls'], [(WV.ICHOR - 1, WV.MEADOW + 44)] * len(meta['falls']), 60.0)
    ev['arch'] = WV.ARCH_T
    lake = [f for f in frames if f['agl'] < 9.0 and 6.0 < f['t'] < 9.0 and f['eye'][0] > 120]
    ev['lake'] = [lake[0]['t'], lake[-1]['t']] if lake else [7.0, 8.2]
    ev['show'] = [dict(t=float(b['t']), launch=float(b['t'] - fx.RISE), c=[float(x) for x in b['c']])
                  for b in fx._ST['show']]
    ev['blubs'] = _passes(fl, [(p[0], p[1]) for p, _, _ in fx._ST['blubs']], None, 16.0)
    return dict(fps=FL.FPS, nframes=n, duration=FL.DURATION, bpm=FL.BPM, frames=frames, events=ev)


def write(path):
    cs = build()
    with open(path, 'w') as fh:
        json.dump(cs, fh)
    return cs


if __name__ == '__main__':
    cs = write(sys.argv[1] if len(sys.argv) > 1 else 'cues.json')
    ev = cs['events']
    for k, v in ev.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            print(f'{k}: {len(v)}  ' + ' '.join(f"{e['t']:.2f}" for e in v[:14]) + (' ...' if len(v) > 14 else ''))
        else:
            print(f'{k}: {v}')
