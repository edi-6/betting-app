"""The press test in simulation time (when each block goes on the platen) and the edit that shows it: slow motion
around the big moments, camera shots, the cold open. Every block gives (and the press gives, at the end) exactly on
a beat of the 130 BPM track: the tests' start times are nudged until the video time of each break is on the grid.
"""
import numpy as np

import press as P

FPS = 30
BPM = 130.0
BEAT = 60.0 / BPM
GRID = BEAT / 2          # breaks land on eighth notes
ALIGN = {'bedrock': 4 * BEAT}             # ... and the press gives on a bar line: the drop
KINDS = ('grass', 'glass', 'melon', 'slime', 'tnt', 'chest', 'diamond', 'obsidian', 'bedrock')
T0_NOMINAL = (0.0, 1.9, 3.4, 5.3, 8.55, 10.9, 12.8, 15.4, 18.6)
MIN_GAP = {'grass': 0.6, 'glass': 0.6, 'melon': 0.6, 'slime': 0.65, 'tnt': 0.45, 'chest': 0.6,
           'diamond': 0.6, 'obsidian': 0.6}
# slow motion around a test's break: (from, to) seconds relative to the break, the time scale
SLOW = {'glass': (-0.03, 0.42, 0.3), 'tnt': (-0.02, 0.5, 0.35), 'diamond': (-0.02, 0.3, 0.5),
        'bedrock': (-0.04, 0.75, 0.3)}
WAIT = 0.28                       # the block pops in, a beat, the ram comes down
RAMP = 0.08                       # sim seconds to ease in and out of slow motion
TAIL = 3.4                        # sim seconds after the press breaks
COLD = dict(tau=(1.45, 2.95), dur=1.5)    # the cold open: this stretch of the bedrock test, at normal speed
REWIND = 0.4                      # video seconds of rewinding after it
MAIN_START = COLD['dur'] + REWIND


def sec(t):
    return int(round(t * FPS))


def _ease(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


# ---------------------------------------------------------------------------------------------
# the story
# ---------------------------------------------------------------------------------------------
def _tests(t0s):
    return [dict(kind=k, t0=float(t), wait=WAIT) for k, t in zip(KINDS, t0s)]


def events(tests):
    """[(sim time, test index, event name)] for every scripted event."""
    out = []
    for i, ts in enumerate(tests):
        for name, off in P.event_offsets(ts).items():
            out.append((ts['t0'] + off, i, name))
    return sorted(out)


def slow_windows(tests):
    wins = []
    for ts in tests:
        if ts['kind'] in SLOW:
            b = ts['t0'] + P.event_offsets(ts)['break']
            a0, a1, sc = SLOW[ts['kind']]
            wins.append((b + a0, b + a1, sc))
    return wins


def scale_at_sim(wins, s):
    sc = 1.0
    for (a, b, m) in wins:
        if a - RAMP < s < b + RAMP:
            w = min(_ease((s - (a - RAMP)) / RAMP), _ease(((b + RAMP) - s) / RAMP))
            sc = min(sc, 1.0 - (1.0 - m) * w)
    return sc


def warp(tests, s_end):
    """Per-frame time scales of the main edit (it runs the story continuously from 0 to s_end)."""
    wins = slow_windows(tests)
    out = []
    s = 0.0
    while s < s_end - 1e-9:
        sc = scale_at_sim(wins, s + 0.5 * scale_at_sim(wins, s) / FPS)
        out.append(sc)
        s += sc / FPS
    return np.array(out)


def video_time(tests, s_end, s):
    """Video time (s, from the start of the main edit) at which the story reaches sim time s."""
    sc = warp(tests, s_end)
    acc = np.concatenate([[0.0], np.cumsum(sc / FPS)])
    f = np.searchsorted(acc, s) - 1
    f = int(np.clip(f, 0, len(sc) - 1))
    return (f + (s - acc[f]) / (sc[f] / FPS)) / FPS


def _align(t0s):
    """Nudge the start times so every break lands on the beat grid set by the first one."""
    t0s = list(t0s)
    for k in range(1, len(KINDS)):
        tests = _tests(t0s)
        end = tests[-1]['t0'] + P.event_offsets(tests[-1])['break'] + TAIL
        v0 = video_time(tests, end, tests[0]['t0'] + P.event_offsets(tests[0])['break'])
        vk = video_time(tests, end, tests[k]['t0'] + P.event_offsets(tests[k])['break'])
        grid = ALIGN.get(KINDS[k], GRID)
        n = (vk - v0) / grid
        prev = tests[k - 1]
        prev_break = prev['t0'] + P.event_offsets(prev)['break']
        for target in sorted({np.round(n), np.ceil(n), np.ceil(n) + 1}, key=lambda x: abs(x - n)):
            d = (target - n) * grid          # video seconds; the gap before test k plays at normal speed
            if t0s[k] + d - prev_break >= MIN_GAP[KINDS[k - 1]] - 1e-6:
                break
        t0s[k:] = [t + d for t in t0s[k:]]
    return t0s


_PLAN = None


def plan():
    global _PLAN
    if _PLAN is None:
        tests = _tests(_align(T0_NOMINAL))
        end = tests[-1]['t0'] + P.event_offsets(tests[-1])['break'] + TAIL
        _PLAN = dict(tests=tests, end=end)
    return _PLAN


def cold_plan():
    """The cold open's story: the bedrock test alone."""
    return dict(tests=[dict(kind='bedrock', t0=0.0, wait=WAIT)], end=6.0)


# ---------------------------------------------------------------------------------------------
# the edit
# ---------------------------------------------------------------------------------------------
def cam_path(keys, t):
    """Keyframes (time_s, eye, target, fov), eased between keys."""
    ts = [k[0] for k in keys]
    if t <= ts[0] or len(keys) == 1:
        k = keys[0]
        return np.array(k[1], float), np.array(k[2], float), float(k[3])
    if t >= ts[-1]:
        k = keys[-1]
        return np.array(k[1], float), np.array(k[2], float), float(k[3])
    i = int(np.searchsorted(ts, t)) - 1
    a, b = keys[i], keys[i + 1]
    u = (t - a[0]) / (b[0] - a[0])
    e = float(_ease(u))
    eye = np.array(a[1], float) * (1 - e) + np.array(b[1], float) * e
    tgt = np.array(a[2], float) * (1 - e) + np.array(b[2], float) * e
    return eye, tgt, a[3] * (1 - e) + b[3] * e


def shots():
    """Camera shots of the main edit: name, the sim time it cuts in at, camera keys in video seconds from the cut.
    The block sits on the platen at z 6..10; the ram rests 7 units above it."""
    p = plan()
    t0 = {ts['kind']: ts['t0'] for ts in p['tests']}
    ev = {(ts['kind'], n): ts['t0'] + off for ts in p['tests'] for n, off in P.event_offsets(ts).items()}
    return [
        # the press, then straight in: grass
        dict(name='grass', at=0.0,
             keys=[(0.0, (3.0, -46.0, 15.0), (0.0, 0.0, 13.0), 42.0),
                   (0.75, (1.4, -29.0, 9.6), (0.0, 0.0, 10.0), 40.0),
                   (2.0, (1.2, -27.5, 9.2), (0.0, 0.0, 9.8), 40.0)]),
        # glass, low and close: the shards fly at us
        dict(name='glass', at=t0['glass'] - 0.1, clear=True,
             keys=[(0.0, (2.5, -18.0, 7.0), (0.0, 0.0, 9.2), 42.0),
                   (2.2, (1.5, -16.5, 6.8), (0.0, 0.0, 9.0), 42.0)]),
        # melon from the left: the juice
        dict(name='melon', at=t0['melon'] - 0.1, clear=True,
             keys=[(0.0, (-15.0, -21.0, 10.5), (0.0, 0.0, 9.5), 40.0),
                   (2.0, (-13.5, -19.5, 10.0), (0.0, 0.0, 9.3), 40.0)]),
        # slime: a bit wider, it throws the ram back up
        dict(name='slime', at=t0['slime'] - 0.1, clear=True,
             keys=[(0.0, (1.5, -31.0, 10.5), (0.0, 0.0, 11.0), 40.0),
                   (3.4, (1.2, -28.5, 10.0), (0.0, 0.0, 10.6), 40.0)]),
        # TNT: primed right in front of us ...
        dict(name='tnt', at=t0['tnt'] - 0.1, clear=True,
             keys=[(0.0, (1.0, -24.0, 8.8), (0.0, 0.0, 9.4), 40.0),
                   (2.0, (0.8, -21.0, 8.6), (0.0, 0.0, 9.0), 40.0)]),
        # ... BOOM, from further back and low, in slow motion
        dict(name='tnt_boom', at=ev[('tnt', 'break')] - 0.01, snap=False,
             keys=[(0.0, (10.0, -27.0, 7.0), (0.0, 0.0, 10.5), 46.0),
                   (2.0, (9.0, -29.0, 7.4), (0.0, 0.0, 10.5), 46.0)]),
        # chest from above: the loot
        dict(name='chest', at=t0['chest'] - 0.1, clear=True,
             keys=[(0.0, (10.0, -20.0, 19.0), (0.0, 0.0, 7.5), 40.0),
                   (2.0, (9.0, -18.5, 18.0), (0.0, 0.0, 7.5), 40.0)]),
        # diamond, macro: it cracks stage by stage
        dict(name='diamond', at=t0['diamond'] - 0.1, clear=True,
             keys=[(0.0, (1.5, -20.0, 8.8), (0.0, 0.0, 9.0), 40.0),
                   (2.2, (0.8, -14.5, 8.4), (0.0, 0.0, 8.6), 40.0),
                   (3.0, (0.8, -14.0, 8.4), (0.0, 0.0, 8.6), 40.0)]),
        # obsidian, low from the right: sparks
        dict(name='obsidian', at=t0['obsidian'] - 0.1, clear=True,
             keys=[(0.0, (15.0, -20.0, 6.2), (0.0, 0.0, 9.5), 40.0),
                   (3.2, (11.5, -16.5, 6.4), (0.0, 0.0, 9.2), 40.0)]),
        # bedrock: it goes on, the ram comes down
        dict(name='bedrock', at=t0['bedrock'] - 0.1, clear=True,
             keys=[(0.0, (1.5, -30.0, 10.0), (0.0, 0.0, 10.5), 40.0),
                   (1.2, (1.2, -26.0, 9.2), (0.0, 0.0, 9.6), 40.0)]),
        # the pressure climbs: low and close on the contact
        dict(name='strain', at=ev[('bedrock', 'contact')] + 0.35,
             keys=[(0.0, (-3.0, -17.0, 6.8), (0.0, 0.0, 9.6), 42.0),
                   (1.5, (-2.2, -15.0, 6.9), (0.0, 0.0, 9.6), 42.0)]),
        # the hoses give: the whole press shaking
        dict(name='hoses', at=ev[('bedrock', 'hoses')] - 0.05,
             keys=[(0.0, (4.0, -44.0, 14.0), (0.0, 0.0, 14.0), 44.0),
                   (1.5, (3.5, -40.0, 13.0), (0.0, 0.0, 13.0), 44.0)]),
        # the press blows apart (slow motion)
        dict(name='boom', at=ev[('bedrock', 'break')] - 0.03, snap=False,
             keys=[(0.0, (2.0, -40.0, 11.0), (0.0, 0.0, 12.0), 46.0),
                   (2.6, (2.0, -44.0, 12.0), (0.0, 0.0, 12.0), 46.0)]),
        # ... and the bedrock is still there
        dict(name='survivor', at=ev[('bedrock', 'break')] + 1.1,
             keys=[(0.0, (2.2, -22.0, 7.4), (0.0, 0.0, 8.6), 40.0),
                   (2.0, (1.0, -14.0, 7.3), (0.0, 0.0, 8.4), 38.0)]),
    ]


def cold_shot():
    return dict(name='cold', keys=[(0.0, (-4.0, -19.0, 6.2), (0.0, 0.0, 10.0), 44.0),
                                   (1.5, (-3.2, -16.5, 6.4), (0.0, 0.0, 9.8), 42.0)])


def cut_frames():
    """Frame of the main edit at which each shot starts: the first frame at its sim time, moved by up to a few
    frames onto the nearest eighth note (except the cuts that must hit an explosion exactly)."""
    p = plan()
    sc = warp(p['tests'], p['end'])
    acc = np.concatenate([[0.0], np.cumsum(sc / FPS)])[:-1]
    g0, _ = beat_grid()
    out = []
    for k, sh in enumerate(shots()):
        f = int(np.searchsorted(acc, sh['at'] - 1e-9))
        if k and sh.get('snap', True):
            cands = range(max(out[-1] + 6, f - 5), f + 3)
            f = min(cands, key=lambda c: abs(((MAIN_START + c / FPS - g0) / GRID + 0.5) % 1.0 - 0.5))
        out.append(0 if k == 0 else f)
    return out


def schedule():
    """Frame-by-frame plan of the main edit: [(shot index, frame in shot, time scale)]."""
    p = plan()
    sc = warp(p['tests'], p['end'])
    cuts = cut_frames()
    out = []
    si = 0
    f0 = 0
    for i, x in enumerate(sc):
        while si + 1 < len(cuts) and i >= cuts[si + 1]:
            si += 1
            f0 = i
        out.append((si, i - f0, float(x)))
    return out


def beat_grid():
    """(time of a beat in video seconds, beat length): the grid every break sits on."""
    p = plan()
    ts = p['tests'][0]
    v = video_time(p['tests'], p['end'], ts['t0'] + P.event_offsets(ts)['break'])
    return MAIN_START + v, BEAT


if __name__ == '__main__':
    p = plan()
    print('t0', [round(ts['t0'], 3) for ts in p['tests']], 'end', round(p['end'], 2))
    sc = warp(p['tests'], p['end'])
    print('main edit', round(len(sc) / FPS, 2), 's; total', round(MAIN_START + len(sc) / FPS, 2), 's')
    g0, b = beat_grid()
    for (s, i, name) in events(p['tests']):
        v = MAIN_START + video_time(p['tests'], p['end'], s)
        print(f'  {KINDS[i]:9s} {name:8s} sim {s:6.2f}  video {v:6.2f}  beats {(v - g0) / b:6.2f}')
    sched = schedule()
    sh = shots()
    last = None
    for i, (si, f, x) in enumerate(sched):
        if si != last:
            v = MAIN_START + i / FPS
            print(f'  cut -> {sh[si]["name"]:9s} at video {v:6.2f}  ({(v - g0) / b:6.2f} beats)')
            last = si
