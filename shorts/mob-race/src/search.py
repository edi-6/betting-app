"""Find races worth showing: runs the race for a range of seeds, each with its own shuffled line-up, and prints
the ones that qualify, best first (one JSON line each).

A race qualifies when exactly one marble survives, every trapdoor opens, the anti-jam nudge helped at most twice,
the final is decided between 26 and 36 s of race time, no round keeps its losers waiting too long, and nobody
saved at a trapdoor gets over it more than a second late or falls in later. Qualifying races are ranked by close
calls (saves), lead changes and how quickly the losers go in. The race in the video is seed 111:

python search.py 0 200
python search.py 111 112          # a single race, with its line-up for timeline.py
"""
import json
import sys

import numpy as np

import course as C
import heads as H
import race as RC


def lineup_for(seed):
    return [str(x) for x in np.random.default_rng(1000 + seed).permutation(H.ORDER)]


def run(seed, t_max=60.0):
    lineup = lineup_for(seed)
    r = RC.Race(C.build(), lineup, seed=seed)
    events = []
    leaders = []
    while r.t < t_max:
        for e in r.step(RC.SUB):
            if e[0] != 'hit':
                events.append((round(r.t, 3),) + tuple(e))
        if r.go:
            leaders.append((r.t - r.go_t, r.standings()[0]))
        last = r.traps[-1]['open_at']
        if last is not None and len(r.alive()) <= 1 and r.t > last + 2.0:
            break
    return r, lineup, events, leaders


def score(r, lineup, events, leaders):
    alive = [m['name'] for m in r.alive()]
    opens = {e[2]: e[0] for e in events if e[1] == 'open'}
    saves = [dict(t=e[0], trap=e[2], name=e[3], late=round(e[0] - opens.get(e[2], e[0]), 2))
             for e in events if e[1] == 'saved']
    outs = {e[2]: e[0] for e in events if e[1] == 'lava'}
    nudges = sum(1 for e in events if e[1] == 'nudge')
    lead_changes = 0
    prev = None
    for t, name in leaders:
        if t >= 2.0 and prev is not None and name != prev:
            lead_changes += 1
        prev = name
    # how long each round's losers take to go in after its trapdoor opens
    delays = []
    for k, tr in enumerate(r.traps):
        if tr['open_at'] is None:
            continue
        rel = r.pens[k]['release'] if k < len(r.pens) and not r.pens[k].get('podium') else None
        lost = [t for t in outs.values() if t >= tr['open_at'] and (rel is None or t <= rel)]
        if lost:
            delays.append(max(lost) - tr['open_at'])
    t_end = r.traps[-1]['open_at']
    ok = (len(alive) == 1 and all(tr['open_at'] is not None for tr in r.traps) and nudges <= 2
          and t_end is not None and 26.0 <= t_end <= 36.0 and delays and max(delays) < 2.6
          and not any(s['late'] > 1.0 for s in saves) and not any(s['name'] in outs for s in saves))
    points = 3.0 * len(saves) + 0.5 * min(lead_changes, 8) - (max(delays) if delays else 9.0)
    return dict(ok=bool(ok), points=round(points, 2), winner=alive[0] if len(alive) == 1 else None,
                t_end=round(t_end, 2) if t_end is not None else None, nudges=nudges, saves=saves,
                lead_changes=lead_changes, max_delay=round(max(delays), 2) if delays else None,
                out=list(outs), lineup=lineup,
                # race times to put the slow motion on (SPEED in timeline.py)
                opens=[round(tr['open_at'], 2) if tr['open_at'] is not None else None for tr in r.traps],
                booms=[e[0] for e in events if e[1] == 'boom'])


def main(first, last):
    found = []
    for seed in range(first, last):
        s = score(*run(seed))
        s['seed'] = seed
        if s['ok'] or last - first == 1:
            found.append(s)
    for s in sorted(found, key=lambda s: -s['points']):
        print(json.dumps(s))


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]))
