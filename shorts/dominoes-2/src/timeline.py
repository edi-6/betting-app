"""The edit: which moment of the simulation each video frame shows, and when things happen.

The chain reactions are recorded in simulation time (dominoes.py); the video plays them at a speed that changes
with the story: real time for the punch, fast along the run and through the small growth dominoes, a little slower
as the giant falls and in slow motion as it slams down, brisk for the race, slow motion for the finish (red falls
short, gold and blue photo finish), fast over the bridge, slow as the last domino falls short, real time for the
silence and the storm, slow for the first lightning strike, then the flight round the field and the field falling
towards the cliff, and real time for the end. Speeds change smoothly (an eased ramp), never in a jump.
"""
import numpy as np

import layout as LY

FPS = 60
T0 = -0.55                        # simulation time of the first frame (the punch lands at 0)
SWING_AT = -0.15                  # the swing starts (the fist lands halfway through, at 0)
ORBIT = 4.6                       # from the first strike to the camera settling on the cliff's edge
HOLD = 1.15                       # from 100,000 to the turn
TURN = 0.55                       # the turn
STARE = 1.05                      # face to face
BLACK = 0.45                      # the glitch to black
HEROBRINE_END = (0.0, LY.FINAL_EYE[1] + 2.05, float(LY.CLIFF_Z))


def speed_keys(ev):
    """(simulation time, playback speed) from the recorded events."""
    return [
        (T0, 1.0),
        (0.5, 1.55),                              # the run
        (ev['grow'] - 0.5, 1.25),
        (ev['grow'] + 0.9, 1.8),                  # the growth: the small ones go quickly
        (ev['giant'] - 1.0, 1.15),
        (ev['giant'] - 0.1, 0.85),                # the giant falls
        (ev['land'] - 0.14, 0.33),                # ... and slams down
        (ev['land'] + 0.45, 1.0),
        (ev['race'] + 1.0, 1.55),                 # the race
        (ev['red_end'] - 0.3, 0.32),              # the finish: red falls short, gold and blue
        (ev['merge'] + 0.25, 1.0),
        (ev['merge'] + 0.6, 1.95),                # the tail and the bridge
        (ev['stop'] - 1.0, 0.6),                  # the last one falls... short
        (ev['stop'] + 0.15, 1.0),                 # silence, the storm comes
        (ev['strike'] - 0.1, 0.5),                # lightning
        (ev['strike'] + 0.6, 1.15),               # the flight round the field
        (ev['arrive'] - 0.5, 1.9),                # the field falls towards the cliff
        (ev['done'] - 1.0, 1.3),
        (ev['done'] + 0.05, 1.0),                 # the end, in real time
    ]


def schedule(ev, ease=0.25):
    """Simulation time of every video frame."""
    keys = speed_keys(ev)
    t_end = ev['video_end']
    ts = []
    t = T0
    cur = keys[0][1]
    while t < t_end:
        ts.append(t)
        goal = keys[0][1]
        for tk, v in keys:
            if t >= tk:
                goal = v
        cur += (goal - cur) * min(1.0, 1.0 / (ease * FPS))
        t += cur / FPS
    return np.array(ts)


def events(chains, t_done):
    C = chains
    red = C.race[0]
    ok = np.isfinite(red.t0)
    ev = {'grow': C.t_grow, 'giant': C.t_giant, 'land': C.t_land, 'race': C.t_race,
          'red_end': float(red.t0[ok][-1]), 'red_stop': C.t_red_stop, 'merge': C.t_merge,
          'stop': C.t_stop, 'strike': C.t_strike0, 'strikes': [s for s in C.strikes], 'field0': C.t_field0,
          'end': C.t_end, 'done': t_done}
    ev['arrive'] = ev['strike'] + ORBIT
    ev['turn'] = t_done + HOLD
    ev['black'] = ev['turn'] + TURN + STARE
    ev['video_end'] = ev['black'] + BLACK
    return ev
