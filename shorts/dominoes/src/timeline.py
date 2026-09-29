"""The edit: which moment of the simulation each video frame shows.

The chain reactions are recorded in simulation time (dominoes.py); the video plays them at a speed that changes
with the story: real time for the punch, fast through the spiral, slow for the race's photo finish, slow as the
last domino of the run falls short, real time for the silence and the creeper, slow for the blast, brisk while the
field falls, and real time for the end. Speeds change smoothly (an eased ramp), never in a jump.
"""
import numpy as np

FPS = 30
T0 = -0.55                        # simulation time of the first frame (the punch lands at 0)
SWING_AT = -0.15                  # the swing starts (the fist lands halfway through, at 0)
PAUSE = 1.1                       # silence after the run stops
WALK = 2.1                        # the creeper walks in
TURN = 0.4                        # ... turns to look at the camera
FUSE = 1.5                        # ... and its fuse burns (as long as in the game)
BLAST_AFTER = PAUSE + WALK + TURN + FUSE
OUTRO = 3.6                       # after the field's last domino: the picture, then the twist


def speed_keys(ev):
    """(simulation time, playback speed) from the recorded events."""
    return [
        (T0, 1.0),
        (0.55, 1.25),
        (2.3, 1.6),                               # the spiral
        (ev['split'] - 1.0, 1.25),
        (ev['split'] - 0.1, 1.0),                 # the race
        (ev['merge'] - 0.4, 0.33),                # photo finish
        (ev['merge'] + 0.25, 1.45),               # the tail and the bridge
        (ev['stop'] - 1.0, 0.5),                  # the last one falls... short
        (ev['stop'] + 0.2, 1.0),                  # silence, the creeper
        (ev['blast'] - 0.06, 0.3),                # the blast
        (ev['blast'] + 0.55, 1.0),
        (ev['blast'] + 1.5, 1.32),                # the field falls
        (ev['end'] - 0.5, 1.0),
    ]


def schedule(ev, ease=0.25):
    """Simulation time of every video frame."""
    keys = speed_keys(ev)
    t_end = ev['end'] + OUTRO
    ts = []
    t = T0
    cur = keys[0][1]
    while t < t_end:
        ts.append(t)
        goal = keys[0][1]
        for tk, v in keys:
            if t >= tk:
                goal = v
        # ease towards the target speed (about `ease` s of video to get most of the way)
        cur += (goal - cur) * min(1.0, 1.0 / (ease * FPS))
        t += cur / FPS
    return np.array(ts)


def events(chains):
    return {'split': chains.t_split, 'merge': chains.t_merge, 'stop': chains.t_stop, 'blast': chains.t_blast,
            'end': chains.t_end, 'lit': chains.t_blast - FUSE, 'walk': chains.t_stop + PAUSE,
            'turn': chains.t_stop + PAUSE + WALK}
