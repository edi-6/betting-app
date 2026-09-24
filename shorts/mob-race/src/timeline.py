"""The race that is shown, and the edit.

The race is seed 111 of the search (tools: the race is deterministic and a video frame always advances a whole
number of fixed physics steps, so slow motion and fast forward never change what happens): the Warden gets over
the first trapdoor 0.07 s before it drops, scrapes through the third as the last one across, and wins the final.

The edit is a list of speed keys in race time: the countdown plays in slow motion before GO, the quiet stretches a
little fast, the big moments slow.
"""
import numpy as np

import race as RC

FPS = RC.FPS
SUB = RC.SUB
SEED = 111
LINEUP = ['creeper', 'warden', 'pig', 'skeleton', 'enderman', 'villager', 'fox', 'blaze']
COUNTDOWN = 2.5                   # video seconds before GO
T_GO = RC.GO_STEP / (FPS * SUB)    # race time at GO
# (race time from which on, playback speed); speeds are multiples of 1 / SUB
SPEED = [
    (T_GO, 1.2),
    (7.45, 0.3),                   # the Warden and the first trapdoor
    (8.25, 1.2),
    (9.7, 1.3),                    # round 2
    (16.5, 1.0),                   # its trapdoor
    (18.8, 1.3),                   # round 3
    (21.05, 0.4),                  # TNT
    (21.9, 1.3),
    (27.15, 0.3),                  # the Warden, last one over the third trapdoor
    (27.85, 1.2),
    (30.8, 1.0),                   # the final
    (32.35, 0.4),                  # the finish
    (33.75, 1.0),
]
T_END = 36.9                       # race time at the end of the video


def speed_at(t):
    s = SPEED[0][1]
    for t0, v in SPEED:
        if t >= t0:
            s = v
    return s


def schedule():
    """Physics steps per video frame for the whole video: the countdown (the race's first GO_STEP steps spread
    over COUNTDOWN seconds), then the race, speeds eased between keys in steps of 1 / SUB."""
    out = []
    n_cd = int(round(COUNTDOWN * FPS))
    steps = RC.GO_STEP
    base = [steps // n_cd + (1 if k < steps % n_cd else 0) for k in range(n_cd)]
    out.extend(sorted(base))
    t = T_GO
    cur = speed_at(t)
    while t < T_END:
        goal = speed_at(t)
        if abs(goal - cur) > 1e-6:                   # ease: one step of 0.1 per frame
            cur = cur + np.sign(goal - cur) * min(abs(goal - cur), 1.0 / SUB)
        n = max(1, int(round(cur * SUB)))
        out.append(n)
        t += n / (FPS * SUB)
    return out


def frame_times():
    """Race time at the end of every video frame."""
    return np.cumsum(schedule()) / (FPS * SUB)


if __name__ == '__main__':
    sc = schedule()
    ft = frame_times()
    print(f'{len(sc)} frames = {len(sc) / FPS:.2f} s of video; race time at the end {ft[-1]:.2f}')
    for t0, v in SPEED:
        f = int(np.searchsorted(ft, t0))
        print(f'  race {t0:6.2f}  video {f / FPS:6.2f}  speed {v}')
