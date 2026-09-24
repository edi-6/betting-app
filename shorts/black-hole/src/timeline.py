"""The story of the black hole in simulation time (seconds): when it spawns and grows, when each giant is pulled
in; and the edit that shows it: shots with camera keys and slow motion, the cold open."""
import numpy as np

FPS = 30
HOLE_XY = (-14.0, -16.0)
TILT = np.radians(32.0)          # the disk leans back, so from the usual front view we see it nearly edge-on
DISK_NORMAL = (0.0, float(np.sin(TILT)), float(np.cos(TILT)))

SPAWN = 1.2
STAGES = [(SPAWN, 0.3), (4.0, 1.4), (9.0, 4.0), (17.3, 9.0)]      # SIZE 1, 10, 100, 1,000
SIZE_LABELS = ['1', '10', '100', '1,000']
COLLAPSE = (23.7, 24.5)
BOOM = 24.7
REBORN = 29.1                  # ... and a tiny new one pops up in the crater
REBORN_C = (HOLE_XY[0], HOLE_XY[1], 5.0)


def sec(t):
    return int(round(t * FPS))


def _ease(u):
    return u * u * (3 - 2 * u)


def plan():
    x, y = HOLE_XY
    return dict(
        hole=dict(spawn=SPAWN, stages=STAGES, collapse=COLLAPSE, boom=BOOM, disk_normal=DISK_NORMAL, grow=0.7,
                  path=[(0.0, (x, y, 4.0)), (4.0, (x, y, 4.5)), (9.0, (x, y, 9.5)), (17.3, (x, y, 18.0))]),
        giants=dict(
            creeper=dict(lean=5.0, lift=6.3, eat=8.5, lean_angle=0.42, prime=6.6, explode_d=3.0, spin=1.0),
            zombie=dict(lean=9.8, sever=(10.6, [('arm_l', 11.7), ('arm_r', 12.1)]), lift=11.2, eat=13.3,
                        lean_angle=0.3, spin=1.0),
            steve=dict(lean=11.8, lift=13.9, eat=16.0, lean_angle=0.36, spin=0.6),
            warden=dict(boom=18.3, head=(23.0, -4.4, 23.5), lean=19.0, lift=19.9, eat=22.4, lean_angle=0.25,
                        swirl=0.8, spin=0.7)),
        props=dict(anvil=60, tnt=40, arrow=260),
        ejecta=dict(blocks=1600, voxels=3200))


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
    e = 0.35 * u + 0.65 * _ease(u) if len(keys) == 2 else _ease(u)
    eye = np.array(a[1], float) * (1 - e) + np.array(b[1], float) * e
    tgt = np.array(a[2], float) * (1 - e) + np.array(b[2], float) * e
    return eye, tgt, a[3] * (1 - e) + b[3] * e


def scale_at(keys, t):
    """Slow motion: keyframes (shot_time_s, time_scale), eased between keys."""
    if not keys:
        return 1.0
    ts = [k[0] for k in keys]
    if t <= ts[0]:
        return float(keys[0][1])
    if t >= ts[-1]:
        return float(keys[-1][1])
    i = int(np.searchsorted(ts, t)) - 1
    u = _ease((t - ts[i]) / (ts[i + 1] - ts[i]))
    return float(keys[i][1] * (1 - u) + keys[i + 1][1] * u)


def shots():
    """The main edit, in order. sim: the stretch of the story (simulation seconds) a shot shows; dur: its video
    seconds; scale: the shape of its slow motion (normalised so the shot covers exactly its stretch); keys:
    camera keys; track: (body name, weight) aims the camera partly at that body while it flies.
    The giants are filmed from beyond the hole on the line they are pulled along, from above: in the tall frame
    the giant stands over the hole and is dragged down into it, towards us."""
    return [
        # the four of them in the arena, a quiet moment
        dict(name='calm', sim=(0.25, 1.1), dur=0.9, scale=[(0.0, 1.0)],
             keys=[(0.0, (-62.0, -58.0, 26.0), (-9.0, -2.0, 12.0), 60.0),
                   (0.9, (-60.5, -56.5, 25.5), (-9.0, -2.0, 12.0), 59.0)]),
        # SIZE 1: it pops into existence in front of the creeper and starts nibbling at the grass
        dict(name='spawn', sim=(1.1, 3.6), dur=2.0, scale=[(0.0, 1.0), (0.45, 1.0), (0.8, 1.5)],
             keys=[(0.0, (-9.0, -27.0, 5.2), (-14.0, -16.0, 4.6), 46.0),
                   (2.0, (-10.0, -24.5, 5.0), (-14.0, -16.0, 4.6), 44.0)]),
        # SIZE 10: the ground rips open around it, everything lying around flies in
        dict(name='size10', sim=(3.6, 5.4), dur=1.8, scale=[(0.0, 1.0)],
             keys=[(0.0, (2.0, -48.0, 9.0), (-14.0, -12.0, 8.0), 58.0),
                   (1.8, (0.0, -46.0, 9.5), (-14.0, -12.0, 8.0), 58.0)]),
        # the creeper can't hold on
        dict(name='creeper_pull', sim=(5.4, 7.1), dur=1.5, scale=[(0.0, 1.0)],
             keys=[(0.0, (8.0, -42.0, 11.0), (-15.0, -8.0, 11.0), 58.0),
                   (1.5, (6.0, -43.0, 12.0), (-15.0, -9.0, 11.0), 58.0)]),
        # ... it primes on the way in and goes off right at the edge
        dict(name='creeper_boom', sim=(7.1, 8.5), dur=2.3, scale=[(0.0, 1.0), (0.5, 1.0), (0.7, 0.3), (1.6, 0.3),
                                                                   (2.0, 0.9)],
             track=('creeper', 0.3),
             keys=[(0.0, (-12.0, -41.0, 12.0), (-17.0, -12.0, 10.0), 60.0),
                   (2.3, (-12.0, -43.0, 12.5), (-16.0, -13.0, 10.0), 60.0)]),
        # SIZE 100
        dict(name='size100', sim=(8.5, 10.0), dur=1.5, scale=[(0.0, 1.0)],
             keys=[(0.0, (-8.0, -64.0, 24.0), (-14.0, -10.0, 10.0), 60.0),
                   (1.5, (-8.0, -61.0, 24.0), (-14.0, -10.0, 11.0), 60.0)]),
        # the zombie's arms are torn off first
        dict(name='zombie_arms', sim=(10.0, 11.4), dur=2.0, scale=[(0.0, 1.0), (0.35, 1.0), (0.55, 0.4), (1.3, 0.4),
                                                                   (1.6, 1.0)],
             keys=[(0.0, (14.0, -38.0, 30.0), (-26.0, -6.0, 17.0), 56.0),
                   (2.0, (13.0, -39.0, 30.0), (-25.0, -7.0, 16.0), 56.0)]),
        # then the rest of it
        dict(name='zombie_in', sim=(11.4, 13.9), dur=2.2, scale=[(0.0, 1.0)], track=('zombie', 0.35),
             keys=[(0.0, (19.0, -38.0, 34.0), (-22.0, -10.0, 14.0), 60.0),
                   (2.2, (17.0, -39.0, 33.0), (-19.0, -12.0, 13.0), 60.0)]),
        # Steve is torn off the ground ...
        dict(name='steve_pull', sim=(13.9, 14.8), dur=1.1, scale=[(0.0, 1.0)], track=('steve', 0.3),
             keys=[(0.0, (-36.0, -44.0, 32.0), (-6.0, -3.0, 17.0), 58.0),
                   (1.1, (-35.0, -45.0, 31.0), (-8.0, -5.0, 16.0), 58.0)]),
        # ... and spaghettified (the cold open's moment)
        dict(name='steve_spag', sim=(14.8, 16.35), dur=2.3, scale=[(0.0, 1.0), (0.3, 0.5), (1.4, 0.5), (1.8, 1.0)],
             track=('steve', 0.25),
             keys=[(0.0, (-48.0, -15.0, 19.0), (-13.0, -11.0, 17.0), 52.0),
                   (2.3, (-45.0, -15.0, 18.5), (-13.0, -12.0, 17.0), 50.0)]),
        # SIZE 1,000
        dict(name='size1000', sim=(16.35, 17.95), dur=1.8, scale=[(0.0, 1.0)],
             keys=[(0.0, (-10.0, -74.0, 58.0), (-8.0, -6.0, 8.0), 62.0),
                   (1.8, (-10.0, -78.0, 62.0), (-8.0, -6.0, 6.0), 62.0)]),
        # the Warden's last stand: a sonic boom, swallowed (seen over its shoulder)
        dict(name='warden_boom', sim=(17.95, 19.2), dur=1.5, scale=[(0.0, 1.0)],
             keys=[(0.0, (48.0, 14.0, 44.0), (-12.0, -14.0, 14.0), 60.0),
                   (1.5, (47.0, 13.0, 43.0), (-12.0, -14.0, 14.0), 60.0)]),
        # then it goes too
        dict(name='warden_in', sim=(19.2, 22.75), dur=3.0, scale=[(0.0, 1.0)], track=('warden', 0.3),
             keys=[(0.0, (44.0, 10.0, 42.0), (-10.0, -12.0, 14.0), 60.0),
                   (3.0, (36.0, 4.0, 38.0), (-12.0, -14.0, 15.0), 60.0)]),
        # nothing left
        dict(name='alone', sim=(22.75, 23.7), dur=0.8, scale=[(0.0, 1.0)],
             keys=[(0.0, (-12.0, -70.0, 40.0), (-14.0, -16.0, 14.0), 58.0),
                   (0.8, (-12.0, -66.0, 38.0), (-14.0, -16.0, 15.0), 57.0)]),
        # it collapses into a point...
        dict(name='collapse', sim=(23.7, 24.7), dur=1.4, scale=[(0.0, 1.0)],
             keys=[(0.0, (-12.0, -64.0, 37.0), (-14.0, -16.0, 15.0), 57.0),
                   (1.4, (-12.0, -60.0, 35.0), (-14.0, -16.0, 16.0), 55.0)]),
        # ... and blows everything back out
        dict(name='boom', sim=(24.7, 26.2), dur=1.9, scale=[(0.0, 1.0)], shake=0.5,
             keys=[(0.0, (-10.0, -82.0, 44.0), (-14.0, -16.0, 14.0), 64.0),
                   (1.9, (-10.0, -88.0, 48.0), (-14.0, -14.0, 12.0), 64.0)]),
        dict(name='rain', sim=(26.2, 28.4), dur=1.8, scale=[(0.0, 1.0)],
             keys=[(0.0, (-34.0, -62.0, 46.0), (-12.0, -8.0, 2.0), 60.0),
                   (1.8, (-32.0, -58.0, 42.0), (-12.0, -8.0, 0.0), 58.0)]),
        # ... or did it?
        dict(name='reborn', sim=(28.4, 30.0), dur=1.5, scale=[(0.0, 1.0)],
             keys=[(0.0, (-12.0, -38.0, 10.0), (-14.0, -16.0, 5.0), 50.0),
                   (1.5, (-12.6, -34.0, 8.8), (-14.0, -16.0, 5.0), 45.0)]),
    ]


def cold_open():
    """Flash-forward, shown first: Steve being spaghettified. sim: where it starts in the story; scale: its slow
    motion."""
    return [
        dict(name='cold', sim=14.0, dur=1.4, scale=0.45, track=('steve', 0.35),
             keys=[(0.0, (-34.0, -42.0, 30.0), (-8.0, -6.0, 17.0), 54.0),
                   (1.4, (-32.0, -40.0, 29.0), (-9.0, -7.0, 17.0), 50.0)]),
    ]


def shot_scales(sh):
    """Time scale of every frame of a shot, so that its frames advance the story from sim[0] to sim[1]."""
    n = sec(sh['dur'])
    shape = np.array([scale_at(sh.get('scale'), f / FPS) for f in range(n)])
    return shape * ((sh['sim'][1] - sh['sim'][0]) * FPS / shape.sum())


def schedule():
    """Frame-by-frame plan of the main edit: [(shot index, frame in shot, time scale, skip_to or None)]."""
    out = []
    for si, sh in enumerate(shots()):
        for f, sc in enumerate(shot_scales(sh)):
            out.append((si, f, float(sc), sh['sim'][0] if f == 0 else None))
    return out


def video_time_of(t_sim):
    """Main-edit video time (s) at which the story reaches t_sim."""
    tv = 0.0
    for sh in shots():
        s0, s1 = sh['sim']
        sc = shot_scales(sh)
        if s0 <= t_sim < s1:
            acc = s0
            for f, x in enumerate(sc):
                if acc + x / FPS > t_sim:
                    return tv + (f + (t_sim - acc) / (x / FPS)) / FPS
                acc += x / FPS
        tv += len(sc) / FPS
    return None


if __name__ == '__main__':
    tv = 0.0
    for sh in shots():
        sc = shot_scales(sh)
        print(f"{sh['name']:14s} video {tv:5.2f}-{tv + len(sc) / FPS:5.2f}  sim {sh['sim'][0]:6.2f} -> {sh['sim'][1]:6.2f}"
              f"  scale {sc.min():.2f}..{sc.max():.2f}")
        tv += len(sc) / FPS
    print('main edit', round(tv, 2), 's; with cold open', round(tv + sum(c['dur'] for c in cold_open()), 2))
    for name, t in (('spawn', SPAWN), ('SIZE 10', 4.0), ('creeper boom', 7.93), ('SIZE 100', 9.0),
                    ('arms', 10.6), ('zombie eaten', 13.67), ('steve eaten', 16.2), ('SIZE 1000', 17.3),
                    ('sonic boom', 18.3), ('warden eaten', 22.6), ('collapse', COLLAPSE[0]), ('BOOM', BOOM),
                    ('reborn', REBORN)):
        print(f'  {name:14s} sim {t:6.2f}  video {video_time_of(t):6.2f}')
