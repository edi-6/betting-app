"""The battle royale itself: 50 flag balls bouncing inside a spinning ring with a gap in it, under gravity (pymunk,
in ring units: the ring's inner radius is 1, y points down). A ball that gets all the way out through the gap is
eliminated. Each time one goes, the survivors grow a little (and the gap with them), so the last few are big and the
final duel fills the ring. A thermostat keeps every ball's energy near the same level, so they never settle into a
pile: they bounce like superballs all the way to the end.

Everything is decided by the physics; the seed only shuffles who starts where and how they fly off. `search` runs
many seeds and scores each run for pace and drama; `run(seed, record=True)` keeps a run frame by frame.

python sim.py search 400        # try seeds, print the best
python sim.py run SEED          # record one run to ../cache
"""
import math
import os
import sys
import time

import numpy as np
import pymunk

import flags as FL

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, '..', 'cache')
VERSION = 4

N = len(FL.COUNTRIES)
R = 1.0                 # inner radius of the ring
SEG_R = 0.012           # half thickness of the ring's physical wall
G = 5.0                 # gravity, ring radii / s^2
R0 = 0.09               # ball radius with all 50 inside
GROW = 0.33             # radius ~ (N / n) ** GROW
R_MAX = 0.26
GROW_TAU = 0.35         # seconds for a ball to grow to its new size
GAP_K = 1.5             # the gap's chord is GAP_K ball diameters
OMEGA0 = 1.05           # ring spin, rad/s (positive: clockwise on screen)
SPEEDUP_AT = 10         # with this many left the ring spins faster
SPEEDUP = 1.45
E_LEVEL = 0.95          # target energy per unit mass, in units of G * R
THERMO_RATE = 1.2       # 1/s
DT = 1 / 240
FPS = 60
T_MAX = 75.0
T_AFTER = 4.0           # keep simulating the winner this long
GAP0 = math.radians(52)  # where the gap starts (0 = right, 90 = bottom)

BALL, WALL = 1, 2


def radius_for(n):
    return min(R_MAX, R0 * (N / max(n, 1)) ** GROW)


def gap_for(r):
    return 2 * math.asin(min(0.95, GAP_K * r / R))


def hex_start(rng):
    """50 starting spots packed in a hexagonal cluster in the middle of the ring."""
    d = 2 * R0 * 1.12
    pts = []
    for j in range(-8, 9):
        for i in range(-8, 9):
            x = (i + 0.5 * (j % 2)) * d
            y = j * d * math.sqrt(3) / 2
            pts.append((x, y))
    pts = np.array(pts)
    pts = pts[np.argsort(np.hypot(pts[:, 0], pts[:, 1] + 0.02))][:N]
    pts[:, 1] += 0.02
    return pts[rng.permutation(N)]


class Ring:
    def __init__(self, space):
        self.space = space
        self.body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.body.angle = GAP0
        space.add(self.body)
        self.shapes = []
        self.gap = None

    def build(self, gap):
        if self.gap is not None and abs(gap - self.gap) < math.radians(0.15):
            return
        self.gap = gap
        if self.shapes:
            self.space.remove(*self.shapes)
        # the wall's inner face sits at R; the gap is centred on the body's angle 0
        a0, a1 = gap / 2 + SEG_R / R, 2 * math.pi - gap / 2 - SEG_R / R
        k = 160
        a = np.linspace(a0, a1, k + 1)
        rr = R + SEG_R
        pts = np.stack([rr * np.cos(a), rr * np.sin(a)], 1)
        self.shapes = []
        for p, q in zip(pts[:-1], pts[1:]):
            s = pymunk.Segment(self.body, tuple(p), tuple(q), SEG_R)
            s.elasticity = 1.0
            s.friction = 0.15
            s.collision_type = WALL
            self.shapes.append(s)
        self.space.add(*self.shapes)


def run(seed, record=False, verbose=False):
    rng = np.random.default_rng(seed)
    space = pymunk.Space()
    space.gravity = (0.0, G)
    space.iterations = 20
    space.collision_slop = 0.002
    ring = Ring(space)
    ring.build(gap_for(R0))
    ring.body.angular_velocity = OMEGA0

    start = hex_start(rng)
    order = rng.permutation(N)             # which country starts at which spot
    balls, shapes = [], []
    for i in range(N):
        b = pymunk.Body(1.0, pymunk.moment_for_circle(1.0, 0, R0))
        b.position = tuple(start[i])
        out = start[i] / (np.linalg.norm(start[i]) + 1e-6)
        ang = rng.uniform(0, 2 * math.pi)
        v = out * rng.uniform(1.2, 2.6) + np.array([math.cos(ang), math.sin(ang)]) * rng.uniform(0.8, 1.8)
        v[1] -= 1.2
        b.velocity = tuple(v)
        s = pymunk.Circle(b, R0)
        s.elasticity = 0.92
        s.friction = 0.05
        s.collision_type = BALL
        s.ball_id = i
        space.add(b, s)
        balls.append(b)
        shapes.append(s)

    alive = np.ones(N, bool)
    r_cur = np.full(N, R0)
    t_out = np.full(N, np.nan)
    out_state = {}
    impacts = []
    t = 0.0

    def post_solve(arb, space_, data):
        a, b = arb.shapes
        imp = arb.total_impulse.length
        if imp < 0.05:
            return
        pt = arb.contact_point_set.points[0].point_a if arb.contact_point_set.points else a.body.position
        n = arb.contact_point_set.normal
        if b.collision_type == WALL or a.collision_type == WALL:
            ball = a if a.collision_type == BALL else b
            impacts.append((t, ball.ball_id, -1, imp, pt.x, pt.y, n.x, n.y))
        else:
            impacts.append((t, a.ball_id, b.ball_id, imp, pt.x, pt.y, n.x, n.y))

    space.on_collision(BALL, WALL, post_solve=post_solve)
    space.on_collision(BALL, BALL, post_solve=post_solve)

    steps_per_frame = int(round(1 / (DT * FPS)))
    frames = []
    near = []
    in_zone = np.zeros(N, bool)
    zone_t = np.zeros(N)
    n_alive = N
    t_win = None
    sped = False
    step = 0
    while t < T_MAX:
        if step % steps_per_frame == 0 and record:
            pos = np.array([b.position for b in balls], np.float32)
            vel = np.array([b.velocity for b in balls], np.float32)
            frames.append((t, pos, vel, r_cur.astype(np.float32).copy(), alive.copy(), ring.body.angle, ring.gap))
        # grow the survivors towards their size for this many players
        target = radius_for(n_alive)
        if np.any(np.abs(r_cur[alive] - target) > 1e-5):
            k = 1 - math.exp(-DT / GROW_TAU)
            for i in np.nonzero(alive)[0]:
                r_cur[i] += (target - r_cur[i]) * k
                shapes[i].unsafe_set_radius(r_cur[i])
            ring.build(gap_for(float(r_cur[alive].max())))
        space.step(DT)
        t += DT
        step += 1
        # thermostat and elimination
        gap_ang = ring.body.angle
        for i in np.nonzero(alive)[0]:
            b = balls[i]
            x, y = b.position
            vx, vy = b.velocity
            ke = 0.5 * (vx * vx + vy * vy)
            e = ke - G * y
            de = E_LEVEL * G * R - e
            k = max(-0.4, min(0.4, de * THERMO_RATE * DT / max(ke, 0.08)))
            if ke > 1e-6:
                f = math.sqrt(1 + k)
                b.velocity = (vx * f, vy * f)
            sp = math.hypot(vx, vy)
            if sp > 6.0:
                b.velocity = (vx * 6.0 / sp, vy * 6.0 / sp)
            d = math.hypot(x, y)
            # near misses: in the mouth of the gap without going out
            rel = (math.atan2(y, x) - gap_ang + math.pi) % (2 * math.pi) - math.pi
            inz = abs(rel) < ring.gap / 2 and d > R - 1.2 * r_cur[i]
            if inz and not in_zone[i]:
                zone_t[i] = t
            if not inz and in_zone[i] and d < R:
                near.append((zone_t[i], t, i, n_alive))
            in_zone[i] = inz
            if d > R + r_cur[i] * 1.02:
                alive[i] = False
                t_out[i] = t
                out_state[i] = (t, x, y, vx, vy, r_cur[i])
                space.remove(b, shapes[i])
                n_alive -= 1
                if verbose:
                    print(f'  t={t:6.2f}  out {FL.COUNTRIES[order[i]][1]:14s}  left {n_alive}')
        if n_alive <= SPEEDUP_AT and not sped:
            ring.body.angular_velocity = OMEGA0 * SPEEDUP
            sped = True
        if n_alive == 1 and t_win is None:
            t_win = t
        if t_win is not None and t > t_win + T_AFTER:
            break

    res = {'seed': seed, 'order': order, 't_out': t_out, 't_win': t_win, 'near': near}
    if record:
        res['frames'] = frames
        res['impacts'] = np.array(impacts, np.float32).reshape(-1, 8)
        res['out_state'] = out_state
    return res


def score(res):
    """How good a video this run would make (higher is better), and why."""
    t_out = res['t_out']
    if res['t_win'] is None:
        return -1e9, 'no winner'
    te = np.sort(t_out[~np.isnan(t_out)])
    first, win = te[0], te[-1]
    duel = te[-1] - te[-2]
    trio = te[-2] - te[-3]
    mid_gaps = np.diff(np.concatenate([[0.0], te[:-3]]))
    worst = mid_gaps.max()
    s = 0.0
    s -= 6 * max(0.0, first - 0.9)
    s -= 1.5 * max(0.0, abs(win - 38.0) - 4.0)
    s -= 2.0 * max(0.0, 4.0 - duel) + 1.0 * max(0.0, duel - 9.0)
    s -= 1.0 * max(0.0, 2.0 - trio) + 0.5 * max(0.0, trio - 7.0)
    s -= 2.0 * max(0.0, worst - 3.0)
    duel_near = sum(1 for (a, b, i, n) in res['near'] if n == 2)
    s += 0.6 * min(duel_near, 4)
    # a burst of several at once is fun now and then, not all the time
    bursts = sum(1 for k in range(len(te) - 2) if te[k + 2] - te[k] < 0.4)
    s -= 0.3 * max(0, bursts - 4)
    why = (f'first {first:.2f}s  win {win:.1f}s  duel {duel:.1f}s  trio {trio:.1f}s  worst gap {worst:.1f}s  '
           f'duel near-misses {duel_near}  bursts {bursts}')
    return s, why


def standings(res):
    """Country codes from the winner (index 0) to the first out."""
    order, t_out = res['order'], res['t_out']
    t = np.where(np.isnan(t_out), np.inf, t_out)
    ranked = np.argsort(-t)
    return [FL.COUNTRIES[order[i]][0] for i in ranked]


def _search_one(seed):
    res = run(seed)
    s, why = score(res)
    st = standings(res) if res['t_win'] is not None else []
    return seed, s, why, st[:3]


def search(n, start=0):
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(4) as p:
        out = p.map(_search_one, range(start, start + n), chunksize=2)
    out.sort(key=lambda r: -r[1])
    print(f'{n} runs in {time.time() - t0:.0f}s')
    for seed, s, why, top in out[:40]:
        print(f'seed {seed:5d}  score {s:6.2f}  {why}  top3 {top}')
    return out


def save(res, path):
    fr = res['frames']
    np.savez_compressed(
        path, seed=res['seed'], order=res['order'], t_out=res['t_out'], t_win=res['t_win'],
        t=np.array([f[0] for f in fr], np.float32), pos=np.stack([f[1] for f in fr]),
        vel=np.stack([f[2] for f in fr]), rad=np.stack([f[3] for f in fr]), alive=np.stack([f[4] for f in fr]),
        ring_angle=np.array([f[5] for f in fr], np.float32), gap=np.array([f[6] for f in fr], np.float32),
        impacts=res['impacts'], near=np.array(res['near'], np.float32).reshape(-1, 4),
        out_state=np.array([[i] + list(v) for i, v in sorted(res['out_state'].items())], np.float32))


def load(seed):
    path = os.path.join(CACHE, f'sim_v{VERSION}_{seed}.npz')
    if not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True)
        res = run(seed, record=True)
        save(res, path)
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


if __name__ == '__main__':
    if sys.argv[1] == 'search':
        search(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    elif sys.argv[1] == 'run':
        seed = int(sys.argv[2])
        t0 = time.time()
        res = run(seed, record=True, verbose=True)
        print('score', score(res), f'{time.time() - t0:.1f}s')
        print('standings', standings(res)[:10])
        os.makedirs(CACHE, exist_ok=True)
        save(res, os.path.join(CACHE, f'sim_v{VERSION}_{seed}.npz'))
