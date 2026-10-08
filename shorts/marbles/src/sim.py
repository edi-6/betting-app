"""The marbles' physics: pymunk (Chipmunk2D) in the plane of the slot (see machine.py).

The hopper isn't simulated, its outflow is scripted: rows of LANES marbles come down out of the chute's throat at
the rate the story wants (a ramp, a steady flow, nothing while the golden ball plugs the hole, a burst when it pops
out, a faster flow, a trickle at the end and one last marble on its own). Everything below the throat is physics:
the marbles bounce off the end rods, pile up in the tank and settle. Fired from the middle, they still spread
evenly: they're bouncy, so the pile levels itself.

When the goat rams the pedestal, every marble in the tank gets a little kick (the picture shivers).

python sim.py   builds cache/sim_*.npy|npz (a few minutes); load() returns them.
"""
import os
import time

import numpy as np
import pymunk

import machine as M

CACHE = os.environ.get('MARBLES_CACHE', os.path.join(os.path.dirname(__file__), '..', 'cache'))
G = 20.0                           # blocks / s^2
DT = 1.0 / 240.0
REC = 2                            # record every REC steps (120 Hz)
COUNT_Z = M.FRAME_TOP - 1.0        # a marble counts once it's down in the tank

# the script of the flow (sim seconds)
T_JAM = 11.0                       # the golden ball drops into the hole: the flow stops
JAM_LEN = 5.0                      # until the goat's ram pops it out
T_SAVE = T_JAM + JAM_LEN
RATE0, RATE1 = 330.0, 400.0        # marbles a second before the jam (ramping up)
BURST, BURST_LEN = 700.0, 0.5      # the backed-up marbles rushing out
RATE2 = 600.0                      # after: the hole is wider now the ball's gone
TAPER = 1.0                        # the flow dies away over this long at the end
PEG_E = 0.62                       # how bouncy the end rods are
HERO_TIME = 4.5                    # time allowed for the last marble's drop
VERSION = 7


def rate(t):
    """Marbles per second coming out of the throat at time t (before the end's taper)."""
    if t < 0.0:
        return 0.0
    if t < T_JAM:
        ramp = min(1.0, t / 0.8)
        return ramp * (RATE0 + (RATE1 - RATE0) * t / T_JAM)
    if t < T_SAVE:
        return 0.0
    if t < T_SAVE + BURST_LEN:
        return BURST
    return RATE2


def spawn_times():
    """When each of the first N - 1 marbles comes out, from the rate (and the taper at the end). The last marble is
    dropped on its own once they've all settled (hero())."""
    ts = []
    t, acc = 0.0, 0.0
    dt = 1e-4
    n_main = M.N - 1
    # the taper: the last TAPER seconds' worth of marbles come out at a falling rate
    while len(ts) < n_main:
        r = rate(t)
        acc += r * dt
        while acc >= 1.0 and len(ts) < n_main:
            acc -= 1.0
            ts.append(t)
        t += dt
    ts = np.array(ts)
    # stretch the last part so the flow dies away instead of stopping dead
    t_end = ts[-1]
    k = ts > t_end - TAPER * 0.5
    u = (ts[k] - (t_end - TAPER * 0.5)) / (TAPER * 0.5)            # 0..1
    ts[k] = (t_end - TAPER * 0.5) + TAPER * (1.0 - np.sqrt(1.0 - u)) * 1.0
    return np.sort(ts)


def _space(pegs=True):
    space = pymunk.Space()
    space.gravity = (0.0, -G)
    space.iterations = 8
    space.use_spatial_hash(0.6, 30000)
    space.collision_slop = 0.01                    # (the default 0.1 lets a marble sit in a notch on a peg)
    st = space.static_body

    def wall(a, b, e=0.4, f=0.4):
        s_ = pymunk.Segment(st, a, b, 0.05)
        s_.elasticity, s_.friction = e, f
        space.add(s_)
    H = M.HALF
    wall((-H, 0.0), (H, 0.0))
    wall((-H, 0.0), (-H, M.BOARD_TOP))
    wall((H, 0.0), (H, M.BOARD_TOP))
    wall((-H, M.BOARD_TOP), (-M.THROAT_HALF - 0.1, M.BOARD_TOP))
    wall((M.THROAT_HALF + 0.1, M.BOARD_TOP), (H, M.BOARD_TOP))
    pegs = []
    for (x, z) in M.pegs():
        c = pymunk.Circle(st, M.PEG_R, (x, z))
        c.elasticity, c.friction = PEG_E, 0.1
        space.add(c)
        pegs.append(c)
    return space


def _marble(x, z, vx, vz):
    b = pymunk.Body(1.0, pymunk.moment_for_circle(1.0, 0.0, M.R))
    b.position = (float(x), float(z))
    b.velocity = (float(vx), float(vz))
    c = pymunk.Circle(b, M.R)
    c.elasticity, c.friction = 0.5, 0.3
    return b, c


def build(seed=7, verbose=True):
    rng = np.random.default_rng(seed)
    space = _space()
    st_t = spawn_times()
    n = len(st_t)                                  # N - 1: the last one comes later, on its own
    lanes = (np.arange(M.LANES) - (M.LANES - 1) / 2) * (2 * M.THROAT_HALF / M.LANES)
    bodies = []
    t = 0.0
    nxt = 0
    t_max = st_t[-1] + 7.0
    n_rec = int(np.ceil((t_max + HERO_TIME + 2.0) / (DT * REC))) + 4
    pos = np.full((n_rec, M.N, 2), np.nan, np.float32)
    rec_t = np.zeros(n_rec)
    count_t = np.full(M.N, np.inf)
    kicked = False
    step = 0
    rec = 0
    quiet = 0
    clock = time.time()
    while t < t_max:
        # bring out the marbles whose time has come, in rows across the throat, already moving down at the speed
        # that keeps the rows apart
        while nxt < n and st_t[nxt] <= t + DT:
            v = 0.54 * max(rate(st_t[nxt]), 60.0) / M.LANES
            age = t + DT - st_t[nxt]
            row = nxt // M.LANES
            off = (row % 2) * (M.THROAT_HALF / M.LANES) - (M.THROAT_HALF / M.LANES) * 0.5
            b, c = _marble(lanes[nxt % M.LANES] + off + rng.normal(0.0, 0.05), M.THROAT_Z - v * age,
                           rng.normal(0.0, 0.8), -v * rng.uniform(0.94, 1.0))
            space.add(b, c)
            bodies.append(b)
            nxt += 1
        if not kicked and t >= T_SAVE:
            # the ram: the whole machine jolts, the pile in the tank jumps a little
            kicked = True
            for b in bodies:
                if b.position.y < COUNT_Z:
                    b.velocity = (b.velocity.x + float(rng.normal(0.0, 0.35)),
                                  b.velocity.y + float(rng.uniform(0.6, 1.8)))
        if step % 8 == 0:
            # a ball can't balance on top of a rod: nudge any marble that's come to rest up among the pegs
            for b in bodies:
                p_, v_ = b.position, b.velocity
                if p_.y > M.PEG_BOTTOM - 1.0 and abs(v_.x) + abs(v_.y) < 0.6:
                    b.velocity = (v_.x + float(rng.choice((-0.5, 0.5))), v_.y)
        space.step(DT)
        t += DT
        step += 1
        if step % REC == 0:
            if bodies:
                p = np.array([b.position for b in bodies], np.float32)
                pos[rec, :len(bodies)] = p
                fresh = (p[:, 1] < COUNT_Z) & ~np.isfinite(count_t[:len(bodies)])
                count_t[:len(bodies)][fresh] = t
                if nxt == n:
                    sp = max(abs(b.velocity.x) + abs(b.velocity.y) for b in bodies)
                    quiet = quiet + 1 if sp < 0.08 else 0
            rec_t[rec] = t
            rec += 1
            if verbose and rec % 600 == 0:
                print(f'  sim t={t:5.1f}s  marbles {len(bodies):5d}  counted {np.isfinite(count_t).sum():5d}  '
                      f'({time.time() - clock:.0f}s)', flush=True)
            if quiet >= 30:                        # everything has been still for a quarter of a second
                break
    t_settled = t
    pile = pos[rec - 1, :n].copy()
    # the last marble, on its own
    t_hero = t_settled + 0.25
    hero = find_hero(pile, verbose)
    k_hero = len(hero['path'])
    while rec_t[rec - 1] < t_hero - 1e-9:
        pos[rec, :n] = pile
        rec_t[rec] = rec_t[rec - 1] + DT * REC
        rec += 1
    for k in range(k_hero):
        pos[rec, :n] = pile
        pos[rec, n] = hero['path'][k]
        rec_t[rec] = rec_t[rec - 1] + DT * REC
        if hero['path'][k][1] < COUNT_Z and not np.isfinite(count_t[n]):
            count_t[n] = rec_t[rec]
        rec += 1
    for _ in range(60):
        pos[rec, :n] = pile
        pos[rec, n] = hero['path'][-1]
        rec_t[rec] = rec_t[rec - 1] + DT * REC
        rec += 1
    pos = pos[:rec]
    rec_t = rec_t[:rec]
    spawn = np.append(st_t, t_hero)
    final = pos[-1].copy()
    return {'pos': pos, 't': rec_t, 'spawn_t': spawn, 'count_t': count_t, 'final': final,
            'hero_hits': np.array(hero['hits']), 'hero_land': hero['land'] + t_hero,
            'events': np.array([T_JAM, T_SAVE, st_t[-1], t_settled, t_hero, hero['land'] + t_hero])}


def _hero_try(space, x0, vx, vz, max_t=4.0):
    """Drop one marble through the pegs onto the settled pile: its path at REC rate, the peg hits, when it lands."""
    b, c = _marble(x0, M.THROAT_Z, vx, vz)
    space.add(b, c)
    path, hits = [], []
    t, step, rest = 0.0, 0, 0
    prev_v = (vx, vz)
    land = None
    pg = M.pegs()
    while t < max_t:
        space.step(DT)
        t += DT
        step += 1
        p, v = b.position, b.velocity
        dv = abs(v.x - prev_v[0]) + abs(v.y - (prev_v[1] - G * DT))
        if dv > 1.0 and p.y > M.PEG_BOTTOM - 0.5:
            d = np.hypot(pg[:, 0] - p.x, pg[:, 1] - p.y).min()
            if d < M.R + M.PEG_R + 0.05:
                hits.append(t)
        prev_v = (v.x, v.y)
        if step % REC == 0:
            path.append((p.x, p.y))
        if p.y < COUNT_Z and abs(v.x) + abs(v.y) < 0.05:
            rest += 1
            if land is None:
                land = t
            if rest > 30:
                break
        else:
            rest = 0
            land = None if p.y >= COUNT_Z else land
    space.remove(b, c)
    return {'path': np.array(path, np.float32), 'hits': hits, 'land': land, 'final': (b.position.x, b.position.y)}


def find_hero(pile, verbose=True):
    """Try drops for the last marble and keep a good one: a couple of seconds through the pegs with plenty of
    bounces, landing near the middle where you can see it, in front of nothing."""
    space = _space()
    st = space.static_body
    for (x, z) in pile:
        c = pymunk.Circle(st, M.R, (float(x), float(z)))
        c.elasticity, c.friction = 0.5, 0.3
        space.add(c)
    best, best_score = None, -1e9
    rng = np.random.default_rng(11)
    for k in range(260):
        x0 = rng.uniform(-1.2, 1.2)
        vx = rng.uniform(-2.5, 2.5)
        vz = -rng.uniform(3.0, 9.0)
        r = _hero_try(space, x0, vx, vz)
        if r['land'] is None:
            continue
        fx, fz = r['final']
        score = min(len(r['hits']), 9) * 1.0 - abs(r['land'] - 2.2) * 3.0 - max(abs(fx) - 5.0, 0.0) * 1.5
        if fz > M.FRAME_TOP - 1.0:
            continue
        if score > best_score:
            best, best_score = r, score
    if verbose:
        print(f'  hero: {len(best["hits"])} peg hits, lands after {best["land"]:.2f}s at '
              f'({best["final"][0]:.1f}, {best["final"][1]:.1f})', flush=True)
    return best


def paths():
    os.makedirs(CACHE, exist_ok=True)
    base = os.path.join(CACHE, f'sim_v{VERSION}')
    return base + '_pos.npy', base + '_meta.npz'


def load(mmap=True):
    p_pos, p_meta = paths()
    if not (os.path.exists(p_pos) and os.path.exists(p_meta)):
        t = time.time()
        d = build()
        np.save(p_pos, d.pop('pos'))
        np.savez(p_meta, **d)
        print(f'[sim] built in {time.time() - t:.0f}s', flush=True)
    meta = np.load(p_meta)
    out = {k: meta[k] for k in meta.files}
    out['pos'] = np.load(p_pos, mmap_mode='r' if mmap else None)
    return out


if __name__ == '__main__':
    d = load()
    f = d['final']
    print('final: top z %.2f (99%% %.2f); events %s; counted %d' % (
        f[:, 1].max(), np.percentile(f[:, 1], 99), np.round(d['events'], 2), np.isfinite(d['count_t']).sum()))
