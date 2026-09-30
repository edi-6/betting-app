"""What moves in the valley besides the rider: the firework rocket in the hand (and the swing that uses it), the
rocket's sparks streaming past, blubs hopping on the meadow, pollen and motes drifting in the low sun, the spray at the
foot of the ichor falls.

Everything here is a function of the animation time (sc['ta']), which the director fades back to the start over the
video's last moments so it loops.
"""
import numpy as np

import entities as EN
import flight as FL
import gfx
import mobs

_ST = {}


def register(r):
    r.streak_col = (1.0, 1.0, 1.0)


def _hash(*a):
    x = np.sin(np.dot(np.asarray(a, float), [12.9898, 78.233, 37.719][:len(a)]) * 43758.5453)
    return x - np.floor(x)


def setup(fl, meta):
    """Place the blubs: little groups beside the path on the meadow and the lake's shore."""
    rng = np.random.default_rng(5)
    H, org = meta['H'], meta['origin']
    zones = meta['zones']
    blubs = []
    picks = [zones['grove'][0] + 30.0, zones['grove'][0] + 70.0, zones['grove'][0] + 110.0,
             zones['lake'][0] + 20.0, zones['lake'][0] + 70.0, zones['towers'][0] + 40.0]
    for s in picks:
        P = fl.pos(s)
        T = fl.tangent(s)
        rh = np.array([T[1], -T[0]])
        rh /= max(np.linalg.norm(rh), 1e-6)
        side = 1.0 if rng.random() < 0.5 else -1.0
        for k in range(4):
            q = P[:2] + rh * side * rng.uniform(5.0, 11.0) + T[:2] * rng.uniform(-4.0, 10.0)
            i, j = int(np.floor(q[0])) - org[0], int(np.floor(q[1])) - org[1]
            if not (0 <= i < H.shape[0] and 0 <= j < H.shape[1]):
                continue
            z = float(H[i, j])
            if z < 44.5 or z > 50:
                continue                       # only on the meadow (not in the lake, not on a mesa)
            yaw = float(np.degrees(np.arctan2(T[0], T[1]))) + rng.uniform(-70, 70)
            blubs.append((np.array([q[0], q[1], z]), yaw, rng.uniform(0, 6.3)))
    _ST['blubs'] = blubs
    _ST['fl'] = fl
    _ST['meta'] = meta
    setup_show(fl, meta)


# ---------------------------------------------------------------------------------------------
# the firework in the hand
# ---------------------------------------------------------------------------------------------
def swing(t):
    """The use swing (0..1 over a quarter second) after each firework."""
    for tf in FL.FIREWORKS:
        d = t - tf
        if 0.0 <= d < 0.28:
            return d / 0.28
    return 0.0


def held_rocket(sc, names):
    """The rocket in the right hand, low in the bottom right corner (kept where it is on screen as the view widens,
    like the game's hand, which has a field of view of its own)."""
    t = sc['t']
    F, R, U = sc['axes']
    eye = sc['eye']
    k = np.tan(np.radians(39.0)) / np.tan(np.radians(sc['cam']['fov'] / 2))
    s = np.sin(np.pi * swing(t))
    kick = FL.fw_kick(t)
    sway = 0.012 * np.sin(t * 2 * np.pi * 6 / FL.DURATION)
    # the hand has a field of view of its own: only its distance changes with the view's, so it keeps its place;
    # using the rocket dips it down and back up (the game's use swing), it doesn't turn
    pos = eye + F * 0.60 * k * (1.0 - 0.06 * s - 0.08 * kick) + R * (0.205 - 0.03 * s + sway) - U * (
        0.315 + 0.16 * s + 0.04 * kick)
    q = EN.basis_quat(F, R, U)
    q = EN.qmul(q, EN.qmul(EN.qz(-0.62 - 0.08 * s), EN.qmul(EN.qx(0.25 + 0.12 * s), EN.qy(0.18))))
    lay = names.index('firework_rocket') if 'firework_rocket' in names else 0
    sz = 0.30
    return [('item_firework_rocket', [*pos, *q, sz, sz, sz, lay, 1, 1, 1, 0, gfx.MAT_HAND])]


def rocket_sparks(sc):
    """The boost: sparks and smoke from the rocket (just below and ahead of the eye while it burns), left behind in
    the air as the rider races on, twinkling and falling; a flash as it lights."""
    fl = sc['fl']
    t = sc['t']
    glow, soft, streaks, lights = [], [], [], []
    for tf in FL.FIREWORKS:
        age0 = t - tf
        if not (-0.05 < age0 < 2.2):
            continue
        # sparks spawned every 1/120 s over the burn (1.1 s), each living ~0.9 s
        for n in range(int(1.1 * 120)):
            ts = tf + n / 120.0
            a = t - ts
            if a < 0.0 or a > 0.9:
                continue
            _, e0, (F0, R0, U0), _ = FL.view(fl, ts)
            h1, h2, h3 = _hash(n, tf, 1.0), _hash(n, tf, 2.0), _hash(n, tf, 3.0)
            ang = h1 * 2 * np.pi
            src = e0 + F0 * 1.3 - U0 * 1.05 + (R0 * np.cos(ang) + U0 * np.sin(ang)) * 0.25 * h2
            vel = (R0 * np.cos(ang) + U0 * np.sin(ang)) * (1.0 + 2.0 * h3) - F0 * 3.0
            p = src + vel * a + np.array([0.0, 0.0, -2.5]) * a * a
            life = 1.0 - a / 0.9
            tw = 0.55 + 0.45 * np.sin(t * 60.0 + n * 1.7)
            col = (1.0, 0.95, 0.8) if h3 < 0.6 else (1.0, 0.55, 0.75) if h3 < 0.8 else (0.55, 0.95, 1.0)
            glow.append([*p, 0.035 + 0.025 * h2, col[0] * 9, col[1] * 9, col[2] * 9, life * tw])
            if a < 0.45:
                streaks.append([*p, *(p - vel * 0.05), 0.012, 0.9 * life, col[0] * 5, col[1] * 5, col[2] * 5])
            if n % 3 == 0 and a < 0.8:
                q = src - F0 * 0.2 + vel * 0.25 * a
                soft.append([*q, 0.5 + 1.5 * a, 0.75, 0.72, 0.74, 0.22 * life])
        if 0 <= age0 < 1.1:
            _, e, (F, R, U), _ = FL.view(fl, t)
            fl_ = np.exp(-age0 / 0.12) * 2.5 + 0.8 * (1.0 - age0 / 1.1)
            lights.append([*(e + F * 1.3 - U * 1.0), 9.0, 1.6 * fl_, 1.2 * fl_, 0.9 * fl_])
    return glow, soft, streaks, lights


# ---------------------------------------------------------------------------------------------
# the show: three rockets rising from the valley and bursting over it on the beats after the second boost
# ---------------------------------------------------------------------------------------------
BEAT = FL.BAR / 4
# (burst time, direction from the landing spot: azimuth, elevation (deg), distance, colour, fade colour): in the
# north-east sky over the valley, behind the spire as the rider swings round it and lands
SHOW = [
    (FL.FIREWORKS[1] + 3 * BEAT, 61.0, 8.0, 112.0, (1.0, 0.30, 0.62), (1.0, 0.80, 0.90)),
    (FL.FIREWORKS[1] + 4 * BEAT, 29.0, 5.0, 122.0, (0.30, 0.92, 1.0), (0.85, 1.0, 1.0)),
    (FL.FIREWORKS[1] + 5 * BEAT, 47.0, 3.0, 104.0, (1.0, 0.76, 0.22), (1.0, 0.96, 0.70)),
]
RISE = 0.85                  # each rocket climbs for this long before it bursts
LIFE = 1.7                   # and its stars burn this long


def _fib_sphere(n):
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    th = np.pi * (1 + 5 ** 0.5) * k
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def setup_show(fl, meta):
    """Where each burst goes, and the ground below it that its rocket is launched from."""
    H, org = meta['H'], meta['origin']
    spot = np.array(FL.CONTROL[-1], float)
    out = []
    for tb, az, el, dist, col, fade in SHOW:
        a, e = np.radians(az), np.radians(el)
        c = spot + np.array([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)]) * dist
        base = c[:2] + np.array([np.cos(a), np.sin(a)]) * 8.0
        i, j = int(np.floor(base[0])) - org[0], int(np.floor(base[1])) - org[1]
        gz = float(H[np.clip(i, 0, H.shape[0] - 1), np.clip(j, 0, H.shape[1] - 1)])
        out.append(dict(t=tb, c=c, base=np.array([base[0], base[1], gz]), col=np.array(col), fade=np.array(fade)))
    _ST['show'] = out
    _ST['dirs'] = _fib_sphere(340)


def show(t):
    """(glow, streaks, lights) of the firework show at time t."""
    glow, streaks = [], []
    dirs = _ST.get('dirs')
    for n, b in enumerate(_ST.get('show', [])):
        tb, c, base = b['t'], b['c'], b['base']
        # the rocket climbing, a sparkling trail behind it
        if tb - RISE <= t < tb:
            u = (t - (tb - RISE)) / RISE
            ease = 1 - (1 - u) ** 1.6
            head = base + (c - base) * ease
            glow.append([*head, 0.5, 12.0, 9.0, 6.0, 1.0])
            for k in range(28):
                uk = max(0.0, ease - k * 0.012)
                p = base + (c - base) * uk + np.array([np.sin(k * 7.3 + n), np.cos(k * 5.1 + n), 0.0]) * 0.15 * k
                p[2] -= 0.02 * k * k
                glow.append([*p, 0.28, 6.0, 4.2, 2.6, 0.9 * (1 - k / 28.0) * (0.6 + 0.4 * np.sin(t * 50 + k))])
            streaks.append([*head, *(base + (c - base) * max(0.0, ease - 0.06)), 0.25, 0.9, 5.0, 3.8, 2.4])
        a = t - tb
        if 0.0 <= a < LIFE:
            # the flash, and the stars flying out, slowing, falling, twinkling out, their colour fading
            if a < 0.25:
                fl_ = np.exp(-a / 0.06)
                glow.append([*c, 16.0 * (0.5 + a * 3), *(b['col'] * 10 * fl_ + 4 * fl_), 1.0])
            k_ = 1.7
            v0 = 33.0 * (0.85 + 0.3 * ((np.sin(np.arange(len(dirs)) * 12.9898 + n) * 43758.5453) % 1.0))
            reach = (1 - np.exp(-k_ * a)) / k_
            P = c[None] + dirs * (v0 * reach)[:, None]
            P[:, 2] -= 3.2 * a * a
            prev = c[None] + dirs * (v0 * (1 - np.exp(-k_ * max(a - 0.07, 0.0))) / k_)[:, None]
            prev[:, 2] -= 3.2 * max(a - 0.07, 0.0) ** 2
            life = np.clip(1.0 - a / LIFE, 0, 1)
            fade = np.clip((a - 0.5) / 0.9, 0, 1)
            col = np.broadcast_to(b['col'] * (1 - fade) + b['fade'] * fade, (len(dirs), 3))
            hot = np.exp(-a / 0.12)
            tw = 1.0 if a < 0.9 else (0.5 + 0.5 * np.sign(np.sin(a * 40 + np.arange(len(dirs)) * 2.3)))
            inten = (12.0 + 16.0 * hot) * life * tw
            for p, q, cc, it in zip(P, prev, col, np.broadcast_to(inten, (len(dirs),))):
                glow.append([*p, 1.15, *(cc * it + hot * 6.0), 1.0])
                if a < 1.1:
                    streaks.append([*p, *q, 0.40, 0.8 * life, *(cc * it * 0.6)])
    return glow, streaks


# ---------------------------------------------------------------------------------------------
# the valley's life
# ---------------------------------------------------------------------------------------------
def motes(eye, ta, cell, radius, z_lo, z_hi, col, size, alpha, seed, rise=1.5, period=(4.0, 8.0)):
    """Specks drifting in the air round the eye: a fixed lattice of emitters (so each stays put in the world from
    frame to frame), each floating, swaying and fading on its own cycle."""
    gx0, gy0 = np.floor((eye[0] - radius) / cell), np.floor((eye[1] - radius) / cell)
    n = int(2 * radius / cell)
    ii, jj = np.meshgrid(np.arange(n) + gx0, np.arange(n) + gy0, indexing='ij')
    ii, jj = ii.ravel(), jj.ravel()
    h1 = (np.sin(ii * 12.9898 + jj * 78.233 + seed) * 43758.5453) % 1.0
    h2 = (np.sin(ii * 39.346 + jj * 11.135 + seed) * 24634.6345) % 1.0
    h3 = (np.sin(ii * 73.156 + jj * 52.235 + seed) * 12345.6789) % 1.0
    per = period[0] + (period[1] - period[0]) * h3
    age = ((ta + h1 * 17.0) % per) / per
    x = (ii + h1) * cell + 0.8 * np.sin(ta * 0.9 + h2 * 6.0)
    y = (jj + h2) * cell + 0.8 * np.cos(ta * 0.7 + h1 * 6.0)
    z = z_lo + (z_hi - z_lo) * h2 + rise * age
    a = alpha * np.sin(np.pi * age) * (0.75 + 0.25 * np.sin(ta * 7.0 + h3 * 20.0))
    dist = np.sqrt((x - eye[0]) ** 2 + (y - eye[1]) ** 2 + (z - eye[2]) ** 2)
    a = a * np.clip((dist - 1.5) / 4.0, 0, 1)            # none right in front of the eye
    s = size * (0.7 + 0.6 * h3)
    return np.column_stack([x, y, z, s, np.full_like(x, col[0]), np.full_like(x, col[1]), np.full_like(x, col[2]),
                            a]).astype(np.float32)


def falls_spray(meta, ta, eye):
    """Mist and glowing drops at the foot of each ichor fall."""
    soft, glow = [], []
    for k, (fx_, fy) in enumerate(meta['falls']):
        if np.hypot(eye[0] - fx_, eye[1] - fy) < 150.0:
            a, b = _spray(fx_, fy, ta + k * 3.7)
            soft.append(a)
            glow.append(b)
    if not soft:
        return np.zeros((0, 8), np.float32), np.zeros((0, 8), np.float32)
    return np.concatenate(soft), np.concatenate(glow)


def _spray(fx_, fy, ta):
    k = np.arange(90)
    h1, h2, h3 = (np.sin(k * 12.9898) * 43758.5453) % 1, (np.sin(k * 78.233) * 43758.5453) % 1, \
        (np.sin(k * 37.719) * 43758.5453) % 1
    age = (ta * 0.5 + h1) % 1.0
    x = fx_ - 3.0 - 7.0 * age * h2 + 2.0 * np.sin(h3 * 6.3)
    y = fy - 4.0 + 8.0 * h3
    z = 44.0 + 5.0 * age * (1.2 - age)
    soft = np.column_stack([x, y, z, 2.5 + 5.0 * age, np.full_like(x, 1.0), np.full_like(x, 0.72),
                            np.full_like(x, 0.62), 0.16 * np.sin(np.pi * age)]).astype(np.float32)
    k2 = np.arange(60)
    g1, g2, g3 = (np.sin(k2 * 3.17) * 43758.5453) % 1, (np.sin(k2 * 7.41) * 43758.5453) % 1, \
        (np.sin(k2 * 5.03) * 43758.5453) % 1
    a2 = (ta * 1.3 + g1) % 1.0
    gx = fx_ - 2.5 - 5.0 * a2 * g2
    gy = fy - 4.0 + 8.0 * g3
    gz = 44.3 + 3.5 * a2 - 4.5 * a2 * a2
    glow = np.column_stack([gx, gy, gz, np.full_like(gx, 0.12), np.full_like(gx, 3.0), np.full_like(gx, 1.3),
                            np.full_like(gx, 0.9), 0.8 * np.sin(np.pi * a2)]).astype(np.float32)
    return soft, glow


def events(sc):
    """(rows, lights, particles, streaks) for the frame."""
    r = sc['r']
    meta = sc['meta']
    ta = sc['ta']
    eye = sc['eye']
    rows = held_rocket(sc, getattr(r, '_item_names', []))
    for (p, yaw, ph) in _ST.get('blubs', []):
        if np.linalg.norm(p - eye) < 110.0:
            rows += mobs.blub_rows(p, yaw, ta, ph)
    glow, soft, streaks, lights = rocket_sparks(sc)
    g2, s2 = show(ta)
    glow += g2
    streaks += s2
    # pollen lit gold by the low sun, and a few pale-blue motes (the Sift's souls) over the meadow
    pollen = motes(eye, ta, 3.0, 24.0, eye[2] - 8.0, eye[2] + 8.0, (2.4, 1.7, 0.9), 0.05, 0.9, 1.3)
    souls = motes(eye, ta, 7.0, 56.0, 44.5, 58.0, (0.9, 2.2, 2.6), 0.08, 0.6, 4.1, rise=4.0)
    sp_soft, sp_glow = falls_spray(meta, ta, eye)
    parts = dict(soft=np.concatenate([np.array(soft, np.float32).reshape(-1, 8), sp_soft]),
                 glow=np.concatenate([np.array(glow, np.float32).reshape(-1, 8), pollen, souls, sp_glow]))
    st = np.array(streaks, np.float32).reshape(-1, 11) if streaks else None
    return rows, lights, parts, st
