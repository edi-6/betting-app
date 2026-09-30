"""Effects and events: particles (spray, embers, sparks, smoke, fire), the transitions between dimensions, and the
per-world events that the director adds to each frame.

Particles go to the renderer as dict(soft=(P, 8), glow=(P, 8)): pos3 size1 rgba4 (glow is additive light).
"""
import numpy as np


def register(r):
    import mobs
    mobs.register(r)


def _rng(seed):
    return np.random.default_rng(seed & 0x7FFFFFFF)


def merge_particles(*ps):
    out = {'soft': [], 'glow': []}
    for p in ps:
        if not p:
            continue
        for k in ('soft', 'glow'):
            if k in p and p[k] is not None and len(p[k]):
                out[k].append(np.asarray(p[k], np.float32))
    return {k: (np.concatenate(v) if v else np.zeros((0, 8), np.float32)) for k, v in out.items()}


# ---------------------------------------------------------------------------------------------
# the overworld: the waterfall's spray
# ---------------------------------------------------------------------------------------------
def waterfall_spray(meta, tau, eye):
    if 'waterfall' not in meta:
        return None, 0.0
    Pc, rt, th = meta['waterfall']
    Pc = np.asarray(Pc, float)
    rng = _rng(int(tau * 60) + 17)
    n = 260
    # mist where the curtain hits the river, and a haze drifting up the curtain
    a = rng.uniform(-5.5, 5.5, n)
    h = np.abs(rng.normal(0, 1.0, n)) ** 1.5 * 6.0
    b = rng.normal(-0.6, 1.2, n)
    base = np.array([Pc[0], Pc[1], 19.9])
    pos = base[None] + np.asarray(rt)[None] * a[:, None] + np.asarray(th)[None] * b[:, None]
    pos[:, 2] += h
    size = rng.uniform(0.5, 1.6, n) * (1 + h / 8)
    alpha = rng.uniform(0.10, 0.28, n) * np.clip(1 - h / 10, 0.15, 1)
    soft = np.column_stack([pos, size, np.full(n, 0.92), np.full(n, 0.95), np.full(n, 1.0), alpha])
    # the curtain's plane: how close the eye is to it (drives the splash overlay)
    d = np.dot(np.asarray(eye) - Pc, th)
    lat = abs(np.dot(np.asarray(eye) - Pc, rt))
    wet = float(np.clip(1.0 - abs(d - 0.25) / 1.2, 0, 1)) if lat < 6.0 and eye[2] < 47 else 0.0
    return dict(soft=soft), wet


def over_events(sc):
    meta = RD_META.get('over', {})
    parts, wet = waterfall_spray(meta, sc['tau'], sc['eye'])
    if wet > 0:
        sc.setdefault('overlays', []).append(('wet', wet))
    return [], [], parts


G_DEBRIS = np.array([0.0, 0.0, -32.0])     # the game's gravity for falling things, in blocks/s^2
LAVA_Z = 31.0


class NetherStory:
    """The ghasts, their fireballs, the span that falls, the landing (all in the Nether track's ride time).

    The first ghast floats right of the track ahead of the hump: as the cart comes round the bend it turns, opens its
    eyes and mouth and fires; the fireball flies on ahead of the cart and blows the span up just before the cart
    gets there, and the span collapses into the lava from the blast outwards. The second ghast waits beyond the gap
    and fires at the cart as it leaves the rails: the fireball comes straight at the camera and passes a couple of
    blocks over the rider's right shoulder in the middle of the (slow-motion) jump."""

    def __init__(self, tr, meta):
        import voxel as VX
        import world_nether
        self.tr = tr
        g0, g1 = tr.gaps[0]
        self.g0, self.g1 = g0, g1
        self.s_imp = g0 + 0.36 * (g1 - g0)
        self.P_imp = tr.pos(self.s_imp) + np.array([0.0, 0.0, -0.4])
        spots = meta.get('ghasts') or world_nether.ghast_spots(tr)
        self.ghast, self.ghast2 = (np.asarray(g, float) for g in spots)
        self.t_take = tr.time(g0)
        self.t_land = tr.time(g1)
        # the first shot: when the ghast is ~36 blocks ahead of the cart
        self.t_shoot = tr.time(g0 - 71.0)
        self.fb1_speed = 48.0
        # the second: fired as the cart leaves the rails, passing just over its right shoulder mid-jump
        self.t_mid = 0.5 * (self.t_take + self.t_land)
        s_mid = tr.s_at(self.t_mid)
        T, R, U = tr.frame(s_mid)
        self.fb2_target = tr.pos(s_mid) + R * 2.3 + U * 2.9
        self.t_shoot2 = self.t_take - 0.32
        self._setup_shots(tr)
        # the pieces of the span: rails, bed blocks and the pillars under it (outside the gap they stay put)
        at = VX.atlas()
        rng = _rng(77)
        pieces = []
        L_rail, L_bed = at['rail'], at['nether_bricks']
        for k in range(int(np.floor(g0 - 6)), int(np.ceil(g1 + 2))):
            ss = k + 0.5
            if not (g0 <= ss <= g1 + 1.0):
                continue
            P = tr.pos(ss)
            T, R, U = tr.frame(ss)
            q = EN_basis(T, R, U)
            for kind, pos, lay, scl in (('rail_tile', P + U * 0.03, L_rail, (1.0, 1.04, 1.0)),
                                        ('prop_cube', P - U * 0.5, L_bed, (1.0, 1.03, 1.0))):
                pieces.append(dict(kind=kind, p0=pos, q0=q, lay=lay, scl=scl, s=ss))
        for (x, y, zt, ss) in meta.get('gap_supports', []):
            fixed = not (g0 - 0.5 <= ss <= g1 + 0.5)
            for z in np.arange(LAVA_Z + 1, zt + 1, 1.0):
                pieces.append(dict(kind='prop_cube', p0=np.array([x + 0.5, y + 0.5, z + 0.5]),
                                   q0=np.array([0, 0, 0, 1.0]), lay=L_bed, scl=(1.0, 1.0, 1.0), s=ss, support=True,
                                   fixed=fixed))
        for pc in pieces:
            pc['hot'] = False
            if pc.get('fixed'):
                pc['t0'] = pc['t_sink'] = 1e9
                continue
            d = np.linalg.norm(pc['p0'] - self.P_imp)
            pc['hot'] = d < 10.0
            # the collapse runs out from the blast; the pillars go a moment after the deck they held
            pc['t0'] = self.t_hit + abs(pc['s'] - self.s_imp) / 64.0 + (0.10 if pc.get('support') else 0.0)
            if d < 10.0:
                dirn = (pc['p0'] - self.P_imp) / max(d, 0.5)
                pc['v0'] = dirn * 26.0 * (1.15 - d / 11.0) + np.array([0, 0, 6.0]) + rng.normal(0, 2.5, 3)
            else:
                # the rest breaks up and drops: a kick down and a little sideways
                pc['v0'] = rng.normal(0, 1.6, 3) * np.array([1.0, 1.0, 0.5]) + np.array([0, 0, -3.5])
            ax = rng.normal(0, 1, 3)
            pc['axis'] = ax / np.linalg.norm(ax)
            pc['w'] = rng.uniform(1.0, 7.0) * (1.6 if d < 10 else 0.7)
            # when it reaches the lava
            a, b, c = 0.5 * G_DEBRIS[2], pc['v0'][2], pc['p0'][2] - LAVA_Z
            disc = b * b - 4 * a * c
            dt = (-b - np.sqrt(disc)) / (2 * a) if disc > 0 else 3.0
            pc['t_sink'] = pc['t0'] + dt
            pc['sink_pos'] = pc['p0'] + pc['v0'] * dt + 0.5 * G_DEBRIS * dt * dt
        for k, pc in enumerate(pieces):
            pc['seed'] = 1000 + k
        self.pieces = pieces

    def ghast_pos(self, which, tau):
        g, ph = (self.ghast, 0.0) if which == 1 else (self.ghast2, 1.7)
        return g + np.array([1.6 * np.sin(tau * 0.4 + ph), 1.2 * np.cos(tau * 0.3 + ph), 0.5 * np.sin(tau * 0.7 + ph)])

    def ghast_face(self, which, tau, eye):
        """Where a ghast faces: the cart, except that the first turns (part way) to the span as it fires."""
        gp = self.ghast_pos(which, tau)
        d_eye = eye - gp
        d_eye /= np.linalg.norm(d_eye)
        if which == 2:
            return d_eye
        d_aim = self.P_imp - gp
        d_aim /= np.linalg.norm(d_aim)
        t = self.t_shoot
        a = float(np.clip((tau - (t - 0.6)) / 0.45, 0, 1) * np.clip(((t + 0.8) - tau) / 0.5, 0, 1))
        a = a * a * (3 - 2 * a) * 0.65
        f = d_eye * (1 - a) + d_aim * a
        return f / np.linalg.norm(f)

    def fireball(self, which, tau):
        """(position, direction, alive) of a fireball."""
        if which == 1:
            t0, frm, to, spd, life = self.t_shoot, self.fb1_from, self.P_imp, self.fb1_speed, self.t_hit - self.t_shoot
        else:
            t0, frm, to, spd, life = self.t_shoot2, self.fb2_from, self.fb2_target, self.fb2_speed, 2.4
        d = np.asarray(to) - np.asarray(frm)
        d /= np.linalg.norm(d)
        alive = t0 <= tau <= t0 + life
        return np.asarray(frm) + d * spd * (tau - t0), d, alive

    def _setup_shots(self, tr):
        eye0 = tr.pos(tr.s_at(self.t_shoot)) + np.array([0.0, 0.0, 1.3])
        gp = self.ghast_pos(1, self.t_shoot)
        self.fb1_from = gp + self.ghast_face(1, self.t_shoot, eye0) * 2.2 + np.array([0.0, 0.0, -0.5])
        self.t_hit = self.t_shoot + np.linalg.norm(self.P_imp - self.fb1_from) / self.fb1_speed
        eye2 = tr.pos(tr.s_at(self.t_shoot2)) + np.array([0.0, 0.0, 1.3])
        gp2 = self.ghast_pos(2, self.t_shoot2)
        self.fb2_from = gp2 + self.ghast_face(2, self.t_shoot2, eye2) * 2.2 + np.array([0.0, 0.0, -0.5])
        self.fb2_speed = np.linalg.norm(self.fb2_target - self.fb2_from) / (self.t_mid - self.t_shoot2)

    def rows_lights_parts(self, sc):
        import mobs
        tau = sc['tau']
        rows, lights = [], []
        soft, glow, streaks = [], [], []
        eye = sc['eye']
        # the ghasts: drifting, facing the cart, the shooting face (eyes and mouth open) around each shot
        for which, t_sh in ((1, self.t_shoot), (2, self.t_shoot2)):
            gp = self.ghast_pos(which, tau)
            shooting = t_sh - 0.5 <= tau <= t_sh + 0.3
            rows += mobs.ghast_rows(gp, self.ghast_face(which, tau, eye), tau + which * 1.3, shooting=shooting)
        # fireballs: a glowing charge with a trail of fire and smoke
        for which in (1, 2):
            pos, dirn, alive = self.fireball(which, tau)
            if not alive:
                continue
            glow.append([*pos, 2.1, 1.0, 0.55, 0.15, 0.9])
            glow.append([*pos, 1.0, 1.0, 0.9, 0.6, 1.0])
            rng = _rng(int(tau * 240) + 5 + which)
            for k in range(16):
                back = pos - dirn * (0.4 + k * 0.5) + rng.normal(0, 0.25, 3)
                a = 1.0 - k / 16.0
                glow.append([*back, 0.9 + 0.08 * k, 1.0, 0.45 + 0.2 * a, 0.1, 0.55 * a])
                if k % 2:
                    soft.append([*(back + np.array([0, 0, 0.3])), 0.8 + 0.1 * k, 0.12, 0.10, 0.09, 0.35 * a])
            lights.append([*pos, 16.0, 3.4, 1.8, 0.55])
        # the explosion
        te = tau - self.t_hit
        if 0 <= te < 3.0:
            rng = _rng(911)
            n = 90
            dirs = rng.normal(0, 1, (n, 3))
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            spd = rng.uniform(4.0, 14.0, n)
            if te < 0.7:
                pos = self.P_imp[None] + dirs * (spd * 1.25 * te * (1.0 - te * 0.6))[:, None]
                heat = max(0.0, 1.0 - te / 0.7) ** 1.3
                for k in range(n):
                    glow.append([*pos[k], 3.5 + 6.0 * te, 1.0, 0.45 + 0.45 * heat, 0.12 * heat, 0.85 * heat])
                glow.append([*self.P_imp, 8.0 + 10.0 * te, 1.0, 0.8, 0.45, 0.9 * heat])
            sm = rng.normal(0, 1, (70, 3))
            for k in range(70):
                p = self.P_imp + sm[k] * (2.5 + 4.0 * min(te, 1.5)) + np.array([0, 0, 2.0 + 7.0 * te])
                a = 0.38 * min(1.0, te / 0.25) * max(0.0, 1.0 - te / 3.0)
                soft.append([*p, 4.0 + 4.0 * te, 0.07, 0.05, 0.045, a])
            fl = max(0.0, 1.0 - te / 0.8)
            lights.append([*self.P_imp, 55.0, 14.0 * fl, 7.0 * fl, 2.0 * fl])
            if te < 0.12:
                sc.setdefault('overlays', []).append(('flash', 1.0 - te / 0.12))
        # the span: intact until its piece is released, then falling, turning, gone into the lava with a splash
        for pc in self.pieces:
            if tau < pc['t0']:
                p, q = pc['p0'], pc['q0']
            else:
                dt = tau - pc['t0']
                if tau >= pc['t_sink']:
                    ds = tau - pc['t_sink']
                    if ds < 0.7:
                        # a splash of lava droplets thrown up where it went in
                        sp = pc['sink_pos']
                        rng = _rng(pc['seed'])
                        n = 7 if pc['kind'] == 'prop_cube' else 3
                        v = rng.normal(0, 1.0, (n, 3)) * np.array([1.8, 1.8, 0.0])
                        v[:, 2] = rng.uniform(4.0, 10.0, n)
                        pp = sp[None] + v * ds + 0.5 * G_DEBRIS[None] * ds * ds
                        a = 1.0 - ds / 0.7
                        for k in range(n):
                            if pp[k, 2] > LAVA_Z - 0.2:
                                glow.append([*pp[k], 0.22 + 0.1 * (k % 3), 1.0, 0.55, 0.12, 0.9 * a])
                    continue
                p = pc['p0'] + pc['v0'] * dt + 0.5 * G_DEBRIS * dt * dt
                q = EN_qmul(EN_qaxis(pc['axis'], pc['w'] * dt), pc['q0'])
            hot = 0.0
            if pc['hot'] and tau >= pc['t0']:
                hot = (0.9 if pc['kind'] == 'prop_cube' else 0.3) * max(0.0, 1.0 - (tau - pc['t0']) / 1.4)
            rows.append((pc['kind'], [*p, *q, *pc['scl'], pc['lay'], 1, 1, 1, hot,
                                      1 if pc['kind'] == 'prop_cube' else 5]))
        # landing: sparks off the wheels, grinding for a moment. They leave the wheels at nearly the cart's speed
        # (so from the seat they hang at the bottom of the view, fanning out), then drop away behind.
        tl = tau - self.t_land
        if 0 <= tl < 0.75:
            rng = _rng(4242)
            n = 140
            te = self.t_land + rng.uniform(0.0, 0.42, n) ** 1.5
            side = np.where(np.arange(n) % 2, 1.0, -1.0)
            keep = rng.uniform(1.12, 1.42, n)
            spr = rng.uniform(1.5, 6.5, n)
            upv = rng.uniform(3.0, 8.5, n)
            v_cam = self.tr.frame(self.tr.s_at(tau))[0] * self.tr.speed(self.tr.s_at(tau))
            for k in range(n):
                age = tau - te[k]
                if age < 0 or age > 0.34:
                    continue
                s_e = self.tr.s_at(te[k])
                P = self.tr.pos(s_e)
                T, R, U = self.tr.frame(s_e)
                v = T * self.tr.speed(s_e) * keep[k] + R * side[k] * spr[k] + U * upv[k]
                v = v + np.array([0, 0, -32.0]) * age
                p = P + T * 0.45 + R * 0.46 * side[k] + (v - np.array([0, 0, -32.0]) * age) * age + \
                    np.array([0, 0, -16.0]) * age * age
                a = 1.0 - age / 0.34
                # a streak along its motion as seen from the moving seat (the camera's shutter)
                tail = p - (v - v_cam) * 0.03
                streaks.append([*p, *tail, 0.028, 0.9 * a])
                if k % 4 == 0:
                    glow.append([*p, 0.06, 1.0, 0.8, 0.4, 0.5 * a])
            if tl < 0.4:
                P = self.tr.pos(self.tr.s_at(tau))
                f = 1.0 - tl / 0.4
                lights.append([*P, 7.0, 2.4 * f, 1.7 * f, 0.8 * f])
        return rows, lights, soft, glow, streaks


def embers(sc, lava_z=LAVA_Z, radius=55.0):
    """Sparks popping up off the lava round the camera: a fixed lattice of emitters, each popping on its own
    cycle (so they stay put in the world from frame to frame)."""
    eye = sc['eye']
    tau = sc['tau']
    cell = 7.0
    gx0, gy0 = np.floor((eye[0] - radius) / cell), np.floor((eye[1] - radius) / cell)
    n = int(2 * radius / cell)
    ii, jj = np.meshgrid(np.arange(n) + gx0, np.arange(n) + gy0, indexing='ij')
    ii, jj = ii.ravel(), jj.ravel()
    h1 = (np.sin(ii * 12.9898 + jj * 78.233) * 43758.5453) % 1.0
    h2 = (np.sin(ii * 39.346 + jj * 11.135) * 24634.6345) % 1.0
    h3 = (np.sin(ii * 73.156 + jj * 52.235) * 12345.6789) % 1.0
    x = (ii + h1) * cell
    y = (jj + h2) * cell
    period = 1.2 + 2.2 * h3
    age = ((tau + h1 * 9.0) % period) / period
    z = lava_z + 1.0 + age * (3.0 + 5.0 * h2)
    alpha = np.sin(np.pi * age) * 0.9
    size = 0.10 + 0.08 * h3
    arr = np.column_stack([x, y, z, size, np.ones_like(x), 0.55 + 0.3 * h2, 0.12 + 0.1 * h1, alpha])
    return arr


_STORY = {}


def nether_events(sc):
    meta = RD_META.get('nether', {})
    if 'nether' not in _STORY:
        _STORY['nether'] = NetherStory(sc['tr'], meta)
    st = _STORY['nether']
    rows, lights, soft, glow, streaks = st.rows_lights_parts(sc)
    if streaks:
        sc['streaks'] = np.array(streaks, np.float32)
        if sc.get('renderer') is not None:
            sc['renderer'].streak_col = (5.0, 2.6, 0.8)
    em = embers(sc)
    parts = dict(soft=np.array(soft, np.float32).reshape(-1, 8), glow=np.concatenate(
        [np.array(glow, np.float32).reshape(-1, 8), em.astype(np.float32)]))
    # fire burning on the bridge's netherrack posts (with its light, nearest first)
    fires = meta.get('fires', [])
    if fires:
        eye = sc['eye']
        fr = int(sc['tau'] * 12)
        near = sorted(fires, key=lambda p: np.linalg.norm(np.asarray(p) - eye))
        for k, (x, y, z) in enumerate(fires):
            rows.append(('fire', [x, y, z, 0, 0, 0, 1, 1, 1, 1.25, (fr + k * 5) % 16, 1, 1, 1, 0.8, 5]))
        for (x, y, z) in near[:6]:
            fl = 0.85 + 0.15 * np.sin(sc['tau'] * 13.0 + x)
            if len(lights) < 14:
                lights.append([x, y, z + 0.7, 9.0, 2.4 * fl, 1.2 * fl, 0.35 * fl])
    # the End portal's surface at the bridge's end
    if 'end_portal' in meta:
        ep = meta['end_portal']
        f = int(sc['tau'] * 14) % 32
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                p = ep['center'] + np.array([dx, dy, 0.0])
                rows.append(('end_portal', [*p, 0, 0, 0, 1, 1, 1, 1, (f + dx * 7 + dy * 11) % 32, 1, 1, 1, 0.9,
                                            5]))
    return rows, lights, parts


class EndStory:
    """The dragon's flight (circling, the swoop over the cart, the perch on the fountain and the roar), the crystals
    and their beams, the endermen watching."""

    def __init__(self, tr, meta):
        import mobs
        self.tr = tr
        self.meta = meta
        c = np.asarray(meta.get('center', (0.0, 24.0, 41.0)), float)
        self.center = c
        t_sw = tr.time(tr.marks['dragon'])
        s_sw = tr.marks['dragon']
        T_sw, _, _ = tr.frame(s_sw)
        Th = np.array([T_sw[0], T_sw[1], 0.0])
        Th /= np.linalg.norm(Th)
        Rh = np.array([Th[1], -Th[0], 0.0])
        t_end = tr.time(tr.length - 0.1)
        self.t_perch = t_end - 1.6
        # it perches beside the fountain, left of the cart's dive, facing the track: the cart dives into the portal
        # right under its roaring head (and never through it)
        Tp, _, _ = tr.frame(tr.marks['portal'])
        Tp = np.array([Tp[0], Tp[1], 0.0]) / np.hypot(Tp[0], Tp[1])
        Lp = np.array([-Tp[1], Tp[0], 0.0])
        perch_body = c + Lp * 17.0 + Tp * 1.0 + np.array([0.0, 0.0, 6.0])
        def cart(t):
            return tr.pos(tr.s_at(t))
        keys = [
            (-1.0, c + np.array([50.0, 40.0, 50.0])),
            (0.8, c + np.array([10.0, 62.0, 46.0])),
            (1.9, c + np.array([40.0, -40.0, 34.0])),
            # the swoop: diving out of the sky ahead and to the left, low over the cart, and away behind
            (t_sw - 1.2, cart(t_sw - 1.2) + Th * 70.0 - Rh * 22.0 + np.array([0, 0, 36.0])),
            (t_sw, cart(t_sw) + Th * 3.0 + Rh * 4.0 + np.array([0, 0, 10.0])),
            (t_sw + 0.9, cart(t_sw + 0.9) - Th * 34.0 - Rh * 10.0 + np.array([0, 0, 18.0])),
            (t_sw + 2.2, c + np.array([-30.0, 40.0, 40.0])),
            (self.t_perch - 1.3, c + Lp * 32.0 + Tp * 22.0 + np.array([0.0, 0.0, 22.0])),
            (self.t_perch, perch_body),
            (t_end + 2.0, perch_body),
        ]
        self.kt = np.array([k[0] for k in keys])
        self.kp = np.array([k[1] for k in keys])
        self.t_roar = t_end - 0.9
        self.dragon = mobs.Dragon(self.path, perch=(self.t_perch, -Lp))
        # endermen: a few on the island near the ride
        rng = _rng(5)
        H = meta.get('H')
        org = meta.get('origin')
        self.endermen = []
        for s in (70.0, 96.0, 118.0, 238.0, 258.0, 264.0):
            P = tr.pos(s)
            T, R, U = tr.frame(s)
            side = 1 if int(s) % 2 else -1
            q = P + np.array([R[0], R[1], 0.0]) * side * rng.uniform(7, 12) + np.array([T[0], T[1], 0]) * 6
            if H is not None:
                i, j = int(np.floor(q[0])) - org[0], int(np.floor(q[1])) - org[1]
                if 0 <= i < H.shape[0] and 0 <= j < H.shape[1] and H[i, j] > -100:
                    self.endermen.append((np.array([q[0], q[1], float(H[i, j])]), s))
        self.pillars = meta.get('pillars', [])

    def path(self, t):
        """Catmull-Rom through the keys: (position, flap amount, roar)."""
        kt, kp = self.kt, self.kp
        i = int(np.clip(np.searchsorted(kt, t) - 1, 0, len(kt) - 2))
        t0, t1 = kt[i], kt[i + 1]
        u = float(np.clip((t - t0) / (t1 - t0), 0, 1))
        p0 = kp[max(i - 1, 0)]
        p1, p2 = kp[i], kp[i + 1]
        p3 = kp[min(i + 2, len(kp) - 1)]
        u2, u3 = u * u, u * u * u
        pos = 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2 + (-p0 + 3 * p1 - 3 * p2 + p3) * u3)
        flap = 1.0 if t < self.t_perch else 0.4
        roar = float(np.clip(1 - abs(t - self.t_roar) / 0.55, 0, 1)) if t > self.t_perch else 0.0
        roar = max(roar, float(np.clip(1 - abs(t - (self.kt[4])) / 0.45, 0, 1)) * 0.8)
        return pos, flap, roar

    def rows_lights_parts(self, sc, r):
        import mobs
        tau = sc['tau']
        rows, lights = [], []
        glow = []
        drows, info = self.dragon.rows(tau)
        rows += drows
        body = info['body']
        # a wake of the End's purple particles behind the dragon (so it reads against the black pillars), and a
        # purple glow round it
        rng = _rng(17)
        offs = rng.normal(0, 1, (90, 3)) * np.array([2.2, 2.2, 1.2])
        for k in range(90):
            age = (k / 90.0) * 0.9
            p = np.asarray(self.path(tau - age)[0]) + offs[k] * (1.0 + 2.0 * age) + np.array([0, 0, 0.8 * age])
            glow.append([*p, 0.16 + 0.05 * (k % 3), 1.0, 0.45, 1.0, 0.75 * (1 - k / 90.0)])
        lights.append([*body, 16.0, 1.4, 0.5, 1.8])
        beams = []
        for k, (x, y, tz, rr, caged) in enumerate(self.pillars):
            cr, c = mobs.crystal_rows((x, y, tz), tau, k)
            rows += cr
            if len(lights) < 12:
                lights.append([*c, 12.0, 1.8, 0.7, 2.0])
            glow.append([*c, 1.6, 1.0, 0.6, 1.0, 0.35])
            if np.linalg.norm(body - c) < 60.0 and tau > self.t_perch - 0.4:
                beams.append([*c, *body, 0.22, 0.85])
        for (p, s) in self.endermen:
            rows += mobs.enderman_rows(p, 90.0 + s, look_at=sc['eye'], holding=(int(s) == 96), t=tau)
        # the exit portal's surface round the fountain's column
        ep = self.meta.get('exit_portal')
        if ep is not None:
            f = int(tau * 14) % 32
            c = ep['center']
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    rr = np.hypot(dx, dy)
                    if rr < 0.5 or rr > ep['radius']:
                        continue
                    p = c + np.array([dx, dy, 0.0])
                    rows.append(('end_portal', [*p, 0, 0, 0, 1, 1, 1, 1, (f + dx * 7 + dy * 11) % 32, 1, 1, 1,
                                                0.9, 5]))
        r.streak_col = (1.8, 0.9, 1.9)
        return rows, lights, glow, (np.array(beams, np.float32) if beams else None), info


def end_events(sc):
    meta = RD_META.get('end', {})
    if 'end' not in _STORY:
        _STORY['end'] = EndStory(sc['tr'], meta)
    st = _STORY['end']
    rows, lights, glow, beams, info = st.rows_lights_parts(sc, sc['renderer'])
    sc['streaks'] = beams
    sc['dragon'] = info
    parts = dict(soft=np.zeros((0, 8), np.float32), glow=np.array(glow, np.float32).reshape(-1, 8))
    return rows, lights, parts


# ---------------------------------------------------------------------------------------------
# the deep dark and the Sift
# ---------------------------------------------------------------------------------------------
def floaters(sc, cell, radius, z_lo, z_hi, rise, col, size, alpha, seed, period=(3.0, 6.0)):
    """Souls drifting upwards round the camera: a fixed lattice of emitters (so they stay put in the world from frame
    to frame), each rising, swaying and fading on its own cycle."""
    eye = sc['eye']
    tau = sc['tau']
    gx0, gy0 = np.floor((eye[0] - radius) / cell), np.floor((eye[1] - radius) / cell)
    n = int(2 * radius / cell)
    ii, jj = np.meshgrid(np.arange(n) + gx0, np.arange(n) + gy0, indexing='ij')
    ii, jj = ii.ravel(), jj.ravel()
    h1 = (np.sin(ii * 12.9898 + jj * 78.233 + seed) * 43758.5453) % 1.0
    h2 = (np.sin(ii * 39.346 + jj * 11.135 + seed) * 24634.6345) % 1.0
    h3 = (np.sin(ii * 73.156 + jj * 52.235 + seed) * 12345.6789) % 1.0
    per = period[0] + (period[1] - period[0]) * h3
    age = ((tau + h1 * 17.0) % per) / per
    x = (ii + h1) * cell + 0.8 * np.sin(tau * 1.3 + h2 * 6.0)
    y = (jj + h2) * cell + 0.8 * np.cos(tau * 1.1 + h1 * 6.0)
    z = z_lo + (z_hi - z_lo) * h2 + rise * age
    a = alpha * np.sin(np.pi * age) * (0.7 + 0.3 * np.sin(tau * 9.0 + h3 * 20.0))
    s = size * (0.7 + 0.6 * h3)
    return np.column_stack([x, y, z, s, np.full_like(x, col[0]), np.full_like(x, col[1]), np.full_like(x, col[2]),
                            a]).astype(np.float32)


class DeepStory:
    """The ancient city's portal wakes as the cart comes (the note blocks' melody is in the soundtrack): its opening
    fills with the Sift's swirl and it lights up the city round it."""

    def __init__(self, tr, meta):
        self.tr = tr
        self.meta = meta
        self.t_gate = tr.time(tr.marks['portal'])
        self.t_wake = self.t_gate - 1.7
        self.t_open = self.t_gate - 0.8

    def glow(self, tau):
        return float(np.clip((tau - self.t_wake) / (self.t_open - self.t_wake), 0, 1)) ** 1.5


def deep_events(sc):
    import props as PR
    meta = RD_META.get('deep', {})
    if 'deep' not in _STORY:
        _STORY['deep'] = DeepStory(sc['tr'], meta)
    st = _STORY['deep']
    tau = sc['tau']
    rows, lights, glow = [], [], []
    gate = meta.get('gate')
    g = st.glow(tau)
    if gate is not None and g > 0.01:
        rows += PR.nether_portal_rows(gate['center'], gate['width'], gate['height'], gate['normal'], tau,
                                      kind='sift_portal', glow=0.55 * g)
        c = np.asarray(gate['center'], float)
        n = np.asarray(gate['normal'], float)
        fl = 0.9 + 0.1 * np.sin(tau * 11.0)
        lights.append([*(c - n * 3.0), 30.0, 0.6 * g * fl, 1.3 * g * fl, 1.4 * g * fl])
        lights.append([*(c - n * 2.0 + np.array([0.0, 0.0, 3.0])), 20.0, 1.2 * g, 0.5 * g, 0.8 * g])
        # souls streaming out of the opening
        rng = _rng(int(tau * 30) + 3)
        k = int(60 * g)
        pos = c[None] + rng.normal(0, 1, (k, 3)) * np.array([3.5, 3.5, 3.0]) - n[None] * rng.uniform(0, 8, (k, 1))
        for p in pos:
            glow.append([*p, 0.18, 0.55, 1.0, 1.0, 0.8 * g])
    parts = dict(soft=np.zeros((0, 8), np.float32),
                 glow=np.concatenate([np.array(glow, np.float32).reshape(-1, 8),
                                      floaters(sc, 6.0, 42.0, 14.0, 40.0, 10.0, (0.45, 0.95, 1.0), 0.13, 0.8, 1.7)]))
    return rows, lights, parts


class SiftStory:
    """Blubs hopping beside the track in groups, the portals' surfaces (the one the cart came out of, the rift ahead)."""

    def __init__(self, tr, meta):
        self.tr = tr
        self.meta = meta
        rng = _rng(31)
        H, org = meta.get('H'), meta.get('origin')
        self.blubs = []
        for s in (tr.marks['meadow'] + 6.0, tr.marks['meadow'] + 34.0, tr.marks['meadow'] + 66.0,
                  tr.marks['lake'] - 12.0, tr.marks['fossil'] - 8.0):
            P = tr.pos(s)
            T, R, U = tr.frame(s)
            rh = np.array([R[0], R[1], 0.0]) / max(np.hypot(R[0], R[1]), 1e-6)
            th = np.array([T[0], T[1], 0.0]) / max(np.hypot(T[0], T[1]), 1e-6)
            side = 1.0 if rng.random() < 0.5 else -1.0
            for k in range(4):
                q = P + rh * side * rng.uniform(3.2, 7.5) + th * rng.uniform(-3.0, 9.0)
                z = P[2] - 4.0
                if H is not None:
                    i, j = int(np.floor(q[0])) - org[0], int(np.floor(q[1])) - org[1]
                    if 0 <= i < H.shape[0] and 0 <= j < H.shape[1]:
                        z = float(H[i, j])
                yaw = float(np.degrees(np.arctan2(th[0], th[1]))) + rng.uniform(-60, 60)
                self.blubs.append((np.array([q[0], q[1], z]), yaw, rng.uniform(0, 6.3)))
        self.t_rift = tr.time(tr.marks['rift'])


def sift_events(sc):
    import mobs
    import props as PR
    meta = RD_META.get('sift', {})
    if 'sift' not in _STORY:
        _STORY['sift'] = SiftStory(sc['tr'], meta)
    st = _STORY['sift']
    tau = sc['tau']
    eye = sc['eye']
    rows, lights = [], []
    for (p, yaw, ph) in st.blubs:
        if np.linalg.norm(p - eye) < 90.0:
            rows += mobs.blub_rows(p, yaw, tau, ph)
    for name, pt in meta.get('portals', {}).items():
        if np.linalg.norm(np.asarray(pt['center']) - eye) < 160.0:
            rows += PR.nether_portal_rows(pt['center'], pt['width'], pt['height'], pt['normal'], tau,
                                          kind='sift_portal')
    rift = meta.get('portals', {}).get('rift')
    if rift is not None:
        g = float(np.clip(1.0 - (st.t_rift - tau) / 1.5, 0, 1))
        if g > 0:
            c = np.asarray(rift['center'], float)
            lights.append([*(c - np.asarray(rift['normal']) * 2.0), 26.0, 1.4 * g, 2.6 * g, 2.8 * g])
    parts = dict(soft=np.zeros((0, 8), np.float32),
                 glow=floaters(sc, 7.0, 50.0, eye[2] - 12.0, eye[2] + 10.0, 8.0, (0.62, 0.95, 1.0), 0.14, 0.7, 4.1))
    return rows, lights, parts


def attention(world, tr, tau):
    """(point, weight) the rider glances at, or None."""
    meta = RD_META.get(world, {})
    if world == 'end':
        if 'end' not in _STORY:
            _STORY['end'] = EndStory(tr, meta)
        st = _STORY['end']
        t_sw = st.kt[4]
        w = float(np.clip((tau - (t_sw - 0.9)) / 0.35, 0, 1) * np.clip(((t_sw + 0.1) - tau) / 0.3, 0, 1))
        if w > 0:
            return st.path(tau)[0], 0.4 * w * w * (3 - 2 * w)
        # the roar on the perch: a look up at its head, then back to the portal for the dive
        w = float(np.clip((tau - (st.t_roar - 0.5)) / 0.3, 0, 1) * np.clip(((st.t_roar + 0.3) - tau) / 0.3, 0, 1))
        if w > 0:
            return st.dragon.rows(tau)[1]['head'], 0.3 * w * w * (3 - 2 * w)
        return None
    if world == 'nether':
        if 'nether' not in _STORY:
            _STORY['nether'] = NetherStory(tr, meta)
        st = _STORY['nether']
        t = st.t_shoot
        w = float(np.clip((tau - (t - 0.9)) / 0.35, 0, 1) * np.clip(((t + 0.45) - tau) / 0.4, 0, 1))
        if w > 0:
            w = w * w * (3 - 2 * w)
            return st.ghast_pos(1, tau), 0.36 * w, 10.0 * w
        pos, d, alive = st.fireball(2, tau)
        w = float(np.clip((tau - (st.t_mid - 0.5)) / 0.3, 0, 1) * np.clip(((st.t_mid + 0.1) - tau) / 0.2, 0, 1))
        if alive and w > 0:
            w = w * w * (3 - 2 * w)
            return pos, 0.24 * w, 0.0
        return None
    return None


def jolt(world, tr, tau):
    """The cart slamming down onto the rails after the jump: the rider's head dips and bounces back."""
    if world != 'nether' or 'nether' not in _STORY:
        return 0.0
    tl = tau - _STORY['nether'].t_land
    if tl < 0 or tl > 0.6:
        return 0.0
    return float(np.exp(-tl * 8.0) * np.cos(tl * 20.0) * min(1.0, tl / 0.02))


def EN_basis(T, R, U):
    import entities as EN
    return EN.basis_quat(T, R, U)


def EN_qmul(a, b):
    import entities as EN
    return EN.qmul(a, b)


def EN_qaxis(axis, ang):
    import entities as EN
    return EN.qaxis(axis, ang)


RD_META = {}


# ---------------------------------------------------------------------------------------------
# transitions (drawn over finished frames)
# ---------------------------------------------------------------------------------------------
_GRID = {}


def _grid(h, w):
    if (h, w) not in _GRID:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w / 2.0, h / 2.0
        ang = np.arctan2(yy - cy, xx - cx)
        rad = np.hypot(xx - cx, yy - cy) / max(w, h)
        _GRID[(h, w)] = (yy, xx, ang, rad)
    return _GRID[(h, w)]


def swirl(h, w, ph, c0=(0.20, 0.02, 0.50), c1=(0.80, 0.52, 1.00)):
    """A portal's swirl, full screen (linear 0..1 rgb): the nether portal's purples, or other colours."""
    yy, xx, ang, rad = _grid(h, w)
    v = 0.5 + 0.5 * np.sin(ang * 3.0 + rad * 26.0 - ph * 5.0)
    v = 0.7 * v + 0.3 * (0.5 + 0.5 * np.sin(xx / w * 18.0 + yy / h * 11.0 + ph * 7.0))
    v = np.floor(v * 6.0) / 6.0                                        # banded, like the game's texture
    c0 = np.array(c0, np.float32)
    c1 = np.array(c1, np.float32)
    return c0[None, None] + (c1 - c0)[None, None] * v[..., None]


def starfield(h, w, ph):
    yy, xx, ang, rad = _grid(h, w)
    out = np.zeros((h, w, 3), np.float32) + np.array([0.02, 0.03, 0.04], np.float32)
    cols = [(0.16, 0.66, 0.58), (0.27, 0.78, 0.43), (0.47, 0.31, 0.78), (0.82, 0.90, 0.86), (0.2, 0.5, 0.7)]
    sc = w / 1080.0
    for k, c in enumerate(cols):
        cell = (5 + 2 * k) * sc
        gx = np.floor((xx + ph * (140 + 90 * k) * sc) / cell)
        gy = np.floor((yy + ph * (60 + 35 * k) * sc) / cell)
        hsh = np.sin(gx * 12.9898 + gy * 78.233 + k * 37.1) * 43758.5453 % 1.0
        on = hsh > 0.975
        out[on] = np.array(c, np.float32)
    return out


def wobble(img, amount, ph):
    """The nausea wobble of going through a portal: rows slide sideways in a slow wave."""
    if amount <= 0.01:
        return img
    h, w = img.shape[:2]
    dx = (np.sin(np.arange(h) / h * 7.0 + ph * 6.0) * 22.0 * amount * (w / 1080)).astype(int)
    idx = (np.arange(w)[None, :] - dx[:, None]) % w
    return np.take_along_axis(img, idx[..., None].repeat(3, 2), 1)


def transition_overlay(img, kind, amount, phase):
    f = img.astype(np.float32) / 255.0
    h, w = f.shape[:2]
    if kind == 'nether':
        f = wobble((f * 255).astype(np.uint8), amount, phase).astype(np.float32) / 255.0
        ov = swirl(h, w, phase)
        f = f * (1 - amount) + ov * amount
    elif kind == 'end':
        ov = starfield(h, w, phase)
        f = f * (1 - amount) + ov * amount
    elif kind == 'sift':
        # the Deep Dark Portal: its teal and pink swirl, the world wobbling
        f = wobble((f * 255).astype(np.uint8), amount, phase).astype(np.float32) / 255.0
        ov = swirl(h, w, phase * 1.2, c0=(0.02, 0.34, 0.38), c1=(1.00, 0.70, 0.84))
        f = f * (1 - amount) + ov * amount
    elif kind == 'white':
        ov = np.array([1.0, 0.96, 1.0], np.float32)
        f = f * (1 - amount) + ov[None, None] * amount
    return (np.clip(f, 0, 1) * 255).astype(np.uint8)


def overlays(img, sc):
    """Per-frame overlays the events asked for (the waterfall's wet flash, the explosion's flash)."""
    for kind, a in sc.get('overlays', []):
        if kind == 'flash' and a > 0:
            f = img.astype(np.float32) / 255.0
            f = f * (1 - 0.55 * a) + np.array([1.0, 0.8, 0.5], np.float32)[None, None] * 0.55 * a
            img = (np.clip(f, 0, 1) * 255).astype(np.uint8)
        if kind == 'wet' and a > 0:
            f = img.astype(np.float32) / 255.0
            h, w = f.shape[:2]
            yy, xx, ang, rad = _grid(h, w)
            streak = 0.5 + 0.5 * np.sin(xx / w * 90.0 + np.sin(yy / h * 9.0) * 3.0)
            ov = np.array([0.70, 0.84, 0.96], np.float32)[None, None] * (0.8 + 0.2 * streak[..., None])
            k = 0.75 * a
            f = f * (1 - k) + ov * k
            img = (np.clip(f, 0, 1) * 255).astype(np.uint8)
    return img
