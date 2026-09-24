"""A black hole spawns in the arena and grows in stages (size 1, 10, 100, 1,000). It rips the ground out block
by block into a funnel, drags everything lying around (anvils, TNT, arrows) into a swirling accretion disk, and
eats the four giants one after another: each one leans into the pull, is torn off the ground, spun, stretched
(spaghettified) and shredded into a stream of voxels. Then it collapses to a point and blows everything back out.

Vectorised numpy, stepped per video frame by `scale / FPS` seconds of simulation time in sub-steps, so slow
motion stays smooth and a render is deterministic for a given schedule.

The pull is a velocity field around the hole: far away plain (softened, inverse-square) attraction; close in,
everything is steered onto a Keplerian spiral in the disk plane (tangential speed ~ 1/sqrt(r), a steady inward
drift, flattening onto the plane), and heats up (red, orange, white) the closer it gets. Whatever crosses the
event horizon is gone and counted.
"""
import numpy as np

from explosion_vfx import VFX
from ground import Ground, REG, B_GRASS, B_DIRT, B_STONE, DEPTH
from mathutil import quat_rotate, integrate_quat, axis_angle_quat, quat_mul
from models import VOXEL_DTYPE, make_giants
from vfx import Dust

FPS = 30
SUB = 3
G = 20.0
BLOCK_VOL = 1.0


def _unit(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-12)


def _normalize(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _ease(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def _ease_back(u, s=1.6):
    u = np.clip(u, 0.0, 1.0) - 1.0
    return 1.0 + (s + 1.0) * u ** 3 + s * u ** 2


def random_quats(rng, n):
    q = rng.normal(0, 1, (n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


# =============================================================================================
class Hole:
    """Where the black hole is, how big it is, and how far it reaches, as functions of time."""

    def __init__(self, plan):
        self.spawn = float(plan['spawn'])
        self.stages = [(float(t), float(r)) for t, r in plan['stages']]
        self.path = [(float(t), np.asarray(p, float)) for t, p in plan['path']]
        self.collapse = tuple(float(x) for x in plan['collapse'])
        self.boom = float(plan['boom'])
        self.normal = _unit(plan['disk_normal'])
        self.grow = float(plan.get('grow', 0.7))

    def radius(self, t):
        if t < self.spawn or t >= self.collapse[1]:
            return 0.0
        r = 0.0
        prev = 0.0
        for k, (ts, rs) in enumerate(self.stages):
            if t < ts:
                break
            u = (t - ts) / (0.25 if k == 0 else self.grow)
            r = prev + (rs - prev) * float(_ease_back(u, 1.2 if k else 0.0))
            prev = rs
        if t >= self.collapse[0]:
            u = (t - self.collapse[0]) / (self.collapse[1] - self.collapse[0])
            r *= (1.0 - float(u) ** 1.6)
        return max(r, 0.0)

    def stage(self, t):
        k = -1
        for i, (ts, _) in enumerate(self.stages):
            if t >= ts:
                k = i
        return k

    def centre(self, t):
        ts = [p[0] for p in self.path]
        if t <= ts[0]:
            return self.path[0][1].copy()
        if t >= ts[-1]:
            return self.path[-1][1].copy()
        i = int(np.searchsorted(ts, t)) - 1
        u = float(_ease((t - ts[i]) / (ts[i + 1] - ts[i])))
        return self.path[i][1] * (1 - u) + self.path[i + 1][1] * u

    def reach(self, t, r=None):
        r = self.radius(t) if r is None else r
        if r <= 0.0:
            return 0.0
        return 5.0 * r + 3.5

    def alive(self, t):
        return self.spawn <= t < self.collapse[1]


# =============================================================================================
class Flyers:
    """A set of things flying about: positions, velocities, orientations, spins, heat, plus per-kind payload."""

    def __init__(self, extra=None):
        self.p = np.zeros((0, 3))
        self.v = np.zeros((0, 3))
        self.q = np.zeros((0, 4))
        self.w = np.zeros((0, 3))
        self.heat = np.zeros(0)
        self.rest = np.zeros(0, bool)
        self.extra = dict(extra or {})          # name -> empty array template
        self.half = 0.5                         # height of the centre above the floor when lying on it

    def __len__(self):
        return len(self.p)

    def add(self, p, v, q, w, **payload):
        n = len(p)
        if n == 0:
            return
        self.p = np.concatenate([self.p, p])
        self.v = np.concatenate([self.v, v])
        self.q = np.concatenate([self.q, q])
        self.w = np.concatenate([self.w, w])
        self.heat = np.concatenate([self.heat, np.zeros(n)])
        self.rest = np.concatenate([self.rest, np.zeros(n, bool)])
        for k, tmpl in self.extra.items():
            val = payload[k]
            cur = getattr(self, k, None)
            if cur is None:
                cur = tmpl[:0]
            setattr(self, k, np.concatenate([cur, val]))

    def keep(self, m):
        for a in ('p', 'v', 'q', 'w', 'heat', 'rest'):
            setattr(self, a, getattr(self, a)[m])
        for k in self.extra:
            setattr(self, k, getattr(self, k)[m])


# =============================================================================================
class Body:
    """A giant (or a torn-off part of one) as one rigid piece: its own voxel occupancy on the giant's grid,
    and a pose (rotation about a pivot, then translation, then a stretch along the pull that also thins it)."""

    def __init__(self, giant, occ, name):
        self.g = giant
        self.occ = occ
        self.name = name
        self.q = np.array([0.0, 0.0, 0.0, 1.0])
        self.t = np.zeros(3)
        i, j, k = np.nonzero(occ)
        self.ijk = np.stack([i, j, k], -1)
        self.ps = giant.index_to_world(i, j, k)          # rest positions of every voxel it started with
        self.alive = np.ones(len(i), bool)
        self.com0 = self.ps.mean(0) if len(i) else giant.center.copy()
        self.pivot = self.com0.copy()
        self.axis = np.array([0.0, 0.0, 1.0])
        self.stretch = 1.0
        self.state = 'stand'
        self.vel = np.zeros(3)
        self.heat = 0.0
        self.total = len(i)
        self.count = self.total
        self.version = 0
        self._pack = None
        self.floor_visible = False
        self.swell = 1.0
        self.white = 0.0
        self.fl = None
        self.lean_axis = np.array([1.0, 0.0, 0.0])

    # world position of the body's (rest) centre of mass
    def com(self):
        return self.pivot + self.t + quat_rotate(self.q, self.com0 - self.pivot)

    def to_world(self, p):
        """Rest positions (N, 3) -> posed world positions (same maths as the voxel shader): rigid pose, then
        the half facing the hole drawn out towards it and thinned."""
        w = self.pivot + self.t + quat_rotate(self.q, p - self.pivot)
        if self.stretch != 1.0:
            c = self.com()
            dp = w - c
            al = dp @ self.axis
            thin = 1.0 - (1.0 - self.stretch ** -0.4) * np.clip(al / 4.0, 0.0, 1.0)
            w = c + self.axis * (al * np.where(al > 0, self.stretch, 1.0))[:, None] \
                + (dp - self.axis * al[:, None]) * thin[:, None]
        return w

    def to_rest(self, w):
        """Inverse of to_world."""
        w = np.asarray(w, float).reshape(-1, 3)
        if self.stretch != 1.0:
            c = self.com()
            dp = w - c
            alw = dp @ self.axis
            al = np.where(alw > 0, alw / self.stretch, alw)
            thin = 1.0 - (1.0 - self.stretch ** -0.4) * np.clip(al / 4.0, 0.0, 1.0)
            w = c + self.axis * al[:, None] + (dp - self.axis * alw[:, None]) / thin[:, None]
        qi = self.q * np.array([-1.0, -1.0, -1.0, 1.0])
        return self.pivot + quat_rotate(qi, w - self.pivot - self.t)

    def remove(self, sel):
        """Take voxels (indices into the body's voxel list) out of the body."""
        if len(sel) == 0:
            return
        i, j, k = self.ijk[sel].T
        self.occ[i, j, k] = False
        self.alive[sel] = False
        self.count = int(self.alive.sum())
        self.version += 1

    def pose(self):
        return {'q': self.q.copy(), 't': self.t.copy(), 'pivot': self.pivot.copy(), 'axis': self.axis.copy(),
                'stretch': float(self.stretch), 'scentre': self.com(), 'heat': 0.0}

    def pack(self):
        """Voxel instances of this body (cached until voxels are removed)."""
        if self._pack is not None and self._pack[0] == self.version:
            return self._pack[1]
        g = self.g
        o = self.occ
        e = np.zeros_like(o)
        e[1:] |= ~o[:-1]
        e[:-1] |= ~o[1:]
        e[:, 1:] |= ~o[:, :-1]
        e[:, :-1] |= ~o[:, 1:]
        e[:, :, 1:] |= ~o[:, :, :-1]
        e[:, :, :-1] |= ~o[:, :, 1:]
        e[0] = e[-1] = True
        e[:, 0] = e[:, -1] = True
        e[:, :, 0] = e[:, :, -1] = True
        i, j, k = np.nonzero(o & e)
        out = g.colours(i, j, k)
        out['pos'] = g.index_to_world(i, j, k)
        out['quat'] = (0, 0, 0, 1)
        out['scale'] = g.VS
        vis = np.zeros(len(i), np.uint8)
        for bit, (di, dj, dk) in enumerate(((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))):
            ni, nj, nk = i + di, j + dj, k + dk
            inb = g.inb(ni, nj, nk)
            full = np.zeros(len(i), bool)
            full[inb] = o[ni[inb], nj[inb], nk[inb]]
            empty = ~full
            if bit == 5 and not self.floor_visible:
                empty &= nk >= 0
            vis |= empty.astype(np.uint8) << bit
        out['cy'][:, 3] = vis
        self._pack = (self.version, out)
        return out


# =============================================================================================
class World:
    def __init__(self, plan, seed=0):
        self.rng = np.random.default_rng(seed)
        self.plan = plan
        self.hole = Hole(plan['hole'])
        self.giants = make_giants()
        self.bodies = [Body(g, g.occ.copy(), g.kind) for g in self.giants]
        feet = [(g.lo[0] - 0.4, g.hi[0] + 0.4, g.lo[1] - 0.4, g.hi[1] + 0.4) for g in self.giants]
        self.ground = Ground(feet)
        self.acc = np.zeros((2 * REG, 2 * REG))
        self.dust = Dust(seed + 5)
        self.vfx = VFX(seed + 7)
        self.t = 0.0
        self.frame = 0
        self.eaten = 0.0
        self.eaten_bodies = []
        self.events = {}
        self.frame_explosions = []
        self.big_blasts = []
        self.ground_hits = []
        self.exploded = False
        self.blocks = Flyers({'kind': np.zeros(0, np.int64)})
        self.debris = Flyers({'col': np.zeros(0, VOXEL_DTYPE), 'scale': np.zeros(0)})
        self.props = {k: Flyers({'var': np.zeros(0), 'attached': np.zeros(0, bool), 'z0': np.zeros(0)})
                      for k in ('anvil', 'tnt', 'arrow')}
        self.props['anvil'].half = 0.0
        self.props['arrow'].half = 0.15
        self._scatter_props(plan.get('props', {}))
        self.scripts = {k: dict(v) for k, v in plan['giants'].items()}
        for s in self.scripts.values():
            s['done'] = set()
        self.rings = []                 # sonic boom rings: (t0, origin, target)
        self.flash = 0.0
        self.shock = []                 # expanding shock rings after the explosion: (t0, centre)

    # ------------------------------------------------------------------------------------------
    def _ev(self, name, n=1, pos=None):
        e = self.events.setdefault(name, [0, np.zeros(3), 0])
        e[0] += int(n)
        if pos is not None and n:
            p = np.asarray(pos, float).reshape(-1, 3)
            e[1] = e[1] + p.sum(0)
            e[2] += len(p)

    def _scatter_props(self, spec):
        """Anvils, TNT and arrows left lying around the arena by the earlier battles."""
        rng = self.rng
        for kind, n in spec.items():
            n = int(n)
            ang = rng.uniform(0, 2 * np.pi, n)
            r = np.sqrt(rng.uniform(0.05, 1.0, n)) * 42.0
            p = np.stack([r * np.cos(ang), r * np.sin(ang) - 6.0, np.zeros(n)], -1)
            # keep clear of the giants' feet
            for g in self.giants:
                inside = np.all((p[:, :2] > g.lo[:2] - 1.0) & (p[:, :2] < g.hi[:2] + 1.0), axis=1)
                p[inside, 1] -= g.hi[1] - g.lo[1] + 3.0
            if kind == 'arrow':
                d = _normalize(np.c_[rng.normal(0, 0.35, (n, 2)), -np.ones(n)])
                side = _normalize(np.cross(d, rng.normal(0, 1, (n, 3))))
                y = np.cross(d, side)
                from mathutil import quat_from_basis
                q = quat_from_basis(d, np.cross(d, y))
                p[:, 2] = 0.55
                var = rng.integers(0, 3, n).astype(float)
            else:
                yaw = rng.uniform(0, 2 * np.pi, n)
                q = axis_angle_quat(np.tile([0.0, 0.0, 1.0], (n, 1)), yaw)
                p[:, 2] = 0.0 if kind == 'anvil' else 0.5
                var = rng.integers(0, 2, n).astype(float) if kind == 'anvil' else np.zeros(n)
            self.props[kind].add(p, np.zeros((n, 3)), q, np.zeros((n, 3)), var=var, attached=np.ones(n, bool),
                                 z0=p[:, 2].copy())

    # ------------------------------------------------------------------------------------------
    # the pull
    # ------------------------------------------------------------------------------------------
    def _field(self, p, v, dt, heavy=1.0):
        """New velocities and heat targets for points p moving at v, over dt."""
        h = self.hole
        R = h.radius(self.t)
        if R <= 0.0:
            v = v.copy()
            v[:, 2] -= G * dt
            return v, np.zeros(len(p))
        C = h.centre(self.t)
        L = h.reach(self.t, R)
        rel = p - C
        d = np.maximum(np.linalg.norm(rel, axis=1), 1e-6)
        rhat = -rel / d[:, None]
        coll = self.t >= h.collapse[0]
        g0 = (26.0 if not coll else 70.0) / heavy
        gmag = np.minimum(g0 * L * L / (d * d + (0.6 * R) ** 2 + 1.0), 450.0)
        a = rhat * gmag[:, None]
        # onto a Keplerian spiral in the disk plane
        n = h.normal
        hh = rel @ n
        rv = rel - hh[:, None] * n
        rho = np.maximum(np.linalg.norm(rv, axis=1), 1e-6)
        rho_hat = rv / rho[:, None]
        that = np.cross(n, rho_hat)
        vt = np.minimum(9.0 * np.sqrt(L / np.maximum(rho, 0.6 * R + 0.3)), 95.0)
        vr = -(0.32 if not coll else 1.6) * vt
        target = that * vt[:, None] + rho_hat * vr[:, None] - n * (hh * 2.2)[:, None]
        near = np.clip((2.2 * L - d) / (1.2 * L), 0.0, 1.0)
        k = (2.4 / heavy) * near * np.clip(L / d, 0.0, 3.0) ** 1.5
        a += (target - v) * k[:, None]
        # far from the hole things still fall
        a[:, 2] -= G * (1.0 - near) * 0.8
        v = v + a * dt
        sp = np.linalg.norm(v, axis=1)
        cap = 70.0
        v = np.where((sp > cap)[:, None], v * (cap / np.maximum(sp, 1e-6))[:, None], v)
        heat = np.clip((2.5 * R + 1.0 - d) / (1.7 * R + 1.0), 0.0, 1.0) ** 0.9
        return v, heat

    def _step_flyers(self, f, dt, heavy=1.0):
        """Advance a set of flyers through the field; returns the mask of those that crossed the horizon."""
        if len(f) == 0:
            return np.zeros(0, bool)
        move = ~f.rest
        attached = getattr(f, 'attached', None)
        if attached is not None:
            move &= ~attached
        eaten = np.zeros(len(f), bool)
        if not move.any():
            return eaten
        idx = np.nonzero(move)[0]
        v, heat = self._field(f.p[idx], f.v[idx], dt, heavy)
        f.v[idx] = v
        f.p[idx] += v * dt
        f.heat[idx] = np.maximum(f.heat[idx] * np.exp(-dt / 0.6), heat)
        f.q[idx] = integrate_quat(f.q[idx], f.w[idx], dt)
        R = self.hole.radius(self.t)
        if R > 0.0:
            d = np.linalg.norm(f.p[idx] - self.hole.centre(self.t), axis=1)
            eaten[idx[d < R * 0.98]] = True
        # the ground (as far as it is left)
        gz = self.ground.surface(f.p[idx, 0], f.p[idx, 1])
        half = 0.5 * f.scale[idx] if 'scale' in f.extra else np.full(len(idx), f.half)
        low = (f.p[idx, 2] - half < gz) & ~eaten[idx]
        if low.any():
            li = idx[low]
            f.p[li, 2] = gz[low] + half[low]
            if R > 0.0:
                # sliding along the floor of the pit on the way in
                f.v[li, 2] = np.maximum(f.v[li, 2], 0.0)
                f.v[li, :2] *= 0.94
            else:
                fast = np.linalg.norm(f.v[li], axis=1) > 4.0
                self.ground_hits.append(f.p[li[fast]] - np.array([0.0, 0.0, 0.3]))
                self._ev('land', int(fast.sum()) + int((~fast).sum() * 0.2), f.p[li][:: max(1, len(li) // 32)])
                f.v[li] *= np.where(fast, -0.25, 0.0)[:, None]
                f.v[li, :2] *= 0.5
                f.w[li] *= 0.4
                calm = ~fast
                f.rest[li[calm]] = True
                f.v[li[calm]] = 0.0
                f.w[li[calm]] = 0.0
        return eaten

    # ------------------------------------------------------------------------------------------
    # the ground
    # ------------------------------------------------------------------------------------------
    def _peel(self, dt):
        h = self.hole
        R = h.radius(self.t)
        if R <= 0.0 or self.t >= h.collapse[0]:
            return
        C = h.centre(self.t)
        L = h.reach(self.t, R) * 1.05
        gr = self.ground
        dg = np.sqrt((gr.cx - C[0]) ** 2 + (gr.cy - C[1]) ** 2 + C[2] ** 2)
        u = np.clip(1.0 - dg / L, 0.0, 1.0)
        rate = 9.0 * u ** 1.4 * (1.0 + 2.0 * (R > 3.0))
        self.acc += rate * dt
        rho = np.sqrt((gr.cx - C[0]) ** 2 + (gr.cy - C[1]) ** 2)
        want = -np.minimum(DEPTH, np.ceil((L - rho) * 0.55 * np.clip(1.0 - C[2] / L, 0.2, 1.0))).astype(np.int32)
        want = np.where(u > 0.0, want, 0)
        ready = self.acc >= 1.0
        self.acc[ready] -= 1.0
        pos, kind = gr.peel(np.where(ready, want, 1))
        n = len(pos)
        if n == 0:
            return
        rng = self.rng
        # torn out: a hop upwards and towards the hole, tumbling
        to = _normalize(C - pos)
        v = to * rng.uniform(1.0, 4.0, (n, 1)) + np.c_[rng.normal(0, 1.2, (n, 2)), rng.uniform(2.5, 7.0, n)]
        w = rng.normal(0, 5.0, (n, 3))
        q = random_quats(rng, n) * 0.0 + np.array([0.0, 0.0, 0.0, 1.0])
        q = integrate_quat(q, rng.normal(0, 0.3, (n, 3)), 1.0)
        self.blocks.add(pos, v, q, w, kind=kind)
        self._ev('peel', n, pos[:: max(1, n // 32)])
        # props on top of a column that went fly with it
        for kind_, f in self.props.items():
            if len(f) == 0:
                continue
            att = np.nonzero(f.attached)[0]
            if len(att) == 0:
                continue
            gz = gr.surface(f.p[att, 0], f.p[att, 1])
            loose = gz < -0.5
            if loose.any():
                k = att[loose]
                f.attached[k] = False
                f.v[k] = _normalize(C - f.p[k]) * 2.0 + np.array([0.0, 0.0, 4.0])
                f.w[k] = rng.normal(0, 3.0, (len(k), 3))
                self._ev('prop_lift', len(k), f.p[k])

    # ------------------------------------------------------------------------------------------
    # the giants
    # ------------------------------------------------------------------------------------------
    def _shed(self, b, n, frac_debris=0.5, scale=1.45, sel=None):
        """Tear voxels off body b (the n nearest to the hole, or the given ones); a fraction of them fly on as
        slightly bigger debris cubes (so the stream carries about the volume that was torn off)."""
        if b.count <= 0:
            return
        alive = np.nonzero(b.alive)[0]
        if sel is None:
            n = min(int(n), len(alive))
            if n <= 0:
                return
            C = self.hole.centre(self.t)
            h = b.to_rest(C)[0]
            d = np.einsum('ij,ij->i', b.ps[alive] - h, b.ps[alive] - h)
            sel = alive[np.argpartition(d, n - 1)[:n]] if n < len(alive) else alive
        if len(sel) == 0:
            return
        g = b.g
        pw = b.to_world(b.ps[sel])
        b.remove(sel)
        self.eaten += len(sel) * g.VS ** 3
        rng = self.rng
        keep = rng.random(len(sel)) < frac_debris
        if keep.any():
            C = self.hole.centre(self.t)
            p = pw[keep]
            m = len(p)
            i, j, k = b.ijk[sel][keep].T
            col = g.colours(i, j, k)
            col['cy'][:, 3] = 0x3F
            col['cz'][:, 3] = np.where(col['cz'][:, 3] == 255, 255, 0)
            vel = b.vel + _normalize(C - p) * rng.uniform(2.0, 6.0, (m, 1)) + rng.normal(0, 1.2, (m, 3))
            self.debris.add(p, vel, random_quats(rng, m), rng.normal(0, 6.0, (m, 3)), col=col,
                            scale=np.full(m, g.VS * scale))
            self.debris.heat[-m:] = b.heat
        self._ev('shed', len(sel), pw[:: max(1, len(sel) // 16)])

    def _horizon(self, b, R, C, margin):
        """Whatever part of the body reaches the event horizon is torn off right there."""
        alive = np.nonzero(b.alive)[0]
        if len(alive) == 0 or R <= 0.0:
            return
        Rs = R + margin
        h = b.to_rest(C)[0]
        d = np.linalg.norm(b.ps[alive] - h, axis=1)
        cand = alive[d < Rs * b.stretch ** 0.4 + 1e-6]
        if len(cand) == 0:
            return
        dw = np.linalg.norm(b.to_world(b.ps[cand]) - C, axis=1)
        inside = cand[dw < R]
        shell = cand[(dw >= R) & (dw < Rs)]
        if len(inside):
            self._shed(b, 0, frac_debris=0.0, sel=inside)
        if len(shell):
            self._shed(b, 0, frac_debris=0.3, sel=shell)
        # the front of a stretched body is drawn right into the hole: whatever reaches it along the pull is gone
        if b.stretch > 1.0 and b.count:
            alive = np.nonzero(b.alive)[0]
            d = float(np.linalg.norm(C - b.com()))
            a_rest = quat_rotate(b.q * np.array([-1.0, -1.0, -1.0, 1.0]), b.axis)
            al = (b.ps[alive] - b.com0) @ a_rest
            past = alive[al * b.stretch > d - 0.3 * R]
            if len(past):
                self._shed(b, 0, frac_debris=0.0, sel=past)

    def _bodies_step(self, dt):
        h = self.hole
        R = h.radius(self.t)
        C = h.centre(self.t)
        L = h.reach(self.t, R) if R > 0 else 1.0
        new_bodies = []
        for bi, b in enumerate(self.bodies):
            if b.state == 'gone':
                continue
            sc = self.scripts.get(b.name, {})
            # --- scripted beats
            if b.state == 'stand' and 'lean' in sc and self.t >= sc['lean']:
                b.state = 'lean'
                to = C - b.com0
                to[2] = 0.0
                to = _unit(to)
                b.lean_axis = _unit(np.cross(np.array([0.0, 0.0, 1.0]), to))
                g = b.g
                half = 0.5 * np.abs(to[:2] @ np.array([g.hi[0] - g.lo[0], g.hi[1] - g.lo[1]]) * 0.5) + 0.5
                b.pivot = np.array([b.com0[0], b.com0[1], 0.0]) + np.r_[to[:2], 0.0] * half
                b.lean_t0 = self.t
                self._ev('lean', 1, b.com())
            if 'sever' in sc and 'sever' not in sc['done'] and self.t >= sc['sever'][0]:
                sc['done'].add('sever')
                for part, t_eat in sc['sever'][1]:
                    new_bodies.append(self._split(b, part, t_eat))
            if b.state == 'lean':
                u = (self.t - b.lean_t0) / max(1e-3, sc.get('lift', self.t + 1) - b.lean_t0)
                ang = sc.get('lean_angle', 0.35) * float(_ease(u))
                trem = (0.012 + 0.02 * u) * (np.sin(self.t * 37.0 + bi) + 0.7 * np.sin(self.t * 53.0 + 2 * bi))
                b.q = axis_angle_quat(b.lean_axis, ang + trem)
                b.t = np.zeros(3)
                # the skin facing the hole starts to peel off in a stream
                if R > 0:
                    d = np.linalg.norm(b.com() - C)
                    n = b.total * sc.get('peel', 0.02) * min(L / max(d, 1.0), 3.0) ** 2 * dt
                    self._shed(b, self._round(n), frac_debris=0.5)
                if self.t >= sc.get('lift', 1e9):
                    self._lift(b, sc)
            elif b.state == 'fly':
                self._fly(b, dt, R, C, L)
        self.bodies.extend(nb for nb in new_bodies if nb is not None)

    def _round(self, x):
        """Stochastic rounding, so small per-step rates still average out right."""
        f = int(np.floor(x))
        return f + int(self.rng.random() < x - f)

    def _split(self, b, part_name, t_eat):
        """Rip a named part (an arm) off body b; it flies into the hole on its own."""
        g = b.g
        pid = g.part_names.index(part_name) + 1
        sel = np.nonzero(b.alive & (g.part[b.ijk[:, 0], b.ijk[:, 1], b.ijk[:, 2]] == pid))[0]
        if len(sel) == 0:
            return None
        part = np.zeros_like(b.occ)
        i, j, k = b.ijk[sel].T
        part[i, j, k] = True
        c_world = b.to_world(b.ps[sel]).mean(0)
        b.remove(sel)
        nb = Body(g, part, b.name + '_part')
        nb.q = b.q.copy()
        nb.pivot = nb.com0.copy()
        nb.t = c_world - nb.com0
        nb.lean_axis = b.lean_axis.copy()
        nb.floor_visible = True
        nb.heat = b.heat
        self._start_flight(nb, t_eat, rise=1.5, swirl=0.7, spin=2.2)
        self._ev('sever', len(sel), c_world)
        return nb

    def _lift(self, b, sc):
        """Torn off the ground: from here on the body spirals into the hole."""
        c_world = b.com()
        b.pivot = b.com0.copy()
        b.t = c_world - b.com0
        b.floor_visible = True
        b.version += 1
        if b.name == b.g.kind:
            self.ground.release(self.giants.index(b.g))
        self._start_flight(b, sc['eat'], rise=sc.get('rise', 3.0), swirl=sc.get('swirl', 0.9),
                           spin=sc.get('spin', 1.6))
        self._ev('lift', 1, c_world)

    def _start_flight(self, b, t_eat, rise=3.0, swirl=0.9, spin=1.6):
        """A spiral path from where the body is now into the centre of the hole, arriving at t_eat."""
        C = self.hole.centre(self.t)
        n = self.hole.normal
        c = b.com()
        rel = c - C
        h0 = float(rel @ n)
        rv = rel - h0 * n
        rho0 = max(float(np.linalg.norm(rv)), 1e-3)
        e1 = rv / rho0
        b.fl = dict(t0=self.t, t1=float(t_eat), rho0=rho0, h0=h0, e1=e1, e2=np.cross(n, e1), phi=swirl * np.pi,
                    rise=rise, q0=b.q.copy(), c0=c.copy(),
                    spin_axis=_unit(b.lean_axis + self.rng.normal(0, 0.35, 3)), spin=spin * np.pi)
        b.state = 'fly'
        b.prev_c = c.copy()

    def _flight(self, b, t):
        f = b.fl
        u = float(np.clip((t - f['t0']) / (f['t1'] - f['t0']), 0.0, 1.0))
        ue = u ** 1.7                                 # falling in: slow to start, faster and faster
        C = self.hole.centre(t)
        n = self.hole.normal
        rho = f['rho0'] * (1.0 - ue)
        phi = f['phi'] * u ** 2.0
        e = f['e1'] * np.cos(phi) + f['e2'] * np.sin(phi)
        p = C + e * rho + n * (f['h0'] * (1.0 - ue) ** 1.4)
        p = p + np.array([0.0, 0.0, f['rise'] * np.sin(np.pi * min(u / 0.7, 1.0)) * (1.0 - u)])
        # blend out of the rest pose over the first moments, so the lift has no jump
        w0 = 1.0 - float(_ease(u / 0.12))
        p = p * (1 - w0) + (f['c0'] + np.array([0.0, 0.0, 0.8 * u])) * w0
        q = quat_mul(axis_angle_quat(f['spin_axis'], f['spin'] * u ** 1.8), f['q0'])
        return p, q, u

    def _fly(self, b, dt, R, C, L):
        c_old = b.com()
        p, q, u = self._flight(b, self.t + dt)
        b.q = q / np.linalg.norm(q)
        b.t = b.t + (p - b.com())
        b.vel = (p - c_old) / max(dt, 1e-6)
        d = float(np.linalg.norm(p - C)) if R > 0 else 999.0
        # spaghettification: stretched along the pull (and thinner across it), harder the closer it gets
        if R > 0:
            b.axis = _unit(C - p)
            x = (2.2 * R + 2.5) / max(d, 1e-3)
            b.stretch = 1.0 + min(2.2, 1.1 * max(0.0, x * x - 0.3))
        heat = np.clip((3.0 * R + 2.0 - d) / (2.0 * R + 2.0), 0.0, 1.0) if R > 0 else 0.0
        b.heat = float(heat) ** 0.8 * 0.85
        sc = self.scripts.get(b.name, {})
        # the creeper primes as it flies, and goes off at the edge of the hole
        if b.name == 'creeper' and 'prime' in sc and 'explode' not in sc['done']:
            if self.t >= sc['prime']:
                w = np.clip((self.t - sc['prime']) / 1.4, 0, 1)
                period = 0.36 - 0.24 * w
                ph = (self.t - sc['prime']) / period
                b.white = (0.5 if (ph % 1.0) < 0.5 else 0.0) * (0.45 + 0.55 * w)
                b.swell = 1.0 + 0.3 * w ** 3
                if 'hiss' not in sc['done']:
                    sc['done'].add('hiss')
                    self._ev('hiss', 1, p)
            if R > 0 and d < R * 1.35 + sc.get('explode_d', 3.0):
                sc['done'].add('explode')
                self._creeper_boom(b)
                return
        if R > 0 and b.count > 0:
            # shredded by the tides on the way in, and whatever touches the horizon is gone
            n = b.total * sc.get('strip', 0.03) * min(L / max(d, 1.0), 3.0) ** 2 * dt
            self._shed(b, self._round(n), frac_debris=0.4)
            self._horizon(b, R, C, margin=0.35 + 0.08 * R)
            if u >= 1.0:
                # its centre is in: the rest is sucked in within a fraction of a second
                self._shed(b, max(1, int(b.count * dt / 0.18)), frac_debris=0.4)
        if b.count <= max(30, 0.01 * b.total):
            self._shed(b, b.count, frac_debris=0.3)
            b.state = 'gone'
            b.count = 0
            b.version += 1
            if not b.name.endswith('_part'):
                self.eaten_bodies.append((b.name, self.t))
                self._ev('eaten', 1, p)

    def _creeper_boom(self, b):
        """The creeper goes off at the edge of the event horizon: its voxels blasted out, then pulled back in."""
        rng = self.rng
        g = b.g
        alive = np.nonzero(b.alive)[0]
        pw = b.to_world(b.ps[alive])
        c = b.com()
        n = len(alive)
        keep = rng.random(n) < 0.35
        i, j, k = b.ijk[alive[keep]].T
        col = g.colours(i, j, k)
        col['cy'][:, 3] = 0x3F
        col['cz'][:, 3] = 0
        m = int(keep.sum())
        d = _normalize(pw[keep] - c + rng.normal(0, 0.6, (m, 3)))
        vel = d * rng.uniform(10.0, 30.0, (m, 1)) + b.vel * 0.5
        self.debris.add(pw[keep], vel, random_quats(rng, m), rng.normal(0, 8.0, (m, 3)), col=col,
                        scale=np.full(m, g.VS * 1.45))
        self.debris.heat[-m:] = rng.uniform(0.0, 0.6, m)
        b.remove(alive)
        self.eaten += n * g.VS ** 3
        b.count = 0
        b.state = 'gone'
        b.version += 1
        self.eaten_bodies.append((b.name, self.t))
        self.frame_explosions.append(np.array([c]))
        self.big_blasts.append((c, 2.6))
        self._ev('creeper_boom', 1, c)
        self._ev('eaten', 1, c)

    # ------------------------------------------------------------------------------------------
    def _warden_boom(self):
        sc = self.scripts.get('warden', {})
        if 'boom' in sc and 'boom' not in sc['done'] and self.t >= sc['boom']:
            sc['done'].add('boom')
            b = next(b for b in self.bodies if b.name == 'warden')
            head = b.to_world(np.asarray(sc['head'], float)[None])[0]
            self.rings.append((self.t, head))
            self._ev('boom', 1, head)

    def ring_instances(self):
        """Sonic boom rings travelling from the Warden's head into the hole, shrinking as they are swallowed."""
        out = []
        if not self.hole.alive(self.t):
            return np.zeros((0, 8), np.float32)
        C = self.hole.centre(self.t)
        R = self.hole.radius(self.t)
        for t0, o in self.rings:
            age = self.t - t0
            path = C - o
            L = np.linalg.norm(path)
            d = path / max(L, 1e-6)
            for k in range(22):
                s = age * 70.0 - k * 2.6
                if s < 0 or s > L:
                    continue
                u = s / L
                alpha = (1.0 - u) ** 0.8 * min(1.0, (0.6 - age + 0.6) / 0.6) if age < 1.2 else 0.0
                if alpha <= 0.01:
                    continue
                rad = (1.8 + 2.4 * min(1.0, age / 0.4)) * (1.0 - 0.85 * u ** 2) + 0.2 * R * u
                out.append([*(o + d * s), *d, rad, alpha])
        return np.array(out, np.float32).reshape(-1, 8)

    # ------------------------------------------------------------------------------------------
    def _explode(self):
        """The collapse ends in a flash: everything left is blasted out and rains back down."""
        rng = self.rng
        h = self.hole
        C = h.centre(self.t)
        self.exploded = True
        self.frame_explosions.append(np.array([C]))
        self.big_blasts.append((C, 6.0))
        self.shock.append((self.t, C.copy()))
        self._ev('explode', 1, C)
        ej = self.plan.get('ejecta', {})
        # what was left in the disk
        for f in [self.blocks, self.debris] + list(self.props.values()):
            if len(f) == 0:
                continue
            mv = ~f.rest
            if hasattr(f, 'attached'):
                mv &= ~f.attached
            idx = np.nonzero(mv)[0]
            if len(idx):
                d = _normalize(f.p[idx] - C + rng.normal(0, 0.5, (len(idx), 3)))
                d[:, 2] = np.abs(d[:, 2]) * 0.8 + 0.5
                f.v[idx] = _normalize(d) * rng.uniform(10.0, 36.0, (len(idx), 1))
                f.w[idx] = rng.normal(0, 7.0, (len(idx), 3))
        # everything it swallowed comes flying back out
        nb = int(ej.get('blocks', 0))
        if nb:
            d = _normalize(rng.normal(0, 1, (nb, 3)) + np.array([0.0, 0.0, 1.1]))
            p = C + d * rng.uniform(0.5, 2.5, (nb, 1))
            v = d * rng.uniform(8.0, 34.0, (nb, 1))
            kind = rng.choice([B_GRASS, B_DIRT, B_STONE], size=nb, p=[0.2, 0.45, 0.35])
            self.blocks.add(p, v, random_quats(rng, nb), rng.normal(0, 7.0, (nb, 3)), kind=kind)
            self.blocks.heat[-nb:] = rng.uniform(0.3, 1.0, nb)
        nv = int(ej.get('voxels', 0))
        if nv:
            cols = []
            for g in self.giants:
                i, j, k = np.nonzero(g.occ)
                pick = rng.integers(0, len(i), nv // len(self.giants))
                cols.append(g.colours(i[pick], j[pick], k[pick]))
            col = np.concatenate(cols)
            m = len(col)
            col['cy'][:, 3] = 0x3F
            col['cz'][:, 3] = 0
            d = _normalize(rng.normal(0, 1, (m, 3)) + np.array([0.0, 0.0, 1.0]))
            self.debris.add(C + d * rng.uniform(0.5, 2.0, (m, 1)), d * rng.uniform(8.0, 34.0, (m, 1)),
                            random_quats(rng, m), rng.normal(0, 8.0, (m, 3)), col=col, scale=np.full(m, 0.45))
            self.debris.heat[-m:] = rng.uniform(0.2, 0.9, m)

    # ------------------------------------------------------------------------------------------
    def step_frame(self, scale=1.0):
        self.events = {}
        self.frame_explosions = []
        self.big_blasts = []
        self.ground_hits = []
        h = self.hole
        n_sub = max(1, int(np.ceil(SUB * scale - 1e-9)))
        dt = scale / FPS / n_sub
        r0 = h.radius(self.t)
        st0 = h.stage(self.t)
        for _ in range(n_sub):
            if not self.exploded and self.t >= h.boom:
                self._explode()
            self._warden_boom()
            self._peel(dt)
            self._bodies_step(dt)
            eb = self._step_flyers(self.blocks, dt)
            if eb.any():
                self.eaten += int(eb.sum()) * BLOCK_VOL
                self._ev('eat', int(eb.sum()), self.blocks.p[eb][:: max(1, int(eb.sum()) // 16)])
                self.blocks.keep(~eb)
            ed = self._step_flyers(self.debris, dt, heavy=1.0)
            if ed.any():
                self._ev('eat_small', int(ed.sum()))
                self.debris.keep(~ed)
            for kind, f in self.props.items():
                ep = self._step_flyers(f, dt, heavy=1.3 if kind == 'anvil' else 1.0)
                if kind == 'tnt' and len(f):
                    hot = (f.heat > 0.45) & ~f.attached & ~ep
                    if hot.any():
                        c = f.p[hot]
                        self.frame_explosions.append(c.copy())
                        self._ev('tnt_boom', int(hot.sum()), c)
                        ep = ep | hot
                if ep.any():
                    self.eaten += int(ep.sum())
                    self._ev('eat_prop', int(ep.sum()), f.p[ep])
                    f.keep(~ep)
            self.t += dt
        st1 = h.stage(self.t)
        if st1 != st0 and st1 >= 0:
            self._ev('stage', 1, h.centre(self.t))
            if st1 == 0:
                self._ev('spawn', 1, h.centre(self.t))
        if r0 > 0 and h.radius(self.t) == 0.0 and self.t >= h.collapse[1] - 1e-6 and 'collapsed' not in self.__dict__:
            self.collapsed = True
            self._ev('collapsed', 1, h.centre(self.t))
        if h.collapse[0] <= self.t < h.collapse[0] + scale / FPS + 1e-6:
            self._ev('collapse', 1, h.centre(self.t))
        C = np.concatenate(self.frame_explosions) if self.frame_explosions else np.zeros((0, 3))
        self.vfx.step(C, scale / FPS, self.big_blasts)
        self._pull_vfx(scale / FPS)
        self.dust.step(np.concatenate(self.ground_hits) if self.ground_hits else np.zeros((0, 3)), scale / FPS)
        self.frame += 1
        return self.events

    def _pull_vfx(self, dt):
        """Smoke and fireballs near the hole get dragged into it too."""
        vfx = self.vfx
        R = self.hole.radius(self.t)
        if R <= 0.0 or len(vfx.pp) == 0:
            return
        v, _ = self._field(vfx.pp, vfx.pv, dt, heavy=1.0)
        vfx.pv = v
        d = np.linalg.norm(vfx.pp - self.hole.centre(self.t), axis=1)
        vfx.page = np.where(d < R, vfx.plife, vfx.page)

    # ------------------------------------------------------------------------------------------
    # render data
    # ------------------------------------------------------------------------------------------
    def instances(self):
        """Voxels (bodies first, then debris), the bodies' instance ranges and poses, props and effects."""
        parts, ranges = [], []
        n0 = 0
        for b in self.bodies:
            if b.state == 'gone' or b.count <= 0:
                continue
            inst = b.pack()
            if len(inst) == 0:
                continue
            parts.append(inst)
            ranges.append((n0, n0 + len(inst), b))
            n0 += len(inst)
        if len(self.debris):
            inst = self.debris.col.copy()
            inst['pos'] = self.debris.p
            inst['quat'] = self.debris.q
            inst['scale'] = self.debris.scale
            hot = self.debris.heat > 0.02
            a = inst['cz'][:, 3]
            inst['cz'][:, 3] = np.where(hot & (a != 255), np.clip(1 + self.debris.heat * 252, 1, 253), a).astype(np.uint8)
            parts.append(inst)
        vox = np.concatenate(parts) if parts else np.zeros(0, VOXEL_DTYPE)
        props = {}
        if len(self.blocks):
            f = self.blocks
            out = np.zeros((len(f), 11), np.float32)
            out[:, 0:3] = f.p
            out[:, 3:7] = f.q
            out[:, 7] = 1.0
            out[:, 8] = f.kind * 3
            out[:, 9] = 1.0
            out[:, 10] = np.where(f.heat > 0.02, np.maximum(f.heat, 0.01), 0.0)
            props['block'] = out
        for kind in ('anvil', 'tnt'):
            f = self.props[kind]
            if len(f):
                out = np.zeros((len(f), 11), np.float32)
                out[:, 0:3] = f.p
                out[:, 3:7] = f.q
                out[:, 7] = 1.0
                out[:, 8] = f.var
                out[:, 9] = 1.0
                if kind == 'tnt':
                    out[:, 10] = np.clip(f.heat * 1.5, 0, 0.9) * (np.floor(self.t / 0.12) % 2)
                props[kind] = out
        arrows = None
        f = self.props['arrow']
        if len(f):
            arrows = np.zeros((len(f), 10), np.float32)
            arrows[:, 0:3] = f.p
            arrows[:, 3:7] = f.q
            arrows[:, 7] = 1.0
            arrows[:, 8] = f.var
            arrows[:, 9] = 1.0
        fx = {'puffs': np.concatenate([self.dust.puffs(), self.vfx.puffs()]), 'flashes': self.vfx.flashes(),
              'lights': self.vfx.lights(), 'rings': self.ring_instances()}
        return vox, ranges, props, arrows, fx
