"""Physics for one round: arrows fly in ballistic arcs into the voxel zombie and stick.

Vectorised numpy with a variable time step: every video frame advances the simulation by `scale / FPS`
seconds in ceil(SUB * scale) equal sub-steps, so slow motion (scale < 1) is smooth, and a render is
deterministic as long as the same time-scale schedule is used (see timeline.py).

Arrows carve a thin channel into the zombie, slow down per voxel (bone costs more) and stop exactly where they
run out of speed. A stuck arrow is anchored to a flesh voxel: if that flesh is destroyed it re-anchors nearby
or falls out; arrows in a body part that breaks off ride along with it. Arrows that miss sink into the ground
at their impact angle and kick up dust. Loose flesh crumbles, disconnected parts fall as rigid chunks and
shatter, debris piles up.
"""
import numpy as np
from scipy import ndimage

from zombie import Giant, VOXEL_DTYPE, VS, NX, NY, NZ, GX0, GY0, GZ0
from mathutil import quat_from_basis, integrate_quat, quat_rotate, axis_angle_quat, quat_mul, slerp
import arrows as AR
from vfx import Dust

FPS = 30
SUB = 4                          # sub-steps per video frame at normal speed
G = 20.0                         # arrow gravity (Minecraft-like, blocks / s^2)
GRAV = np.array([0.0, 0.0, -G])
TIP = AR.TIP - 0.04              # tip distance from the arrow centre along its axis
UP = np.array([0.0, 0.0, 1.0])

# pile height field around the zombie
PILE_HALF = 48.0
PILE_RES = 0.25
PN = int(2 * PILE_HALF / PILE_RES)

# arrow states
WAIT, FLY, STUCK, GROUND, FALL, REST, CARRIED = 0, 1, 2, 3, 4, 5, 6

CHUNK_KEEP = 400                 # pieces with at least this many voxels tumble to rest instead of shattering

# where each part joins the rest of him: world boxes (x0, x1, y0, y1, z0, z1) blown out when it is severed
JOINTS = {
    'arm_r': (-4.75, -4.0, -2.0, 2.0, 20.0, 24.0),
    'arm_l': (4.0, 4.75, -2.0, 2.0, 20.0, 24.0),
    'head': (-4.0, 4.0, -2.0, 2.0, 24.0, 24.75),
}

# search offsets for anchoring (voxels within 2 of the tip voxel, nearest first)
_d = np.arange(-2, 3)
_A, _B, _C = np.meshgrid(_d, _d, _d, indexing='ij')
ANCHOR_OFFS = np.stack([_A.ravel(), _B.ravel(), _C.ravel()], -1)
ANCHOR_OFFS = ANCHOR_OFFS[np.argsort((ANCHOR_OFFS ** 2).sum(1), kind='stable')]

LO = np.array([GX0, GY0, GZ0])
HI = np.array([GX0 + NX * VS, GY0 + NY * VS, GZ0 + NZ * VS])


def _stencil(radius):
    r = int(np.ceil(radius / VS))
    d = np.arange(-r, r + 1)
    a, b, c = np.meshgrid(d, d, d, indexing='ij')
    sel = (a * a + b * b + c * c) * VS * VS <= radius * radius
    return np.stack([a[sel], b[sel], c[sel]], -1)


def _normalize(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def arrow_quat(d, side, roll):
    """Orientation with local X along d; local Y is `side` rotated by `roll` around d."""
    y = side * np.cos(roll)[:, None] + np.cross(d, side) * np.sin(roll)[:, None]
    return quat_from_basis(d, np.cross(d, y))


class RoundSim:
    def __init__(self, formation, seed=0, carve_radius=0.26, voxel_cost=2.2, bone_cost=6.0, embed_speed=3.0,
                 wound_radius=None, wound_delay=(0.1, 0.8), bone_hold=0.6, ground_embed=0.5, spray=1.0, severs=(),
                 debris_frac=1.0):
        """formation: dict with 'pos' (N,3) arrow centres at launch, 'vel' (N,3) launch velocities, 'launch' (N,)
        launch times, 'var' (N,) texture variants, 'hover' (N,) visible (hovering) before launch.
        severs: (time, part, push velocity, spin) - the part is shot off at that moment (see sever()).
        debris_frac: share of the flesh knocked out by arrows that flies out as visible debris (the rest is
        simply gone), so mass volleys don't bury him in a red waterfall."""
        self.severs = sorted(severs, key=lambda x: x[0])
        self.rng = np.random.default_rng(seed)
        self.g = Giant()
        self.dust = Dust(seed + 5)
        self.frame = 0
        self.t = 0.0
        self.dt_frame = 1.0 / FPS
        n = len(formation['pos'])
        self.n = n
        self.p = formation['pos'].astype(np.float64).copy()
        self.v0 = formation['vel'].astype(np.float64).copy()
        self.v = self.v0.copy()
        self.launch = formation['launch'].astype(np.float64)
        self.var = formation['var'].astype(np.float32)
        self.hover = formation.get('hover', np.zeros(n, bool)).astype(bool)
        self.state = np.full(n, WAIT, np.int8)
        side = np.cross(self.v0, UP)
        flat = np.linalg.norm(side, axis=1) < 1e-6
        side[flat] = (1.0, 0.0, 0.0)
        self.side = _normalize(side)
        self.roll = self.rng.uniform(-0.5, 0.5, n) + np.where(self.rng.random(n) < 0.5, 0.0, np.pi / 4)
        self.d = _normalize(self.v0)
        self.q = arrow_quat(self.d, self.side, self.roll)
        self.w = np.zeros((n, 3))
        self.anchor = np.full((n, 3), -1, np.int64)
        self.t_hit = np.full(n, -10.0)
        self.hit_any = np.zeros(n, bool)
        self.carrier = np.full(n, -1, np.int64)
        self.local_p = np.zeros((n, 3))
        self.local_q = np.zeros((n, 4))
        self.carve = _stencil(carve_radius)
        self.voxel_cost = voxel_cost
        self.bone_cost = bone_cost
        self.embed_speed = embed_speed
        self.ground_embed = ground_embed
        self.spray = spray
        self.debris_frac = debris_frac
        self.wound = _stencil(wound_radius) if wound_radius else None
        self.wound_d = (np.sqrt((self.wound.astype(np.float64) ** 2).sum(1)) * VS / wound_radius
                        if wound_radius else None)
        self.wound_delay = wound_delay
        self.bone_hold = bone_hold
        self.crack = np.full((NX, NY, NZ), np.inf)
        # debris
        self.dp = np.zeros((0, 3))
        self.dv = np.zeros((0, 3))
        self.dq = np.zeros((0, 4))
        self.dw = np.zeros((0, 3))
        self.dcol = np.zeros(0, VOXEL_DTYPE)
        self.drest = np.zeros(0, bool)
        self.dscale = np.zeros(0)
        # rigid chunks
        self.chunks = []
        self.chunk_counter = 0
        self.pile = np.zeros((PN, PN))
        self.destroyed = 0
        self.dirty = 0
        self.events = {}
        self.ground_hits = []

    # ------------------------------------------------------------------------------------------
    def _pile_idx(self, x, y):
        ix = np.clip(((x + PILE_HALF) / PILE_RES).astype(np.int64), 0, PN - 1)
        iy = np.clip(((y + PILE_HALF) / PILE_RES).astype(np.int64), 0, PN - 1)
        return ix, iy

    def ground_h(self, x, y):
        ix, iy = self._pile_idx(x, y)
        return self.pile[ix, iy]

    def _ev(self, name, n=1, pos=None):
        e = self.events.setdefault(name, [0, np.zeros(3), 0])
        e[0] += int(n)
        if pos is not None and n:
            p = np.asarray(pos).reshape(-1, 3)
            e[1] = e[1] + p.sum(0)
            e[2] += len(p)

    # ------------------------------------------------------------------------------------------
    def _add_debris(self, pos, vel, col, scale, spin=7.0):
        n = len(pos)
        if n == 0:
            return
        self.dp = np.concatenate([self.dp, pos])
        self.dv = np.concatenate([self.dv, vel])
        self.dq = np.concatenate([self.dq, np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))])
        self.dw = np.concatenate([self.dw, self.rng.normal(0, spin, (n, 3))])
        self.dcol = np.concatenate([self.dcol, col])
        self.drest = np.concatenate([self.drest, np.zeros(n, bool)])
        self.dscale = np.concatenate([self.dscale, np.full(n, scale) if np.isscalar(scale) else scale])

    def _destroy(self, i, j, k, vel, frac=1.0):
        """Remove voxels (unique indices) from the zombie; a share `frac` of them becomes debris with velocity
        vel."""
        if len(i) == 0:
            return
        g = self.g
        g.occ[i, j, k] = False
        self.destroyed += len(i)
        self.dirty += len(i)
        if frac < 1.0:
            keep = self.rng.random(len(i)) < frac
            i, j, k, vel = i[keep], j[keep], k[keep], vel[keep]
            if len(i) == 0:
                return
        col = np.zeros(len(i), VOXEL_DTYPE)
        col['cx'] = g.cx[i, j, k]
        col['cy'] = g.cy[i, j, k]
        col['cz'] = g.cz[i, j, k]
        col['inner'] = g.inner[i, j, k]
        col['cy'][:, 3] = 0x3F
        self._add_debris(Giant.index_to_world(i, j, k), vel, col, VS)

    def _wound(self, ii, jj, kk):
        """Mark voxels around impact points to crumble away after a short, distance-dependent delay."""
        if self.wound is None or len(ii) == 0:
            return
        wi = ii[:, None] + self.wound[None, :, 0]
        wj = jj[:, None] + self.wound[None, :, 1]
        wk = kk[:, None] + self.wound[None, :, 2]
        inb = (wi >= 0) & (wi < NX) & (wj >= 0) & (wj < NY) & (wk >= 0) & (wk < NZ)
        wi, wj, wk = wi[inb], wj[inb], wk[inb]
        dd = np.broadcast_to(self.wound_d[None, :], inb.shape)[inb]
        lo, hi = self.wound_delay
        delay = lo + (hi - lo) * dd * self.rng.uniform(0.6, 1.4, len(dd))
        keep = (dd < 0.7) | (self.rng.random(len(dd)) < 0.45)
        keep &= ~(self.g.bone[wi, wj, wk] & (self.rng.random(len(dd)) < self.bone_hold))
        wi, wj, wk, delay = wi[keep], wj[keep], wk[keep], delay[keep]
        np.minimum.at(self.crack, (wi, wj, wk), self.t + delay)

    # ------------------------------------------------------------------------------------------
    def _find_anchor(self, tip):
        """Nearest occupied voxel within 2 voxels of each tip; returns (ijk (n,3), found (n,))."""
        ti, tj, tk = Giant.world_to_index(tip)
        n = len(tip)
        out = np.full((n, 3), -1, np.int64)
        found = np.zeros(n, bool)
        for off in ANCHOR_OFFS:
            todo = ~found
            if not todo.any():
                break
            ci, cj, ck = ti[todo] + off[0], tj[todo] + off[1], tk[todo] + off[2]
            occ = self.g.occupied(ci, cj, ck)
            if occ.any():
                idx = np.nonzero(todo)[0][occ]
                out[idx] = np.stack([ci[occ], cj[occ], ck[occ]], -1)
                found[idx] = True
        return out, found

    def _stick(self, idx, pos, d):
        """Arrows idx stop in the zombie with centre pos and direction d."""
        self.p[idx] = pos
        self.d[idx] = d
        self.v[idx] = 0.0
        self.q[idx] = arrow_quat(d, self.side[idx], self.roll[idx])
        anc, found = self._find_anchor(pos + d * TIP)
        self.anchor[idx] = anc
        self.state[idx] = STUCK
        self.t_hit[idx] = self.t
        # nothing to hold on to (the flesh around the tip is already gone): it drops out right away
        lost = idx[~found]
        if len(lost):
            self._drop(lost)
        self._ev('stick', len(idx) - len(lost), pos)

    def _drop(self, idx):
        """Arrows fall out of the body (or off a shattered chunk)."""
        n = len(idx)
        self.state[idx] = FALL
        self.carrier[idx] = -1
        self.v[idx] = -self.d[idx] * self.rng.uniform(0.3, 1.5, (n, 1)) + self.rng.normal(0, 0.6, (n, 3))
        self.w[idx] = self.rng.normal(0, 3.0, (n, 3))
        self._ev('fall', n)

    # ------------------------------------------------------------------------------------------
    def _fly(self, dt):
        g = self.g
        rng = self.rng
        t = self.t
        go = (self.state == WAIT) & (self.launch <= t)
        if go.any():
            self.state[go] = FLY
            self._ev('launch', int(go.sum()), self.p[go][:: max(1, int(go.sum()) // 64)])
        fl = np.nonzero(self.state == FLY)[0]
        if len(fl) == 0:
            return
        p0 = self.p[fl]
        v = self.v[fl]
        step = v * dt + 0.5 * GRAV * dt * dt          # exact for constant gravity
        v_new = v + GRAV * dt
        d = _normalize(v_new)
        speed = np.linalg.norm(v_new, axis=1)
        frac = np.ones(len(fl))                        # how far along this step each arrow gets
        stopped = np.zeros(len(fl), bool)
        pad = TIP + 1.0
        near = np.all((p0 > LO - pad - np.abs(step)) & (p0 < HI + pad + np.abs(step)), axis=1)
        if near.any():
            ni = np.nonzero(near)[0]
            L = np.linalg.norm(step[ni], axis=1)
            nsub = max(1, int(np.ceil(L.max() / (VS * 0.5))))
            spd = speed[ni].copy()
            alive = np.ones(len(ni), bool)
            all_i, all_j, all_k, all_src = [], [], [], []
            first = np.zeros(len(ni), bool)
            for s in range(1, nsub + 1):
                f = s / nsub
                tip = p0[ni] + step[ni] * f + d[ni] * TIP
                ii, jj, kk = Giant.world_to_index(tip)
                occ = g.occupied(ii, jj, kk) & alive
                if not occ.any():
                    continue
                hs = np.nonzero(occ)[0]
                ci = ii[hs, None] + self.carve[None, :, 0]
                cj = jj[hs, None] + self.carve[None, :, 1]
                ck = kk[hs, None] + self.carve[None, :, 2]
                inb = (ci >= 0) & (ci < NX) & (cj >= 0) & (cj < NY) & (ck >= 0) & (ck < NZ)
                ci, cj, ck = np.where(inb, ci, 0), np.where(inb, cj, 0), np.where(inb, ck, 0)
                o = inb & g.occ[ci, cj, ck]
                nb = o & g.bone[ci, cj, ck]
                cost = o.sum(1) * self.voxel_cost + nb.sum(1) * (self.bone_cost - self.voxel_cost)
                all_i.append(ci[o])
                all_j.append(cj[o])
                all_k.append(ck[o])
                all_src.append(np.repeat(hs, o.sum(1)))
                g.occ[ci[o], cj[o], ck[o]] = False     # later samples see the channel
                newly = hs[~first[hs] & ~self.hit_any[fl[ni[hs]]]]
                first[newly] = True
                self._wound(ii[hs], jj[hs], kk[hs])
                spd[hs] -= cost
                stop = hs[spd[hs] < self.embed_speed]
                if len(stop):
                    alive[stop] = False
                    frac[ni[stop]] = f
                    stopped[ni[stop]] = True
            if all_i:
                ci = np.concatenate(all_i)
                cj = np.concatenate(all_j)
                ck = np.concatenate(all_k)
                src = np.concatenate(all_src)
                flat = np.ravel_multi_index((ci, cj, ck), (NX, NY, NZ))
                flat, fi = np.unique(flat, return_index=True)
                ci, cj, ck = np.unravel_index(flat, (NX, NY, NZ))
                sd = d[ni[src[fi]]]
                m = len(ci)
                # spray back out of the hole in a cone, then gravity
                back = -sd * rng.uniform(1.0, 6.0, (m, 1)) * self.spray
                perp = rng.normal(0, 1.8, (m, 3)) * self.spray
                perp -= sd * np.sum(perp * sd, 1, keepdims=True)
                vel = back + perp
                vel[:, 2] += rng.uniform(-0.5, 2.5, m)
                g.occ[ci, cj, ck] = True               # _destroy does the bookkeeping
                self._destroy(ci, cj, ck, vel, self.debris_frac)
                self._ev('carve', m, Giant.index_to_world(ci, cj, ck)[:: max(1, m // 48)])
                nbone = int(g.bone[ci, cj, ck].sum())
                if nbone:
                    self._ev('bone', nbone)
            if first.any():
                hi_ = fl[ni[first]]
                self.hit_any[hi_] = True
                self._ev('impact', len(hi_), (p0[ni[first]] + d[ni[first]] * TIP))
            keep = ~stopped[ni]
            v_new[ni[keep]] = d[ni[keep]] * np.maximum(spd[keep], 0.0)[:, None]
        newp = p0 + step * frac[:, None]
        self.p[fl] = newp
        self.v[fl] = v_new
        self.d[fl] = d
        if stopped.any():
            self._stick(fl[stopped], newp[stopped], d[stopped])
        # ground contact for arrows still flying
        fly = fl[~stopped]
        if len(fly):
            tip = self.p[fly] + self.d[fly] * TIP
            gh = self.ground_h(tip[:, 0], tip[:, 1])
            hit = tip[:, 2] < gh
            if hit.any():
                self._ground_stick(fly[hit], tip[hit], gh[hit])

    def _ground_stick(self, idx, tip, gh):
        d = self.d[idx]
        dz = d[:, 2]
        steep = dz < -0.12
        # arrows that come down steeply sink in at their angle
        s = idx[steep]
        if len(s):
            back = (gh[steep] - tip[steep, 2]) / np.maximum(-dz[steep], 1e-3)
            # entering a pile or a fallen piece through its side: stick where the tip is
            back = np.where(back > 0.8, 0.3, back)
            cross = tip[steep] - d[steep] * back[:, None]
            tip_final = cross + d[steep] * self.ground_embed
            self.p[s] = tip_final - d[steep] * TIP
            self.v[s] = 0.0
            self.q[s] = arrow_quat(d[steep], self.side[s], self.roll[s])
            self.state[s] = GROUND
            self.t_hit[s] = self.t
            soil = gh[steep] < 1.0
            if soil.any():
                self.ground_hits.append(cross[soil])
                self._ev('ground_hit', int(soil.sum()), cross[soil][:: max(1, int(soil.sum()) // 48)])
            if (~soil).any():
                self._ev('pile_hit', int((~soil).sum()), cross[~soil][:: max(1, int((~soil).sum()) // 48)])
        # grazing arrows skid and topple
        k = idx[~steep]
        if len(k):
            self.q[k] = arrow_quat(d[~steep], self.side[k], self.roll[k])
            self.v[k] *= 0.3
            self.v[k, 2] = np.abs(self.v[k, 2]) * 0.2
            self.p[k, 2] += gh[~steep] - tip[~steep, 2] + 0.05
            self.state[k] = FALL
            self.w[k] = self.rng.normal(0, 4.0, (len(k), 3))
            self._ev('ground_hit', len(k), tip[~steep])

    def _check_anchors(self):
        st = np.nonzero(self.state == STUCK)[0]
        if len(st) == 0:
            return
        a = self.anchor[st]
        ok = self.g.occupied(a[:, 0], a[:, 1], a[:, 2])
        lost = st[~ok]
        if len(lost) == 0:
            return
        anc, found = self._find_anchor(self.p[lost] + self.d[lost] * TIP)
        self.anchor[lost[found]] = anc[found]
        if (~found).any():
            self._drop(lost[~found])

    def _fall_step(self, dt):
        fa = np.nonzero(self.state == FALL)[0]
        if len(fa) == 0:
            return
        self.v[fa] += GRAV * dt
        self.v[fa] *= (1.0 - 0.2 * dt)
        newp = self.p[fa] + self.v[fa] * dt
        # bounce off what's left of the zombie
        ii, jj, kk = Giant.world_to_index(newp)
        inside = self.g.occupied(ii, jj, kk)
        if inside.any():
            ib = fa[inside]
            self.v[ib, :2] = self.v[ib, :2] * -0.3 + self.rng.normal(0, 0.5, (len(ib), 2))
            self.v[ib, 1] -= 1.0
            self.v[ib, 2] *= 0.2
            newp[inside] = self.p[ib] + self.v[ib] * dt
        self.p[fa] = newp
        self.q[fa] = integrate_quat(self.q[fa], self.w[fa], dt)
        dirs = quat_rotate(self.q[fa], np.tile([1.0, 0.0, 0.0], (len(fa), 1)))
        low = np.minimum(self.p[fa, 2] + dirs[:, 2] * TIP, self.p[fa, 2] - dirs[:, 2] * TIP)
        gh = self.ground_h(self.p[fa, 0], self.p[fa, 1])
        land = low < gh + 0.05
        if land.any():
            li = fa[land]
            n = len(li)
            dflat = dirs[land].copy()
            dflat[:, 2] = 0
            bad = np.linalg.norm(dflat, axis=1) < 1e-3
            rnd = self.rng.normal(0, 1, (n, 3))
            rnd[:, 2] = 0
            dflat[bad] = rnd[bad]
            dflat = _normalize(dflat)
            # lie flat, the crossed quads at +-45 degrees so neither plane stands up
            ang = np.pi / 4
            zax = UP * np.cos(ang) + np.cross(dflat, UP) * np.sin(ang)
            self.q[li] = quat_from_basis(dflat, zax)
            self.p[li, 2] = gh[land] + 0.06
            self.v[li] = 0.0
            self.state[li] = REST
            self._ev('arrow_land', n, self.p[li])

    # ------------------------------------------------------------------------------------------
    def _debris_step(self, dt):
        if len(self.dp) == 0:
            return
        rng = self.rng
        mv = ~self.drest
        if not mv.any():
            return
        idx = np.nonzero(mv)[0]
        v = self.dv[idx]
        v[:, 2] -= G * dt
        v *= (1 - 0.35 * dt)
        p = self.dp[idx] + v * dt
        ii, jj, kk = Giant.world_to_index(p)
        inside = self.g.occupied(ii, jj, kk)
        if inside.any():
            ib = np.nonzero(inside)[0]
            p[ib] = self.dp[idx[ib]]
            v[ib] = -v[ib] * 0.25 + rng.normal(0, 0.5, (len(ib), 3))
            v[ib, 1] -= 0.8
        gh = self.ground_h(p[:, 0], p[:, 1])
        half = self.dscale[idx] * 0.5
        below = p[:, 2] - half < gh
        if below.any():
            b = np.nonzero(below)[0]
            p[b, 2] = gh[b] + half[b]
            vz = v[b, 2]
            spd = np.linalg.norm(v[b], axis=1)
            fast = spd > 2.2
            self._ev('debris_ground', int(fast.sum()) + int((~fast).sum() * 0.3), p[b][:: max(1, len(b) // 32)])
            v[b, 2] = np.where(fast, -vz * 0.28, 0.0)
            v[b, :2] *= np.where(fast, 0.55, 0.0)[:, None]
            self.dw[idx[b]] *= 0.55
            settle = b[~fast]
            if len(settle):
                ix, iy = self._pile_idx(p[settle, 0], p[settle, 1])
                h0 = self.pile[ix, iy]
                best = h0.copy()
                bdx = np.zeros(len(settle))
                bdy = np.zeros(len(settle))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                    hn = self.pile[np.clip(ix + dx, 0, PN - 1), np.clip(iy + dy, 0, PN - 1)]
                    hn = hn + (0.07 if dx and dy else 0.0)
                    better = hn < best
                    best = np.where(better, hn, best)
                    bdx = np.where(better, dx, bdx)
                    bdy = np.where(better, dy, bdy)
                steep = (h0 - best) > 0.3
                if steep.any():
                    sl = settle[steep]
                    dnv = np.stack([bdx[steep], bdy[steep]], -1)
                    dnv /= np.linalg.norm(dnv, axis=1, keepdims=True)
                    v[sl, :2] = dnv * rng.uniform(1.4, 2.0, (len(sl), 1)) + rng.normal(0, 0.25, (len(sl), 2))
                    v[sl, 2] = 0.0
                    settle = settle[~steep]
            if len(settle):
                gi = idx[settle]
                self.drest[gi] = True
                v[settle] = 0.0
                yaw = rng.uniform(0, 2 * np.pi, len(settle))
                self.dq[gi] = axis_angle_quat(np.tile([0.0, 0, 1.0], (len(settle), 1)), yaw)
                ix, iy = self._pile_idx(p[settle, 0], p[settle, 1])
                top = self.pile[ix, iy] + self.dscale[gi] * 0.85
                p[settle, 2] = self.pile[ix, iy] + self.dscale[gi] * 0.5
                np.maximum.at(self.pile, (ix, iy), top)
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    jx = np.clip(ix + dx, 0, PN - 1)
                    jy = np.clip(iy + dy, 0, PN - 1)
                    np.maximum.at(self.pile, (jx, jy), top - self.dscale[gi] * 0.6)
        self.dv[idx] = v
        self.dp[idx] = p
        spin = ~self.drest[idx]
        if spin.any():
            si = idx[spin]
            self.dq[si] = integrate_quat(self.dq[si], self.dw[si], dt)

    # ------------------------------------------------------------------------------------------
    def _structure(self):
        """Delayed wound crumbling, loose voxels, and detaching pieces that lost their connection to the ground."""
        g = self.g
        due = g.occ & (self.crack <= self.t)
        if due.any():
            i, j, k = np.nonzero(due)
            n = len(i)
            vel = self.rng.normal(0, 1.0, (n, 3))
            vel[:, 1] -= self.rng.uniform(0.3, 2.5, n)
            vel[:, 2] = self.rng.uniform(-0.5, 1.5, n)
            self._destroy(i, j, k, vel, self.debris_frac)
            self._ev('wound', n, Giant.index_to_world(i, j, k)[:: max(1, n // 32)])
        self.crack[~g.occ] = np.inf
        o = g.occ
        nb = np.zeros(o.shape, np.int8)
        nb[1:] += o[:-1]
        nb[:-1] += o[1:]
        nb[:, 1:] += o[:, :-1]
        nb[:, :-1] += o[:, 1:]
        nb[:, :, 1:] += o[:, :, :-1]
        nb[:, :, :-1] += o[:, :, 1:]
        nb[:, :, 0] += 1        # the ground supports the bottom layer
        loose = o & ((nb <= 1) | ((nb == 2) & (self.rng.random(o.shape) < 0.35)))
        if loose.any():
            i, j, k = np.nonzero(loose)
            n = len(i)
            vel = self.rng.normal(0, 1.0, (n, 3))
            vel[:, 1] -= self.rng.uniform(0.2, 1.5, n)
            vel[:, 2] = self.rng.uniform(-1.0, 1.0, n)
            self._destroy(i, j, k, vel)
            self._ev('crumble', n)
        if self.dirty < 40:
            return
        self.dirty = 0
        lab, n = ndimage.label(g.occ)
        if n <= 1:
            return
        grounded = np.unique(lab[:, :, 0])
        grounded = grounded[grounded > 0]
        sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        for comp in range(1, n + 1):
            if comp in grounded:
                continue
            i, j, k = np.nonzero(lab == comp)
            if sizes[comp - 1] < 24:
                self._destroy(i, j, k, self.rng.normal(0, 1.0, (len(i), 3)))
                continue
            self._make_chunk(i, j, k)

    def _make_chunk(self, i, j, k, vel=None, w=None):
        g = self.g
        pos = Giant.index_to_world(i, j, k)
        com = pos.mean(0)
        col = np.zeros(len(i), VOXEL_DTYPE)
        col['cx'] = g.cx[i, j, k]
        col['cy'] = g.cy[i, j, k]
        col['cz'] = g.cz[i, j, k]
        col['inner'] = g.inner[i, j, k]
        occ_local = np.zeros((NX, NY, NZ), bool)
        occ_local[i, j, k] = True
        vis = np.zeros(len(i), np.uint8)
        for bit, (di, dj, dk) in enumerate(((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))):
            ni, nj, nk = i + di, j + dj, k + dk
            inb = (ni >= 0) & (ni < NX) & (nj >= 0) & (nj < NY) & (nk >= 0) & (nk < NZ)
            full = np.zeros(len(i), bool)
            full[inb] = occ_local[ni[inb], nj[inb], nk[inb]]
            vis |= (~full).astype(np.uint8) << bit
        col['cy'][:, 3] = vis
        g.occ[i, j, k] = False
        self.destroyed += len(i)
        cid = self.chunk_counter
        self.chunk_counter += 1
        # arrows stuck in this piece ride along with it
        st = np.nonzero(self.state == STUCK)[0]
        if len(st):
            a = self.anchor[st]
            ride = st[occ_local[a[:, 0], a[:, 1], a[:, 2]]]
            if len(ride):
                self.state[ride] = CARRIED
                self.carrier[ride] = cid
                self.local_p[ride] = self.p[ride] - com
                self.local_q[ride] = self.q[ride]
        if w is None:
            w = self.rng.normal(0, 0.35, 3)
            w[0] += self.rng.choice([-1, 1]) * 0.25
        vel = np.array([0.0, -0.6, 0.0]) if vel is None else np.array(vel, float)
        self.chunks.append({'id': cid, 'local': pos - com, 'col': col, 'pos': com.copy(), 'vel': vel,
                            'q': np.array([0.0, 0.0, 0.0, 1.0]), 'w': np.array(w, float), 'landed': False,
                            'rest': False, 'q_rest': None})
        self._ev('detach', len(i), com)

    def sever(self, part, push, spin):
        """Shoot a body part off: the joint where it meets the rest of him is blown out as debris and the part
        falls away as one rigid piece (arrows stuck in it ride along)."""
        from zombie import PARTS
        g = self.g
        x0, x1, y0, y1, z0, z1 = JOINTS[part]
        lo = np.array(Giant.world_to_index(np.array([x0, y0, z0]) + 1e-6))
        hi = np.array(Giant.world_to_index(np.array([x1, y1, z1]) - 1e-6))
        cut = np.zeros_like(g.occ)
        cut[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1] = True
        i, j, k = np.nonzero(cut & g.occ)
        push = np.asarray(push, float)
        if len(i):
            n = len(i)
            vel = self.rng.normal(0, 2.5, (n, 3)) + push * 0.5
            vel[:, 2] += self.rng.uniform(0.0, 2.0, n)
            self._destroy(i, j, k, vel)
            self._ev('sever', n, Giant.index_to_world(i, j, k).mean(0))
        lab, _ = ndimage.label(g.occ)
        grounded = np.unique(lab[:, :, 0])
        bx, by, bz, sx, sy, sz = PARTS[part]
        pl = np.array(Giant.world_to_index(np.array([bx + 0.5, by + 0.5, bz + 0.5])))
        ph = np.array(Giant.world_to_index(np.array([bx + sx - 0.5, by + sy - 0.5, bz + sz - 0.5])))
        sub = lab[pl[0]:ph[0] + 1, pl[1]:ph[1] + 1, pl[2]:ph[2] + 1]
        ids = sub[sub > 0]
        ids = ids[~np.isin(ids, grounded)]
        if len(ids) == 0:
            return
        comp = np.bincount(ids).argmax()
        ci, cj, ck = np.nonzero(lab == comp)
        self._make_chunk(ci, cj, ck, vel=push, w=spin)

    def _shatter(self, c, wp, rid):
        n = len(wp)
        r = wp - c['pos']
        v = c['vel'][None, :] + np.cross(c['w'][None, :], r)
        v = v * 0.35 + self.rng.normal(0, 2.2, (n, 3))
        v[:, 2] = np.abs(v[:, 2]) * 0.6 + self.rng.uniform(0, 3.0, n) * (self.rng.random(n) < 0.5)
        col = c['col'].copy()
        col['cy'][:, 3] = 0x3F
        self.dp = np.concatenate([self.dp, wp])
        self.dv = np.concatenate([self.dv, v])
        self.dq = np.concatenate([self.dq, np.tile(c['q'], (n, 1))])
        self.dw = np.concatenate([self.dw, self.rng.normal(0, 6.0, (n, 3))])
        self.dcol = np.concatenate([self.dcol, col])
        self.drest = np.concatenate([self.drest, np.zeros(n, bool)])
        self.dscale = np.concatenate([self.dscale, np.full(n, VS)])
        if len(rid):
            self._drop(rid)
            self.v[rid] += c['vel'] * 0.35
        self._ev('shatter', n, c['pos'])

    @staticmethod
    def _face_down(q, local):
        """The nearest stable resting orientation: one of the piece's broad faces flat on the ground (a long
        piece lies down rather than standing on its end), turned as little as possible from how it landed."""
        dirs = np.array([[1.0, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
        ext = np.ptp(local, axis=0)[[0, 0, 1, 1, 2, 2]]            # height when resting on that side
        axes = quat_rotate(np.tile(q, (6, 1)), dirs)
        ok = ext <= ext.min() * 1.25 + 1e-6
        a = axes[ok][np.argmin(axes[ok, 2])]
        down = np.array([0.0, 0.0, -1.0])
        ax = np.cross(a, down)
        sn = np.linalg.norm(ax)
        if sn < 1e-9:
            return q
        r = axis_angle_quat(ax / sn, np.arctan2(sn, float(np.dot(a, down))))
        out = quat_mul(r, q)
        return out / np.linalg.norm(out)

    def _chunks_step(self, dt):
        g = self.g
        keep = []
        for c in self.chunks:
            if c['rest']:
                keep.append(c)
                continue
            rid = np.nonzero((self.state == CARRIED) & (self.carrier == c['id']))[0]
            big = len(c['local']) >= CHUNK_KEEP
            pos0, q0 = c['pos'].copy(), c['q'].copy()
            c['vel'][2] -= G * dt
            c['pos'] = c['pos'] + c['vel'] * dt
            c['q'] = integrate_quat(c['q'][None], c['w'][None], dt)[0]
            if c['landed']:
                c['q'] = slerp(c['q'][None], c['q_rest'][None], min(1.0, 5.0 * dt))[0]
            wp = quat_rotate(np.tile(c['q'], (len(c['local']), 1)), c['local']) + c['pos']
            ii, jj, kk = Giant.world_to_index(wp)
            if g.occupied(ii, jj, kk).sum() > 6:
                if not big:
                    self._shatter(c, wp, rid)
                    continue
                # a big piece glances off what is still standing
                c['pos'], c['q'] = pos0, q0
                out = c['pos'][:2] / max(np.linalg.norm(c['pos'][:2]), 1e-6)
                c['vel'] = np.array([out[0] * 3.0, out[1] * 3.0, min(c['vel'][2], 0.0)])
                c['w'] *= 0.8
                wp = quat_rotate(np.tile(c['q'], (len(c['local']), 1)), c['local']) + c['pos']
            gh = self.ground_h(wp[:, 0], wp[:, 1])
            pen = float((gh + VS * 0.5 - wp[:, 2]).max())
            if pen > 0:
                if not big:
                    self._shatter(c, wp, rid)
                    continue
                c['pos'][2] += pen
                wp[:, 2] += pen
                if not c['landed']:
                    c['landed'] = True
                    c['q_rest'] = self._face_down(c['q'], c['local'])
                    self._ev('chunk_land', len(wp), c['pos'])
                if c['vel'][2] < 0:
                    c['vel'][2] = -c['vel'][2] * 0.25
                c['vel'][:2] *= 0.5
                c['w'] *= 0.5
            if c['landed']:
                c['vel'][:2] *= (1.0 - min(1.0, 2.0 * dt))
                ang = 2 * np.arccos(min(1.0, abs(float(np.dot(c['q'], c['q_rest'])))))
                if np.linalg.norm(c['vel']) < 0.8 and ang < 0.03:
                    c['q'] = c['q_rest']
                    c['vel'][:] = 0.0
                    c['w'][:] = 0.0
                    wp = quat_rotate(np.tile(c['q'], (len(c['local']), 1)), c['local']) + c['pos']
                    gh = self.ground_h(wp[:, 0], wp[:, 1])
                    c['pos'][2] += float((gh + VS * 0.5 - wp[:, 2]).max())
                    wp = quat_rotate(np.tile(c['q'], (len(c['local']), 1)), c['local']) + c['pos']
                    c['rest'] = True
                    # it is part of the ground now: debris piles up on it, arrows stick into it
                    ix, iy = self._pile_idx(wp[:, 0], wp[:, 1])
                    np.maximum.at(self.pile, (ix, iy), wp[:, 2] + VS * 0.5)
                    self._ev('chunk_rest', len(wp), c['pos'])
            if len(rid):
                qq = np.tile(c['q'], (len(rid), 1))
                self.p[rid] = quat_rotate(qq, self.local_p[rid]) + c['pos']
                self.q[rid] = quat_mul(qq, self.local_q[rid])
                self.d[rid] = quat_rotate(self.q[rid], np.tile([1.0, 0.0, 0.0], (len(rid), 1)))
            keep.append(c)
        self.chunks = keep

    # ------------------------------------------------------------------------------------------
    def step_frame(self, scale=1.0):
        """Advance one video frame; scale < 1 is slow motion."""
        self.events = {}
        self.ground_hits = []
        n_sub = max(1, int(np.ceil(SUB * scale - 1e-9)))
        dt = scale / FPS / n_sub
        for _ in range(n_sub):
            while self.severs and self.severs[0][0] <= self.t:
                _, part, push, spin = self.severs.pop(0)
                self.sever(part, push, spin)
            self._fly(dt)
            self._fall_step(dt)
            self._debris_step(dt)
            self._chunks_step(dt)
            self.t += dt
        self._structure()
        self._check_anchors()
        self.dust.step(np.concatenate(self.ground_hits) if self.ground_hits else np.zeros((0, 3)), scale / FPS)
        self.dt_frame = scale / FPS
        nfly = int((self.state == FLY).sum())
        if nfly:
            self._ev('flying', nfly, self.p[self.state == FLY][:: max(1, nfly // 64)])
        self.frame += 1
        return self.events

    # ------------------------------------------------------------------------------------------
    def arrow_instances(self):
        """(M, 10) float32 pos3 quat4 scale1 variant1 fade1 for every visible arrow, with a quiver after impact."""
        vis = (self.state != WAIT) | self.hover
        idx = np.nonzero(vis)[0]
        q = self.q[idx].copy()
        p = self.p[idx].copy()
        fl = self.state[idx] == FLY
        if fl.any():
            f = idx[fl]
            q[fl] = arrow_quat(self.d[f], self.side[f], self.roll[f])
        age = self.t - self.t_hit[idx]
        quiv = ((self.state[idx] == STUCK) | (self.state[idx] == GROUND)) & (age < 0.7)
        if quiv.any():
            k = idx[quiv]
            a = age[quiv]
            ang = 0.1 * np.exp(-a / 0.13) * np.sin(2 * np.pi * 13.0 * a)
            axis = self.side[k]
            dq = axis_angle_quat(axis, ang)
            tip = self.p[k] + self.d[k] * TIP
            p[quiv] = tip + quat_rotate(dq, self.p[k] - tip)       # pivot around the buried tip
            q[quiv] = quat_mul(dq, q[quiv])
        out = np.zeros((len(idx), 10), np.float32)
        out[:, 0:3] = p
        out[:, 3:7] = q
        out[:, 7] = 1.0
        out[:, 8] = self.var[idx]
        out[:, 9] = 1.0
        return out

    def streaks(self):
        """(K, 8) float32 head3 tail3 width alpha: short motion trails behind flying arrows."""
        f = np.nonzero(self.state == FLY)[0]
        if len(f) == 0:
            return np.zeros((0, 8), np.float32)
        spd = np.linalg.norm(self.v[f], axis=1)
        f = f[spd > 6.0]
        if len(f) == 0:
            return np.zeros((0, 8), np.float32)
        head = self.p[f] - self.d[f] * TIP * 0.8
        tail = head - self.v[f] * (0.5 * self.dt_frame)
        out = np.zeros((len(f), 8), np.float32)
        out[:, 0:3] = head
        out[:, 3:6] = tail
        out[:, 6] = 0.12
        out[:, 7] = 0.35
        return out

    def instances(self):
        """Voxel instances (zombie + chunks + debris), arrow instances and effects for the renderer."""
        from zombie import pack_static
        parts = [pack_static(self.g)]
        self.n_static = len(parts[0])      # the standing zombie comes first (the renderer tints it when hurt)
        for c in self.chunks:
            n = len(c['local'])
            inst = c['col'].copy()
            inst['pos'] = quat_rotate(np.tile(c['q'], (n, 1)), c['local']) + c['pos']
            inst['quat'] = c['q']
            inst['scale'] = VS
            parts.append(inst)
        if len(self.dp):
            inst = self.dcol.copy()
            inst['pos'] = self.dp
            inst['quat'] = self.dq
            inst['scale'] = self.dscale
            parts.append(inst)
        vox = np.concatenate(parts) if len(parts) > 1 else parts[0]
        fx = {'puffs': self.dust.puffs(), 'streaks': self.streaks()}
        return vox, self.arrow_instances(), fx


# ---------------------------------------------------------------------------------------------
# formations
# ---------------------------------------------------------------------------------------------
def ballistic_velocity(p0, target, T):
    """Launch velocity that reaches `target` from `p0` after time T under gravity (exact)."""
    return (target - p0) / T[:, None] - 0.5 * GRAV[None, :] * T[:, None]


def body_targets(n, rng, parts=None, miss_frac=0.0, miss_r=(3.0, 12.0), shrink=0.4, approach=None):
    """Random aim points inside the zombie's body parts (weighted by the part's front area), plus misses on the
    ground around it. With `approach` (horizontal unit vector from the zombie towards the archers) each aim point
    is moved out to where it enters its body part, so a volley's flight time is the time of first contact."""
    from zombie import PARTS
    names = list(PARTS) if parts is None else parts
    boxes = np.array([PARTS[k] for k in names], float)
    area = boxes[:, 3] * boxes[:, 5]
    choice = rng.choice(len(names), size=n, p=area / area.sum())
    b = boxes[choice]
    lo = b[:, :3] + shrink
    hi = b[:, :3] + b[:, 3:] - shrink
    pts = lo + (hi - lo) * rng.random((n, 3))
    if approach is not None:
        a = np.asarray(approach, float)[:2]
        a = a / np.linalg.norm(a)
        s = np.full(n, np.inf)
        for ax in range(2):
            if abs(a[ax]) > 1e-9:
                edge = b[:, ax] + (b[:, 3 + ax] if a[ax] > 0 else 0.0)
                s = np.minimum(s, (edge - pts[:, ax]) / a[ax])
        pts[:, :2] += a[None, :] * s[:, None]
    miss = rng.random(n) < miss_frac
    m = int(miss.sum())
    if m:
        ang = rng.uniform(0, 2 * np.pi, m)
        r = rng.uniform(*miss_r, m)
        pts[miss] = np.stack([r * np.cos(ang), r * np.sin(ang), np.zeros(m)], -1)
    return pts


def ground_targets(n, rng, r_max, r_min=0.0, center=(0.0, 0.0), az=(0.0, 360.0)):
    """Aim points on the ground in a disc, ring or sector (azimuth range in degrees) around `center`,
    area-uniform."""
    r = np.sqrt(rng.uniform(r_min ** 2, r_max ** 2, n))
    ang = np.radians(rng.uniform(az[0], az[1], n))
    return np.stack([center[0] + r * np.cos(ang), center[1] + r * np.sin(ang), np.zeros(n)], -1)


def volley(targets, origin, origin_spread, flight_time, launch, launch_spread, rng, time_spread=0.08):
    """Arrows fired from around `origin` (off screen) so that each one lands on its target."""
    n = len(targets)
    p0 = np.asarray(origin, float) + rng.normal(0, 1, (n, 3)) * np.asarray(origin_spread, float)
    T = flight_time * (1.0 + rng.uniform(-time_spread, time_spread, n))
    vel = ballistic_velocity(p0, targets, T)
    lt = launch + rng.uniform(0, launch_spread, n)
    return {'pos': p0, 'vel': vel, 'launch': lt, 'var': rng.integers(0, len(AR.FEATHERS), n),
            'hover': np.zeros(n, bool)}


def hover_line(starts, targets, speed, launch, launch_jitter, rng):
    """Arrows hovering in place (visible) and aimed at their targets; they fire at `launch`."""
    n = len(starts)
    dist = np.linalg.norm(targets - starts, axis=1)
    T = dist / speed
    vel = ballistic_velocity(starts, targets, T)
    lt = launch + rng.uniform(0, launch_jitter, n)
    return {'pos': starts.copy(), 'vel': vel, 'launch': lt, 'var': rng.integers(0, len(AR.FEATHERS), n),
            'hover': np.ones(n, bool)}


def contact_time(p0, v0, t_max=4.0, dt=1.0 / 240):
    """Flight time until each arrow's tip first touches the intact zombie (or the ground)."""
    g = Giant()
    p = np.asarray(p0, np.float64).copy()
    v = np.asarray(v0, np.float64).copy()
    out = np.full(len(p), t_max)
    alive = np.ones(len(p), bool)
    t = 0.0
    while alive.any() and t < t_max:
        a = np.nonzero(alive)[0]
        p[a] += v[a] * dt + 0.5 * GRAV * dt * dt
        v[a] += GRAV * dt
        t += dt
        tip = p[a] + _normalize(v[a]) * TIP
        ii, jj, kk = Giant.world_to_index(tip)
        hit = g.occupied(ii, jj, kk) | (tip[:, 2] < 0.0)
        out[a[hit]] = t
        alive[a[hit]] = False
    return out


def arrive_between(form, t0, t1, rng):
    """Re-time a volley so that every arrow makes first contact between t0 and t1 (simulation time)."""
    ct = contact_time(form['pos'], form['vel'])
    form['launch'] = np.maximum(0.0, rng.uniform(t0, t1, len(ct)) - ct)
    return form


def merge(*forms):
    return {k: np.concatenate([f[k] for f in forms]) for k in forms[0]}
