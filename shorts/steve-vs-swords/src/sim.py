"""Physics for one round: a sword formation hits the voxel giant.

Everything is vectorised numpy. Fixed sub-steps (SUB per video frame). Produces per-frame render
instances plus event counts that drive the sound design and the health bar.
"""
import numpy as np
from scipy import ndimage

from steve import Giant, VOXEL_DTYPE, VS, NX, NY, NZ, GX0, GY0, GZ0, K
from mathutil import quat_from_basis, integrate_quat, quat_rotate, slerp, axis_angle_quat, quat_mul
import swords as SW

FPS = 30
SUB = 4
DT = 1.0 / (FPS * SUB)
G = 17.0                        # gravity (units/s^2); slightly heavy so the giant still feels huge
TIP = SW.TIP_X - 0.05           # tip distance from sword centre along its axis
HILT = -0.95

# pile height field around the giant
PILE_HALF = 48.0
PILE_RES = 0.25
PN = int(2 * PILE_HALF / PILE_RES)

# sword states
WAIT, FLY, STUCK, FALL, REST = 0, 1, 2, 3, 4


def _stencil(radius):
    r = int(np.ceil(radius / VS))
    d = np.arange(-r, r + 1)
    a, b, c = np.meshgrid(d, d, d, indexing='ij')
    sel = (a * a + b * b + c * c) * VS * VS <= radius * radius
    return np.stack([a[sel], b[sel], c[sel]], -1)


class RoundSim:
    def __init__(self, formation, seed=0, carve_radius=0.34, speed=28.0, voxel_cost=0.9, bone_cost=2.4,
                 embed_speed=4.0, crumble=True, wound_radius=0.62, wound_delay=(0.12, 0.95), bone_hold=0.7,
                 loosen=True):
        """formation: dict with arrays 'pos' (N,3) start centres, 'dir' (N,3) unit flight dirs,
        'launch' (N,) launch times in seconds, 'mat' (N,) material index, optional 'speed' (N,)."""
        self.rng = np.random.default_rng(seed)
        self.g = Giant()
        self.frame = 0
        self.t = 0.0
        n = len(formation['pos'])
        self.n_sw = n
        self.sp = formation['pos'].astype(np.float64).copy()
        self.sdir = formation['dir'].astype(np.float64)
        self.sdir /= np.linalg.norm(self.sdir, axis=1, keepdims=True)
        spd = formation.get('speed', np.full(n, speed))
        self.sv = self.sdir * spd[:, None]
        self.sspeed = spd.astype(np.float64).copy()
        self.sq = quat_from_basis(self.sdir, np.array([0.0, 0.0, 1.0]))
        # tiny random roll so a big formation doesn't look computer-perfect
        roll = self.rng.normal(0, 0.05, n)
        self.sq = quat_mul(self.sq, axis_angle_quat(np.tile([1.0, 0, 0], (n, 1)), roll))
        self.sw = np.zeros((n, 3))
        self.state = np.zeros(n, np.int8)
        self.launch = formation['launch'].astype(np.float64)
        self.mat = formation['mat'].astype(np.float32)
        self.stuck_t = np.zeros(n)
        self.hit_any = np.zeros(n, bool)
        self.passed = np.zeros(n, bool)
        self.rest_q = np.zeros((n, 4))
        self.rest_blend = np.zeros(n)
        self.carve = _stencil(carve_radius)
        self.voxel_cost = voxel_cost
        self.bone_cost = bone_cost
        self.embed_speed = embed_speed
        self.crumble = crumble
        self.loosen = loosen
        self.wound = _stencil(wound_radius) if wound_radius else None
        self.wound_d = (np.sqrt((self.wound.astype(np.float64) ** 2).sum(1)) * VS / wound_radius
                        if wound_radius else None)
        self.wound_delay = wound_delay
        self.crack = np.full((NX, NY, NZ), np.inf)
        self.bone_hold = bone_hold
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
        self.pile = np.zeros((PN, PN))
        self.destroyed = 0
        self.bone_destroyed = 0
        self.dirty = 0
        self.events = {}
        self._last_label_frame = -99

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
            e[1] = e[1] + np.asarray(pos).reshape(-1, 3).sum(0)
            e[2] += len(np.asarray(pos).reshape(-1, 3))

    # ------------------------------------------------------------------------------------------
    def _spawn_debris(self, idx, vel, quat=None, scale=VS):
        i, j, k = idx
        n = len(i)
        if n == 0:
            return
        g = self.g
        pos = Giant.index_to_world(i, j, k)
        col = np.zeros(n, VOXEL_DTYPE)
        col['cx'] = g.cx[i, j, k]
        col['cy'] = g.cy[i, j, k]
        col['cz'] = g.cz[i, j, k]
        col['inner'] = g.inner[i, j, k]
        col['cy'][:, 3] = 0x3F
        self.dp = np.concatenate([self.dp, pos])
        self.dv = np.concatenate([self.dv, vel])
        q = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)) if quat is None else quat
        self.dq = np.concatenate([self.dq, q])
        self.dw = np.concatenate([self.dw, self.rng.normal(0, 7.0, (n, 3))])
        self.dcol = np.concatenate([self.dcol, col])
        self.drest = np.concatenate([self.drest, np.zeros(n, bool)])
        self.dscale = np.concatenate([self.dscale, np.full(n, scale)])

    def _destroy(self, i, j, k, vel):
        """Remove voxels (unique indices) from the giant and turn them into debris with velocity vel."""
        if len(i) == 0:
            return
        g = self.g
        self.destroyed += len(i)
        self.bone_destroyed += int(g.bone[i, j, k].sum())
        self._spawn_debris((i, j, k), vel)
        g.occ[i, j, k] = False
        self.dirty += len(i)

    # ------------------------------------------------------------------------------------------
    def _wound(self, ii, jj, kk):
        """Mark voxels around impact points to crumble away after a short, distance-dependent delay."""
        if self.wound is None:
            return
        wi = ii[:, None] + self.wound[None, :, 0]
        wj = jj[:, None] + self.wound[None, :, 1]
        wk = kk[:, None] + self.wound[None, :, 2]
        inb = (wi >= 0) & (wi < NX) & (wj >= 0) & (wj < NY) & (wk >= 0) & (wk < NZ)
        wi, wj, wk = wi[inb], wj[inb], wk[inb]
        dd = np.broadcast_to(self.wound_d[None, :], inb.shape)[inb]
        lo, hi = self.wound_delay
        delay = lo + (hi - lo) * dd * self.rng.uniform(0.6, 1.4, len(dd))
        # outer shell of the zone only sometimes breaks -> ragged edges
        keep = (dd < 0.7) | (self.rng.random(len(dd)) < 0.45)
        # bones are sturdier: most of them survive the crumbling so ribs/skull show through the wound
        keep &= ~(self.g.bone[wi, wj, wk] & (self.rng.random(len(dd)) < self.bone_hold))
        wi, wj, wk, delay = wi[keep], wj[keep], wk[keep], delay[keep]
        np.minimum.at(self.crack, (wi, wj, wk), self.t + delay)

    def _swords_step(self):
        g = self.g
        rng = self.rng
        t = self.t
        # launch
        go = (self.state == WAIT) & (self.launch <= t)
        self.state[go] = FLY
        # ---------------- flying (and ballistic after a hit)
        fl = np.nonzero(self.state == FLY)[0]
        if len(fl):
            v = self.sv[fl]
            passed = self.sp[fl, 1] > 2.6          # flew past the giant's front without stopping
            newly = passed & ~self.passed[fl]
            if newly.any():
                # punching through the debris cloud scatters them a little
                ni_ = fl[newly]
                v[newly, 0] += rng.normal(0, 2.6, len(ni_))
                v[newly, 2] += rng.normal(1.0, 1.4, len(ni_))
                self.passed[ni_] = True
            ballistic = self.hit_any[fl] | passed
            v[ballistic, 2] -= G * DT
            v[passed] *= (1.0 - 0.3 * DT)
            fresh = ~self.hit_any[fl] & ~passed
            if fresh.any():
                # ease from hovering to full speed over ~0.2 s instead of teleporting to full speed
                fi_ = fl[fresh]
                ramp = np.clip((t - self.launch[fi_]) / 0.2, 0.1, 1.0)
                v[fresh] = self.sdir[fi_] * (self.sspeed[fi_] * ramp)[:, None]
            if passed.any():
                # arrows-style: the blade follows its trajectory once it's past the giant
                pi_ = fl[passed]
                vd = v[passed] / np.maximum(np.linalg.norm(v[passed], axis=1, keepdims=True), 1e-9)
                self.sdir[pi_] = vd
                self.sq[pi_] = quat_from_basis(vd, np.array([0.0, 0.0, 1.0]))
            step = v * DT
            dist = np.linalg.norm(step, axis=1)
            nsub = max(1, int(np.ceil(dist.max() / (VS * 0.5))))
            p0 = self.sp[fl]
            d = self.sdir[fl]
            hit_mask = np.zeros(len(fl), bool)
            cost = np.zeros(len(fl))
            all_i, all_j, all_k, all_src = [], [], [], []
            for s in range(1, nsub + 1):
                tip = p0 + step * (s / nsub) + d * TIP
                ii, jj, kk = Giant.world_to_index(tip)
                occ = g.occupied(ii, jj, kk)
                if not occ.any():
                    continue
                hs = np.nonzero(occ)[0]
                hit_mask[hs] = True
                # carve a small sphere around each hitting tip
                ci = ii[hs, None] + self.carve[None, :, 0]
                cj = jj[hs, None] + self.carve[None, :, 1]
                ck = kk[hs, None] + self.carve[None, :, 2]
                inb = (ci >= 0) & (ci < NX) & (cj >= 0) & (cj < NY) & (ck >= 0) & (ck < NZ)
                ci = np.where(inb, ci, 0)
                cj = np.where(inb, cj, 0)
                ck = np.where(inb, ck, 0)
                o = inb & g.occ[ci, cj, ck]
                nb = o & g.bone[ci, cj, ck]
                cost[hs] += o.sum(1) * self.voxel_cost + nb.sum(1) * (self.bone_cost - self.voxel_cost)
                src = np.repeat(hs, o.sum(1))
                all_i.append(ci[o])
                all_j.append(cj[o])
                all_k.append(ck[o])
                all_src.append(src)
                self._wound(ii[hs], jj[hs], kk[hs])
                g.occ[ci[o], cj[o], ck[o]] = False   # remove now so later sub-samples see the hole
            if all_i:
                ci = np.concatenate(all_i)
                cj = np.concatenate(all_j)
                ck = np.concatenate(all_k)
                src = np.concatenate(all_src)
                # restore temporarily for colour lookup is not needed (colours live in separate arrays)
                flat = np.ravel_multi_index((ci, cj, ck), (NX, NY, NZ))
                flat, first = np.unique(flat, return_index=True)
                ci, cj, ck = np.unravel_index(flat, (NX, NY, NZ))
                src = src[first]
                sv = v[src]
                sd = d[src]
                n = len(ci)
                # spray: mostly back out of the wound with a wide cone, a bit forward, then gravity
                back = -sd * rng.uniform(1.5, 9.0, (n, 1))
                fwd = sd * rng.uniform(0.0, 3.0, (n, 1)) * (rng.random((n, 1)) < 0.35)
                perp = rng.normal(0, 3.2, (n, 3))
                perp -= sd * np.sum(perp * sd, 1, keepdims=True)
                up = np.zeros((n, 3))
                up[:, 2] = rng.uniform(-1.0, 3.5, n)
                vel = back + fwd + perp + up
                g.occ[ci, cj, ck] = True      # _destroy expects them occupied (bookkeeping)
                self._destroy(ci, cj, ck, vel)
                self._ev('carve', n, Giant.index_to_world(ci, cj, ck)[:: max(1, n // 64)])
                bn = int(self.g.bone[ci, cj, ck].sum())
                if bn:
                    self._ev('bone', bn)
            # energy loss & state changes
            spd = np.linalg.norm(v, axis=1)
            new_spd = np.maximum(spd - cost, 0.0)
            first_hit = hit_mask & ~self.hit_any[fl]
            if first_hit.any():
                self._ev('impact', first_hit.sum(), (p0 + d * TIP)[first_hit])
            scale = np.where(spd > 1e-6, new_spd / np.maximum(spd, 1e-6), 0.0)
            v = v * scale[:, None]
            # deflect a little on impact so swords don't stay perfectly aligned
            if hit_mask.any():
                hm = np.nonzero(hit_mask)[0]
                self.sw[fl[hm]] += rng.normal(0, 1.2, (len(hm), 3)) * (cost[hm, None] > 0.5)
            self.hit_any[fl] |= hit_mask
            self.sv[fl] = v
            self.sp[fl] = p0 + v * DT
            # orientation follows angular velocity only after a hit
            hs = self.hit_any[fl]
            if hs.any():
                idx = fl[hs]
                self.sq[idx] = integrate_quat(self.sq[idx], self.sw[idx] * 0.35, DT)
                # keep flight direction consistent with the (slightly rotated) sword axis
                self.sdir[idx] = quat_rotate(self.sq[idx], np.tile([1.0, 0.0, 0.0], (len(idx), 1)))
            stuck = (new_spd < self.embed_speed) & hit_mask
            if stuck.any():
                s_idx = fl[stuck]
                self.state[s_idx] = STUCK
                self.sv[s_idx] = 0.0
                self.stuck_t[s_idx] = self.t
                self._ev('stick', len(s_idx))
            # ground contact while flying (swords that passed through / missed)
            self._ground_contact(fl[self.state[fl] == FLY])
        # ---------------- stuck: fall when the flesh around the blade is gone
        st = np.nonzero(self.state == STUCK)[0]
        if len(st) and (self.frame % 2 == 0):
            d = quat_rotate(self.sq[st], np.tile([1.0, 0, 0], (len(st), 1)))
            support = np.zeros(len(st), int)
            for f in (TIP, TIP - 0.25, TIP - 0.5, TIP - 0.75):
                p = self.sp[st] + d * f
                ii, jj, kk = Giant.world_to_index(p)
                support += g.occupied(ii, jj, kk)
                for off in ((1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)):
                    support += g.occupied(ii + off[0], jj + off[1], kk + off[2])
            age = self.t - self.stuck_t[st]
            drop = (support < 3) & (age > 0.05)
            # swords stuck in thin remains wobble loose after a while
            if self.loosen:
                drop |= (support < 9) & (age > rng.uniform(0.6, 2.5, len(st)))
            if drop.any():
                di = st[drop]
                self.state[di] = FALL
                n = len(di)
                back = -d[drop]
                self.sv[di] = back * rng.uniform(0.5, 2.5, (n, 1)) + rng.normal(0, 0.8, (n, 3))
                self.sw[di] = rng.normal(0, 2.5, (n, 3))
        # ---------------- falling
        fa = np.nonzero(self.state == FALL)[0]
        if len(fa):
            self.sv[fa, 2] -= G * DT
            self.sv[fa] *= (1 - 0.15 * DT)
            newp = self.sp[fa] + self.sv[fa] * DT
            # don't fall through what's left of the giant: probe the centre and both ends
            d = quat_rotate(self.sq[fa], np.tile([1.0, 0, 0], (len(fa), 1)))
            inside = np.zeros(len(fa), bool)
            for f in (0.0, TIP, HILT):
                ii, jj, kk = Giant.world_to_index(newp + d * f)
                inside |= g.occupied(ii, jj, kk)
            if inside.any():
                ib = np.nonzero(inside)[0]
                self.sv[fa[ib], :2] = self.sv[fa[ib], :2] * -0.3 + self.rng.normal(0, 0.6, (len(ib), 2))
                self.sv[fa[ib], 1] -= 1.2        # nudge towards the front, out of the body
                self.sv[fa[ib], 2] *= 0.2
                newp[ib] = self.sp[fa[ib]] + self.sv[fa[ib]] * DT
            self.sp[fa] = newp
            self.sq[fa] = integrate_quat(self.sq[fa], self.sw[fa], DT)
            self._ground_contact(fa)
        # ---------------- resting: ease into lying flat
        rs = np.nonzero((self.state == REST) & (self.rest_blend < 1.0))[0]
        if len(rs):
            self.rest_blend[rs] = np.minimum(1.0, self.rest_blend[rs] + DT * 6.0)
            self.sq[rs] = slerp(self.sq[rs], self.rest_q[rs], np.full(len(rs), 0.25))

    def _ground_contact(self, idx):
        if len(idx) == 0:
            return
        rng = self.rng
        d = quat_rotate(self.sq[idx], np.tile([1.0, 0, 0], (len(idx), 1)))
        lowest = np.minimum(self.sp[idx, 2] + d[:, 2] * TIP, self.sp[idx, 2] + d[:, 2] * HILT)
        gh = self.ground_h(self.sp[idx, 0], self.sp[idx, 1])
        hit = lowest < gh + 0.06
        if not hit.any():
            return
        h, dh, ghh, low = idx[hit], d[hit], gh[hit], lowest[hit]
        v = self.sv[h]
        spd = np.linalg.norm(v, axis=1)
        self._ev('sword_ground', len(h), self.sp[h])
        # fast swords arriving point-first bury the tip in the ground and stay there, like arrows
        spear = (spd > 6.0) & (dh[:, 2] < -0.3) & (self.state[h] == FLY)
        if spear.any():
            si = h[spear]
            tip_z = self.sp[si, 2] + dh[spear, 2] * TIP
            self.sp[si, 2] += (ghh[spear] - 0.35) - tip_z
            self.sv[si] = 0.0
            self.sw[si] = 0.0
            self.state[si] = REST
            self.rest_blend[si] = 1.0
            self._ev('sword_stab_ground', len(si), self.sp[si])
        o = ~spear
        h, v, spd, ghh, low = h[o], v[o], spd[o], ghh[o], low[o]
        if len(h) == 0:
            return
        self.sp[h, 2] += ghh + 0.06 - low          # lift out of the ground
        bounce = spd > 3.0
        vb = v.copy()
        vb[:, 2] = np.abs(v[:, 2]) * 0.22
        vb[:, :2] *= 0.5
        self.sv[h] = np.where(bounce[:, None], vb, 0.0)
        self.sw[h] = np.where(bounce[:, None], self.sw[h] * 0.5 + rng.normal(0, 3, (len(h), 3)), 0.0)
        # anything still marked as flying is now a tumbling body
        self.state[h[bounce]] = np.where(self.state[h[bounce]] == FLY, FALL, self.state[h[bounce]])
        self._settle_swords(h[~bounce])

    def _settle_swords(self, settle):
        rng = self.rng
        if len(settle):
            self.state[settle] = REST
            # target orientation: blade horizontal, sprite plane flat on the ground, random yaw
            d0 = quat_rotate(self.sq[settle], np.tile([1.0, 0, 0], (len(settle), 1)))
            d0[:, 2] = 0
            nrm = np.linalg.norm(d0, axis=1, keepdims=True)
            rnd = rng.normal(0, 1, (len(settle), 3))
            rnd[:, 2] = 0
            d0 = np.where(nrm > 1e-3, d0 / np.maximum(nrm, 1e-9), rnd / np.linalg.norm(rnd, axis=1, keepdims=True))
            flip = np.where(rng.random(len(settle)) < 0.5, 1.0, -1.0)
            up = np.array([0.0, 0.0, 1.0])
            # local Z (in sprite plane) -> horizontal perpendicular, local Y (sprite normal) -> +/- up
            zax = np.cross(up, d0) * flip[:, None]
            self.rest_q[settle] = quat_from_basis(d0, zax)
            self.rest_blend[settle] = 0.0
            gz = self.ground_h(self.sp[settle, 0], self.sp[settle, 1])
            self.sp[settle, 2] = gz + SW.PX * 0.5 + 0.01
            # swords add a little to the pile so debris lands on top of them
            ix, iy = self._pile_idx(self.sp[settle, 0], self.sp[settle, 1])
            np.add.at(self.pile, (ix, iy), 0.03)

    # ------------------------------------------------------------------------------------------
    def _debris_step(self):
        if len(self.dp) == 0:
            return
        rng = self.rng
        mv = ~self.drest
        if not mv.any():
            return
        idx = np.nonzero(mv)[0]
        v = self.dv[idx]
        v[:, 2] -= G * DT
        v *= (1 - 0.35 * DT)
        p = self.dp[idx] + v * DT
        # collide with the giant (simple: stop and reflect when entering an occupied voxel)
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
                # granular sliding: if a neighbouring cell of the pile is much lower, slide towards it
                ix, iy = self._pile_idx(p[settle, 0], p[settle, 1])
                h0 = self.pile[ix, iy]
                best = h0.copy()
                bdx = np.zeros(len(settle))
                bdy = np.zeros(len(settle))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                    hn = self.pile[np.clip(ix + dx, 0, PN - 1), np.clip(iy + dy, 0, PN - 1)]
                    hn = hn + (0.07 if dx and dy else 0.0)       # diagonals are further away
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
                # land flat with random yaw
                yaw = rng.uniform(0, 2 * np.pi, len(settle))
                self.dq[gi] = axis_angle_quat(np.tile([0.0, 0, 1.0], (len(settle), 1)), yaw)
                # stack onto the pile: cube occupies its cell; neighbours get a little too (mound)
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
            self.dq[si] = integrate_quat(self.dq[si], self.dw[si], DT)

    # ------------------------------------------------------------------------------------------
    def _structure(self):
        """Detach pieces that lost their connection to the ground; crumble loose voxels."""
        g = self.g
        due = g.occ & (self.crack <= self.t)
        if due.any():
            i, j, k = np.nonzero(due)
            n = len(i)
            vel = self.rng.normal(0, 1.0, (n, 3))
            vel[:, 1] -= self.rng.uniform(0.5, 3.5, n)
            vel[:, 2] = self.rng.uniform(-0.5, 2.0, n)
            self._destroy(i, j, k, vel)
            self._ev('wound', n, Giant.index_to_world(i, j, k)[:: max(1, n // 32)])
        self.crack[~g.occ] = np.inf
        if self.crumble:
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
                vel = self.rng.normal(0, 1.2, (n, 3))
                vel[:, 1] -= self.rng.uniform(0.3, 2.0, n)
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
            sel = lab == comp
            i, j, k = np.nonzero(sel)
            if sizes[comp - 1] < 24:
                vel = self.rng.normal(0, 1.0, (len(i), 3))
                self._destroy(i, j, k, vel)
                continue
            self._make_chunk(i, j, k)

    def _make_chunk(self, i, j, k):
        g = self.g
        pos = Giant.index_to_world(i, j, k)
        com = pos.mean(0)
        col = np.zeros(len(i), VOXEL_DTYPE)
        col['cx'] = g.cx[i, j, k]
        col['cy'] = g.cy[i, j, k]
        col['cz'] = g.cz[i, j, k]
        col['inner'] = g.inner[i, j, k]
        # visibility inside the chunk: faces towards other chunk voxels stay hidden
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
        self.bone_destroyed += int(g.bone[i, j, k].sum())
        # tip over away from where it was attached: small random spin, slightly towards the front
        w = self.rng.normal(0, 0.35, 3)
        w[0] += self.rng.choice([-1, 1]) * 0.25
        self.chunks.append({
            'local': pos - com, 'col': col, 'pos': com.copy(), 'vel': np.array([0.0, -0.6, 0.0]),
            'q': np.array([0.0, 0.0, 0.0, 1.0]), 'w': w, 'bone': g.bone[i, j, k].copy(),
            'visible_all': col['cy'][:, 3] == 0x3F,
        })
        self._ev('detach', len(i), com)

    def _chunks_step(self):
        g = self.g
        keep = []
        for c in self.chunks:
            c['vel'][2] -= G * DT
            c['pos'] = c['pos'] + c['vel'] * DT
            c['q'] = integrate_quat(c['q'][None], c['w'][None], DT)[0]
            wp = quat_rotate(np.tile(c['q'], (len(c['local']), 1)), c['local']) + c['pos']
            gh = self.ground_h(wp[:, 0], wp[:, 1])
            hit_ground = (wp[:, 2] - VS * 0.5 < gh).any()
            ii, jj, kk = Giant.world_to_index(wp)
            hit_body = g.occupied(ii, jj, kk).sum() > 6
            if hit_ground or hit_body:
                # shatter: every voxel becomes debris carrying the chunk's motion
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
                self._ev('shatter', n, c['pos'])
            else:
                keep.append(c)
        self.chunks = keep

    # ------------------------------------------------------------------------------------------
    def step_frame(self):
        self.events = {}
        for s in range(SUB):
            self._swords_step()
            self._debris_step()
            self._chunks_step()
            self.t += DT
        self._structure()
        self.frame += 1
        return self.events

    # ------------------------------------------------------------------------------------------
    def instances(self):
        """Voxel instances (static + chunks + debris) and sword instances for the renderer."""
        from steve import pack_static
        parts = [pack_static(self.g)]
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
        live = np.ones(self.n_sw, bool)          # waiting swords hover in formation, so they are drawn too
        sw = np.zeros((int(live.sum()), 9), np.float32)
        sw[:, 0:3] = self.sp[live]
        sw[:, 3:7] = self.sq[live]
        sw[:, 7] = 1.0
        sw[:, 8] = self.mat[live]
        return vox, sw


# ---------------------------------------------------------------------------------------------
# formations
# ---------------------------------------------------------------------------------------------
def grid_formation(nx, nz, x_range, z_range, y_start, layers=1, layer_gap=1.6, dir=(0.0, 1.0, 0.0),
                   launch=0.0, launch_jitter=0.02, layer_delay=0.0, seed=0, mats=None, jitter=0.03,
                   speed=28.0):
    rng = np.random.default_rng(seed)
    xs = np.linspace(x_range[0], x_range[1], nx) if nx > 1 else np.array([np.mean(x_range)])
    zs = np.linspace(z_range[0], z_range[1], nz) if nz > 1 else np.array([np.mean(z_range)])
    X, Z, Lyr = np.meshgrid(xs, zs, np.arange(layers), indexing='ij')
    X, Z, Lyr = X.ravel(), Z.ravel(), Lyr.ravel()
    n = len(X)
    d = np.asarray(dir, float)
    d = d / np.linalg.norm(d)
    pos = np.stack([X, np.full(n, y_start), Z], -1)
    pos[:, 1] -= Lyr * layer_gap
    pos += rng.normal(0, jitter, (n, 3))
    lt = launch + rng.uniform(0, launch_jitter, n) + Lyr * layer_delay
    if mats is None:
        # wood, stone, iron, gold, diamond, netherite: favour the shiny, colourful tiers
        mats = rng.choice(len(SW.MATERIALS), size=n, p=[0.07, 0.07, 0.22, 0.24, 0.30, 0.10])
    return {'pos': pos, 'dir': np.tile(d, (n, 1)), 'launch': lt, 'mat': np.asarray(mats), 'speed': np.full(n, speed)}


def silhouette_mask(x, z, margin=0.45):
    """True where (x, z) lies in front of the giant (legs, torso + arms, head), expanded by margin."""
    legs = (np.abs(x) <= 4 + margin) & (z >= 0.35) & (z <= 12)
    torso = (np.abs(x) <= 8 + margin) & (z >= 12) & (z <= 24)
    head = (np.abs(x) <= 4 + margin) & (z >= 24) & (z <= 32 + margin * 0.3)
    return legs | torso | head


def silhouette_formation(per_layer, layers, y_start, layer_gap=1.9, launch=0.45, launch_jitter=0.08,
                         seed=0, speed=28.0, jitter=0.05):
    """Exactly per_layer * layers swords, packed on a hex-ish grid covering the giant's silhouette."""
    rng = np.random.default_rng(seed)
    pts_all = []
    for layer in range(layers):
        ox, oz = rng.uniform(0, 1, 2)
        lo, hi = 0.2, 2.0
        for _ in range(40):                      # bisection on spacing so we get at least per_layer points
            s = 0.5 * (lo + hi)
            zs = np.arange(0.35 + oz * s * 0.5, 32.2, s * 0.866)
            pts = []
            for r, z in enumerate(zs):
                xs = np.arange(-8.45 + ((r % 2) * 0.5 + ox) * s, 8.46, s)
                pts.append(np.stack([xs, np.full(len(xs), z)], -1))
            pts = np.concatenate(pts)
            pts = pts[silhouette_mask(pts[:, 0], pts[:, 1])]
            if len(pts) >= per_layer:
                lo = s
            else:
                hi = s
        s = lo
        zs = np.arange(0.35 + oz * s * 0.5, 32.2, s * 0.866)
        pts = []
        for r, z in enumerate(zs):
            xs = np.arange(-8.45 + ((r % 2) * 0.5 + ox) * s, 8.46, s)
            pts.append(np.stack([xs, np.full(len(xs), z)], -1))
        pts = np.concatenate(pts)
        pts = pts[silhouette_mask(pts[:, 0], pts[:, 1])]
        pick = rng.choice(len(pts), per_layer, replace=False)
        p = pts[np.sort(pick)]
        pos = np.stack([p[:, 0], np.full(per_layer, y_start - layer * layer_gap), p[:, 1]], -1)
        pos += rng.normal(0, jitter, pos.shape)
        pts_all.append(pos)
    pos = np.concatenate(pts_all)
    n = len(pos)
    assert n == per_layer * layers
    lt = launch + rng.uniform(0, launch_jitter, n)
    mats = rng.choice(len(SW.MATERIALS), size=n, p=[0.07, 0.07, 0.22, 0.24, 0.30, 0.10])
    return {'pos': pos, 'dir': np.tile([0.0, 1.0, 0.0], (n, 1)), 'launch': lt, 'mat': mats,
            'speed': np.full(n, speed)}
