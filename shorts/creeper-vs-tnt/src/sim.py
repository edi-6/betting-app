"""Physics for one round: a formation of primed TNT flies into the voxel creeper and explodes.

Everything is vectorised numpy with fixed sub-steps (SUB per video frame). TNT explodes on contact with the
creeper or the ground, or in mid-air when its fuse runs out after flying past. Each blast removes a ragged
sphere of voxels (flung outward through the crater), craters the ground (dirt clods), and is handed to the
VFX system. The step returns per-frame event counts that drive the sound design and the health bar.
"""
import numpy as np
from scipy import ndimage

from creeper import Giant, VOXEL_DTYPE, VS, NX, NY, NZ, GX0, GY0, GZ0
from mathutil import integrate_quat, quat_rotate, axis_angle_quat
from ground import Ground, REG, block_colours
from vfx import VFX

FPS = 30
SUB = 4
DT = 1.0 / (FPS * SUB)
G = 17.0                      # gravity (units/s^2)
R_EXP = 1.7                   # blast radius inside the creeper (world units)
R_CRATER = 1.6                # blast radius in the ground (blocks)
FLASH_PERIOD = 0.25           # primed TNT toggles its white flash every 5 game ticks
BACK_Y = 7.0                  # past this the TNT has flown by the creeper
PUSH_R = 3.5

# pile height field around the creeper (absolute heights; craters lower it)
PILE_HALF = 48.0
PILE_RES = 0.25
PN = int(2 * PILE_HALF / PILE_RES)

WAIT, FLY, GONE = 0, 1, 2

_c = np.array([-0.42, 0.42])
CONTACT_OFFS = np.array([[x, y, z] for x in _c for y in _c for z in _c] +
                        [[0.5, 0, 0], [-0.5, 0, 0], [0, 0.5, 0], [0, -0.5, 0], [0, 0, 0.5], [0, 0, -0.5],
                         [0, 0, 0]])


def _stencil(radius):
    r = int(np.ceil(radius / VS))
    d = np.arange(-r, r + 1)
    a, b, c = np.meshgrid(d, d, d, indexing='ij')
    dist = np.sqrt(a * a + b * b + c * c) * VS
    sel = dist <= radius
    return np.stack([a[sel], b[sel], c[sel]], -1), dist[sel]


class RoundSim:
    def __init__(self, formation, seed=0, blast=R_EXP, crater=R_CRATER, push=4.0, debris_speed=(3.5, 10.0),
                 ground_debris_budget=18000, vaporize=0.5):
        """formation: dict with 'pos' (N,3) start centres, 'dir' (N,3) flight dirs, 'launch' (N,) launch times,
        'speed' (N,), 'fuse' (N,) absolute explosion times if nothing is hit first."""
        self.rng = np.random.default_rng(seed)
        self.g = Giant()
        self.ground = Ground()
        self.vfx = VFX(seed + 7)
        self.frame = 0
        self.t = 0.0
        n = len(formation['pos'])
        self.n_tnt = n
        self.tp = formation['pos'].astype(np.float64).copy()
        d = formation['dir'].astype(np.float64)
        self.tdir = d / np.linalg.norm(d, axis=1, keepdims=True)
        self.tspeed = formation['speed'].astype(np.float64)
        self.tv = np.zeros((n, 3))
        self.tstate = np.zeros(n, np.int8)
        self.tlaunch = formation['launch'].astype(np.float64)
        self.tfuse = formation['fuse'].astype(np.float64)
        self.tball = np.zeros(n, bool)            # ballistic (gravity) once pushed or past the creeper
        self.tphase = self.rng.uniform(-0.05, 0.05, n)        # flashes spread over ~3 frames, no hard strobe
        # big formations flash more subtly (a 10,000-block wall toggling hard would strobe the whole screen)
        self.flash_amp = 0.3 / (1.0 + 0.5 * np.log10(max(n, 1)))
        self.blast = blast
        self.crater = crater
        self.push = push
        self.debris_speed = debris_speed
        self.vaporize = vaporize                 # share of blasted voxels that are pulverised (no debris)
        self.stencil, self.stencil_d = _stencil(blast * 1.12)
        # debris
        self.dp = np.zeros((0, 3))
        self.dv = np.zeros((0, 3))
        self.dq = np.zeros((0, 4))
        self.dw = np.zeros((0, 3))
        self.dcol = np.zeros(0, VOXEL_DTYPE)
        self.drest = np.zeros(0, bool)
        self.dscale = np.zeros(0)
        self.ground_budget = ground_debris_budget
        self.chunks = []
        self.pile = np.zeros((PN, PN))
        self.destroyed = 0
        self.dirty = 0
        self.events = {}
        self.frame_explosions = []

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

    def _destroy(self, i, j, k, vel):
        """Remove voxels (unique indices) from the creeper and turn them into debris with velocity vel."""
        if len(i) == 0:
            return
        g = self.g
        col = np.zeros(len(i), VOXEL_DTYPE)
        col['cx'] = g.cx[i, j, k]
        col['cy'] = g.cy[i, j, k]
        col['cz'] = g.cz[i, j, k]
        col['inner'] = g.inner[i, j, k]
        col['cy'][:, 3] = 0x3F
        self._add_debris(Giant.index_to_world(i, j, k), vel, col, VS)
        self.destroyed += len(i)
        g.occ[i, j, k] = False
        self.dirty += len(i)

    # ------------------------------------------------------------------------------------------
    def _explode(self, C):
        """Detonate TNT at centres C (E, 3)."""
        rng = self.rng
        g = self.g
        E = len(C)
        if E == 0:
            return
        self.frame_explosions.append(C.copy())
        self._ev('explode', E, C)
        # --- 1. blast the creeper -----------------------------------------------------------------
        R = self.blast
        lo = np.array([GX0, GY0, GZ0]) - R * 1.2
        hi = np.array([GX0 + NX * VS, GY0 + NY * VS, GZ0 + NZ * VS]) + R * 1.2
        near = np.all((C > lo) & (C < hi), axis=1)
        if near.any():
            Cn = C[near]
            ic, jc, kc = Giant.world_to_index(Cn)
            S = self.stencil
            ci = ic[:, None] + S[None, :, 0]
            cj = jc[:, None] + S[None, :, 1]
            ck = kc[:, None] + S[None, :, 2]
            inb = (ci >= 0) & (ci < NX) & (cj >= 0) & (cj < NY) & (ck >= 0) & (ck < NZ)
            thr = R * rng.uniform(0.8, 1.12, inb.shape)                 # ragged crater edges
            ci0, cj0, ck0 = np.where(inb, ci, 0), np.where(inb, cj, 0), np.where(inb, ck, 0)
            hit = inb & (self.stencil_d[None, :] <= thr) & g.occ[ci0, cj0, ck0]
            if hit.any():
                src = np.nonzero(hit)[0]
                vi, vj, vk = ci0[hit], cj0[hit], ck0[hit]
                flat = np.ravel_multi_index((vi, vj, vk), (NX, NY, NZ))
                flat, first = np.unique(flat, return_index=True)
                vi, vj, vk = np.unravel_index(flat, (NX, NY, NZ))
                src = src[first]
                p = Giant.index_to_world(vi, vj, vk)
                c = Cn[src]
                n = len(vi)
                away = p - c
                dist = np.linalg.norm(away, axis=1)
                away /= np.maximum(dist, 1e-6)[:, None]
                # material is thrown back out of the crater, away from the creeper's core
                axis = np.stack([np.zeros(n), np.zeros(n), p[:, 2]], -1)
                out = c - axis
                out[:, 2] *= 0.3
                out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-6)
                dirv = out * 1.0 + away * 0.35 + rng.normal(0, 0.5, (n, 3))
                dirv[:, 2] += 0.45                       # spall arcs up and out rather than straight at you
                dirv /= np.maximum(np.linalg.norm(dirv, axis=1, keepdims=True), 1e-6)
                lo_s, hi_s = self.debris_speed
                spd = rng.uniform(lo_s, hi_s, n) * np.sqrt(np.clip(1.2 - dist / R, 0.15, 1.0))
                vel = dirv * spd[:, None]
                vel[:, 2] += rng.uniform(0.5, 3.5, n)
                gone = rng.random(n) < self.vaporize * np.clip(1.3 - dist / R, 0.3, 1.0)
                if gone.any():                           # pulverised in the fireball: removed, no debris
                    g.occ[vi[gone], vj[gone], vk[gone]] = False
                    self.destroyed += int(gone.sum())
                    self.dirty += int(gone.sum())
                    keep = ~gone
                    vi, vj, vk, vel = vi[keep], vj[keep], vk[keep], vel[keep]
                self._destroy(vi, vj, vk, vel)
                self._ev('blast', n, p[:: max(1, n // 64)])
                core = int(g.bone[vi, vj, vk].sum())
                if core:
                    self._ev('powder', core)
        # --- 2. crater the ground -----------------------------------------------------------------
        low = C[:, 2] < self.crater + 0.3
        if low.any():
            changes = self.ground.blast(C[low], self.crater)
            if changes:
                self._ground_changed(changes, C[low])
        # --- 3. push nearby flying TNT (TNT-cannon physics) -------------------------------------
        if self.push > 0:
            fl = np.nonzero(self.tstate == FLY)[0]
            if len(fl):
                P = self.tp[fl]
                for c in C[:: max(1, E // 64)]:
                    dv = P - c
                    d = np.linalg.norm(dv, axis=1)
                    s = d < PUSH_R
                    if s.any():
                        k = fl[s]
                        imp = self.push * (1.0 - d[s] / PUSH_R) ** 2
                        self.tv[k] += dv[s] / np.maximum(d[s], 1e-6)[:, None] * imp[:, None]
                        self.tball[k] = True

    def _ground_changed(self, changes, centres):
        rng = self.rng
        ch = np.array(changes)                       # ix, iy, old, new
        n_blocks = int((ch[:, 2] - ch[:, 3]).sum())
        self._ev('ground', n_blocks, np.stack([ch[:, 0] - REG + 0.5, ch[:, 1] - REG + 0.5, ch[:, 3]], -1))
        # dirt clods
        pos, vel, cols = [], [], []
        for ix, iy, old, new in changes:
            for k in range(new, old):
                if self.ground_budget <= 0 or rng.random() > 0.55:
                    continue
                self.ground_budget -= 1
                p = np.array([ix - REG + rng.uniform(0.2, 0.8), iy - REG + rng.uniform(0.2, 0.8), k + 0.5])
                c = centres[np.argmin(np.linalg.norm(centres - p, axis=1))]
                d = p - c
                d[2] = abs(d[2]) + 0.6
                d /= max(np.linalg.norm(d), 1e-6)
                v = d * rng.uniform(7.0, 15.0) + rng.normal(0, 1.5, 3)
                pos.append(p)
                vel.append(v)
                side, top = block_colours(k)
                cols.append((side, top))
        if pos:
            n = len(pos)
            col = np.zeros(n, VOXEL_DTYPE)
            side = np.array([c[0] for c in cols], np.uint8)
            top = np.array([c[1] for c in cols], np.uint8)
            col['cx'][:, :3] = side
            col['cy'][:, :3] = side
            col['cz'][:, :3] = top
            col['inner'][:, :3] = side
            col['cx'][:, 3] = 0x3F
            col['cy'][:, 3] = 0x3F
            self._add_debris(np.array(pos), np.array(vel), col, rng.uniform(0.4, 0.55, n), spin=6.0)
        # lower the pile in the changed columns and wake anything resting there
        changed = np.zeros((2 * REG, 2 * REG), bool)
        for ix, iy, old, new in changes:
            x0 = ix - REG
            y0 = iy - REG
            px0, py0 = self._pile_idx(np.array([x0 + 0.01]), np.array([y0 + 0.01]))
            span = int(round(1.0 / PILE_RES))
            self.pile[px0[0]:px0[0] + span, py0[0]:py0[0] + span] = new
            changed[ix, iy] = True
        if len(self.dp):
            r = np.nonzero(self.drest)[0]
            if len(r):
                cx = np.floor(self.dp[r, 0]).astype(np.int64) + REG
                cy = np.floor(self.dp[r, 1]).astype(np.int64) + REG
                ok = (cx >= 0) & (cx < 2 * REG) & (cy >= 0) & (cy < 2 * REG)
                wake = np.zeros(len(r), bool)
                wake[ok] = changed[cx[ok], cy[ok]]
                if wake.any():
                    w = r[wake]
                    self.drest[w] = False
                    self.dv[w] = self.rng.normal(0, 0.5, (len(w), 3))

    # ------------------------------------------------------------------------------------------
    def _tnt_step(self):
        g = self.g
        t = self.t
        go = (self.tstate == WAIT) & (self.tlaunch <= t)
        self.tstate[go] = FLY
        fl = np.nonzero(self.tstate == FLY)[0]
        if len(fl) == 0:
            return
        v = self.tv[fl]
        p = self.tp[fl]
        fresh = ~self.tball[fl]
        if fresh.any():
            fi = fl[fresh]
            ramp = np.clip((t - self.tlaunch[fi]) / 0.2, 0.1, 1.0)
            v[fresh] = self.tdir[fi] * (self.tspeed[fi] * ramp)[:, None]
        passed = fresh & (p[:, 1] > BACK_Y)
        if passed.any():
            self.tball[fl[passed]] = True
        bal = self.tball[fl]
        v[bal, 2] -= G * DT
        v[bal] *= (1.0 - 0.3 * DT)
        newp = p + v * DT
        self.tv[fl] = v
        self.tp[fl] = newp
        # contact tests
        contact = np.zeros(len(fl), bool)
        for off in CONTACT_OFFS:
            ii, jj, kk = Giant.world_to_index(newp + off)
            contact |= g.occupied(ii, jj, kk)
        gh = self.ground_h(newp[:, 0], newp[:, 1])
        on_ground = newp[:, 2] - 0.5 < gh
        fuse = t >= self.tfuse[fl]
        boom = contact | on_ground | fuse
        if boom.any():
            b = fl[boom]
            C = newp[boom].copy()
            C[:, 2] = np.maximum(C[:, 2], gh[boom] + 0.5)
            self.tstate[b] = GONE
            n_c = int(contact[boom].sum())
            if n_c:
                self._ev('hit', n_c)
            self._explode(C)

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
        ii, jj, kk = Giant.world_to_index(p)
        inside = self.g.occupied(ii, jj, kk)
        if inside.any():
            ib = np.nonzero(inside)[0]
            p[ib] = self.dp[idx[ib]]
            v[ib] = -v[ib] * 0.25 + rng.normal(0, 0.5, (len(ib), 3))
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
            self.dq[si] = integrate_quat(self.dq[si], self.dw[si], DT)

    # ------------------------------------------------------------------------------------------
    def _structure(self):
        """Detach pieces that lost their connection to the ground; crumble loose voxels."""
        g = self.g
        o = g.occ
        nb = np.zeros(o.shape, np.int8)
        nb[1:] += o[:-1]
        nb[:-1] += o[1:]
        nb[:, 1:] += o[:, :-1]
        nb[:, :-1] += o[:, 1:]
        nb[:, :, 1:] += o[:, :, :-1]
        nb[:, :, :-1] += o[:, :, 1:]
        nb[:, :, 0] += 1
        loose = o & ((nb <= 1) | ((nb == 2) & (self.rng.random(o.shape) < 0.35)))
        if loose.any():
            i, j, k = np.nonzero(loose)
            n = len(i)
            vel = self.rng.normal(0, 1.2, (n, 3))
            vel[:, 2] = self.rng.uniform(-1.0, 1.0, n)
            self._destroy(i, j, k, vel)
            self._ev('crumble', n)
        if self.dirty < 40:
            return
        self.dirty = 0
        # 26-connectivity: the creeper's legs only meet the body along an edge (as in Minecraft's model)
        lab, n = ndimage.label(g.occ, structure=np.ones((3, 3, 3), bool))
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
                self._destroy(i, j, k, self.rng.normal(0, 1.0, (len(i), 3)))
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
        w = self.rng.normal(0, 0.45, 3)
        w[0] += self.rng.choice([-1, 1]) * 0.3
        self.chunks.append({'local': pos - com, 'col': col, 'pos': com.copy(),
                            'vel': np.array([0.0, 1.2, 0.8]), 'q': np.array([0.0, 0.0, 0.0, 1.0]), 'w': w})
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
        self.frame_explosions = []
        for s in range(SUB):
            self._tnt_step()
            self._debris_step()
            self._chunks_step()
            self.t += DT
        self._structure()
        C = np.concatenate(self.frame_explosions) if self.frame_explosions else np.zeros((0, 3))
        self.vfx.step(C)
        hover = int((self.tstate == WAIT).sum())
        flying = int((self.tstate == FLY).sum())
        if hover + flying:
            self._ev('primed', hover + flying)
        self.frame += 1
        return self.events

    # ------------------------------------------------------------------------------------------
    def instances(self):
        """Voxel instances (creeper + chunks + debris), TNT instances and effects for the renderer."""
        from creeper import pack_static
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
        live = self.tstate != GONE
        tnt = np.zeros((int(live.sum()), 5), np.float32)
        if len(tnt):
            tnt[:, 0:3] = self.tp[live]
            left = self.tfuse[live] - self.t
            swell = np.clip(1.0 - left / 0.5, 0, 1) ** 4
            tnt[:, 3] = 1.0 + 0.3 * swell
            ph = np.floor((self.t + self.tphase[live]) / FLASH_PERIOD).astype(np.int64)
            tnt[:, 4] = np.where(ph % 2 == 1, self.flash_amp, 0.0)
        fx = {'puffs': self.vfx.puffs(), 'flashes': self.vfx.flashes(), 'lights': self.vfx.lights()}
        return vox, tnt, fx


# ---------------------------------------------------------------------------------------------
# formations
# ---------------------------------------------------------------------------------------------
def block_formation(nx, nz, layers, x_center, z_bottom, y_front, spacing=1.0, launch=0.6, speed=24.0,
                    launch_jitter=0.02, jitter=0.0, seed=0, fuse_y=(14.0, 19.0)):
    """A solid nx x nz x layers block of TNT (layers stacked behind each other along -Y) flying along +Y."""
    rng = np.random.default_rng(seed)
    xs = x_center + (np.arange(nx) - (nx - 1) / 2.0) * spacing
    zs = z_bottom + np.arange(nz) * spacing
    ys = y_front - np.arange(layers) * spacing
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    pos = np.stack([X.ravel(), Y.ravel(), Z.ravel()], -1)
    n = len(pos)
    pos += rng.normal(0, jitter, (n, 3))
    lt = launch + rng.uniform(0, launch_jitter, n)
    spd = np.full(n, speed)
    # a TNT that hits nothing explodes after flying some way past the creeper (well inside the clearing)
    target_y = rng.uniform(fuse_y[0], fuse_y[1], n)
    fuse = lt + 0.1 + (target_y - pos[:, 1]) / spd
    return {'pos': pos, 'dir': np.tile([0.0, 1.0, 0.0], (n, 1)), 'launch': lt, 'speed': spd, 'fuse': fuse}
