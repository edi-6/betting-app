"""Physics for one round: anvils fall on the voxel Warden, crush him, stack up and bury him.

Anvils behave like the block game's falling blocks: they drop straight down without tumbling, locked to a grid
of CELL-sized columns, and land on whatever is highest in their column - the ground, other anvils, the Warden
or a piece that broke off him. Landing on flesh crushes a dent whose depth grows with the impact speed and
splashes flesh out; landing on anvils rings off them and, if the stack would become too steep, the anvil slides
off to a lower neighbour, so big volleys pour into a mound. Heavy stacks slowly sink into him. Loose flesh
crumbles, parts that lose their connection fall as rigid pieces and settle; anvils riding on them drop when they
go. A sonic boom (the Warden fighting back) blasts falling anvils out of the sky.

Vectorised numpy with a variable time step: every video frame advances the simulation by `scale / FPS`
seconds in ceil(SUB * scale) sub-steps, so slow motion is smooth and deterministic for a given schedule.
"""
import numpy as np
from scipy import ndimage

from warden import Giant, VOXEL_DTYPE, VS, NX, NY, NZ, GX0, GY0, GZ0
from mathutil import integrate_quat, quat_rotate, axis_angle_quat, quat_mul, slerp
import anvil as AN
from vfx import Dust

FPS = 30
SUB = 4
G = 26.0                          # anvil gravity (blocks / s^2)
VT = 46.0                         # terminal speed
DRAG = G / VT
CELL = AN.CELL
GH = 40                           # anvil grid covers columns -GH..GH on each axis
GN = 2 * GH + 1

# debris pile height field
PILE_HALF = 48.0
PILE_RES = 0.25
PN = int(2 * PILE_HALF / PILE_RES)

WAIT, FALL, REST, BLAST, FREE = 0, 1, 2, 3, 4
CHUNK_KEEP = 400                  # pieces with at least this many voxels tumble to rest instead of shattering


def yaw_quat(k):
    """Quarter-turn orientations about z."""
    k = np.asarray(k)
    ang = k * (np.pi / 2)
    return np.stack([np.zeros_like(ang, float), np.zeros_like(ang, float), np.sin(ang / 2), np.cos(ang / 2)], -1)


def cell_of(x, y):
    return (np.clip(np.round(np.asarray(x) / CELL).astype(np.int64), -GH, GH) + GH,
            np.clip(np.round(np.asarray(y) / CELL).astype(np.int64), -GH, GH) + GH)


def cell_center(ix, iy):
    return (np.asarray(ix) - GH) * CELL, (np.asarray(iy) - GH) * CELL


class RoundSim:
    def __init__(self, formation, seed=0, crush_k=0.00035, crush_max=1.2, splash=1.0, debris_frac=0.6,
                 weight_crush=0.0, booms=(), slide_steep=1.5):
        """formation: dict of per-anvil arrays: 'x', 'y' (snapped to the grid), 'z' (spawn height of the anvil's
        base), 't0' (when it starts falling), 'vz' (initial downward speed), 'yaw' (0-3 quarter turns), 'var'
        (wear 0-2), 'hover' (visible and hanging still before t0).
        booms: (time, origin, direction) sonic booms that blast falling anvils out of the way."""
        self.rng = np.random.default_rng(seed)
        self.g = Giant()
        self.dust = Dust(seed + 5)
        self.frame = 0
        self.t = 0.0
        self.dt_frame = 1.0 / FPS
        n = len(formation['x'])
        self.n = n
        self.ix, self.iy = cell_of(formation['x'], formation['y'])
        cx, cy = cell_center(self.ix, self.iy)
        self.xy = np.stack([cx, cy], -1).astype(np.float64)
        self.z = np.asarray(formation['z'], np.float64).copy()
        self.vz = np.asarray(formation['vz'], np.float64).copy()
        self.t0 = np.asarray(formation['t0'], np.float64)
        self.yaw = np.asarray(formation['yaw'], np.int64)
        self.var = np.asarray(formation['var'], np.float64).copy()
        self.hover = np.asarray(formation.get('hover', np.zeros(n, bool)), bool)
        self.state = np.full(n, WAIT, np.int8)
        self.q = yaw_quat(self.yaw)
        self.t_land = np.full(n, -10.0)
        self.slides = np.zeros(n, np.int64)
        self.slide_from = np.c_[self.xy, self.z]
        self.slide_t = np.full(n, -10.0)
        # free flight after a sonic boom
        self.p3 = np.zeros((n, 3))
        self.v3 = np.zeros((n, 3))
        self.w3 = np.zeros((n, 3))
        self.crush_k = crush_k
        self.crush_max = crush_max
        self.splash = splash
        self.debris_frac = debris_frac
        self.weight_crush = weight_crush
        self.slide_steep = slide_steep
        self.booms = [dict(t=float(b[0]), o=np.asarray(b[1], float), d=np.asarray(b[2], float) /
                           np.linalg.norm(b[2]), fired=False) for b in booms]
        # columns: resting stacks, tops of stacks, of the Warden and of fallen pieces
        self.stacks = {}
        self.H = np.zeros((GN, GN))
        self.CH = np.zeros((GN, GN))
        self.WT = np.zeros((GN, GN))
        vi, vj = np.meshgrid(np.arange(NX), np.arange(NY), indexing='ij')
        wx = GX0 + (vi + 0.5) * VS
        wy = GY0 + (vj + 0.5) * VS
        self.v_cell = cell_of(wx, wy)                      # anvil column of every voxel column
        self._update_wt()
        self.next_weight = 0.0
        # debris
        self.dp = np.zeros((0, 3))
        self.dv = np.zeros((0, 3))
        self.dq = np.zeros((0, 4))
        self.dw = np.zeros((0, 3))
        self.dcol = np.zeros(0, VOXEL_DTYPE)
        self.drest = np.zeros(0, bool)
        self.dscale = np.zeros(0)
        self.chunks = []
        self.chunk_counter = 0
        self.pile = np.zeros((PN, PN))
        self.destroyed = 0
        self.dirty = 0
        self.events = {}
        self.ground_hits = []
        self.sparks = []                                    # (pos, t) of anvil-on-anvil clangs
        self.n_static = 0

    # ------------------------------------------------------------------------------------------
    def _ev(self, name, n=1, pos=None):
        e = self.events.setdefault(name, [0, np.zeros(3), 0])
        e[0] += int(n)
        if pos is not None and n:
            p = np.asarray(pos, float).reshape(-1, 3)
            e[1] = e[1] + p.sum(0)
            e[2] += len(p)

    def _vmax(self, name, v):
        """Largest value of the frame under `name` (kept in the event count, e.g. impact speeds)."""
        e = self.events.setdefault(name, [0, np.zeros(3), 0])
        e[0] = max(e[0], int(round(v)))

    def _pile_idx(self, x, y):
        ix = np.clip(((x + PILE_HALF) / PILE_RES).astype(np.int64), 0, PN - 1)
        iy = np.clip(((y + PILE_HALF) / PILE_RES).astype(np.int64), 0, PN - 1)
        return ix, iy

    def ground_h(self, x, y):
        """Resting surface for debris and pieces: the debris pile, anvil stacks and settled pieces."""
        ix, iy = self._pile_idx(x, y)
        cx, cy = cell_of(x, y)
        return np.maximum(self.pile[ix, iy], np.maximum(self.H[cx, cy], self.CH[cx, cy]))

    def support(self, cx, cy):
        return np.maximum(np.maximum(self.H[cx, cy], self.WT[cx, cy]), self.CH[cx, cy])

    def _update_wt(self):
        """Top of the Warden in every anvil column."""
        occ = self.g.occ
        anyv = occ.any(2)
        top = NZ - 1 - np.argmax(occ[:, :, ::-1], axis=2)
        topz = np.where(anyv, GZ0 + (top + 1) * VS, 0.0)
        wt = np.zeros((GN, GN))
        np.maximum.at(wt, (self.v_cell[0].ravel(), self.v_cell[1].ravel()), topz.ravel())
        self.WT = wt
        self.v_top = top
        self.v_any = anyv

    def _update_wt_cell(self, fi, fj, cx, cy):
        """Refresh the Warden's top in one anvil column after a local change."""
        occ = self.g.occ[fi, fj]
        anyv = occ.any(1)
        top = NZ - 1 - np.argmax(occ[:, ::-1], axis=1)
        self.v_top[fi, fj] = top
        self.v_any[fi, fj] = anyv
        topz = np.where(anyv, GZ0 + (top + 1) * VS, 0.0)
        self.WT[cx, cy] = topz.max() if len(topz) else 0.0

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
        """Remove voxels from the Warden; a share `frac` of them flies off as debris with velocity vel."""
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

    def _footprint(self, cx, cy):
        """Voxel columns (i, j) whose centres lie in anvil column (cx, cy)."""
        sel = (self.v_cell[0] == cx) & (self.v_cell[1] == cy)
        return np.nonzero(sel)

    def _crush(self, a, speed):
        """Anvil a hits the Warden at `speed`: carve a dent from the top of its column and splash flesh."""
        cx, cy = self.ix[a], self.iy[a]
        fi, fj = self._footprint(cx, cy)
        if len(fi) == 0:
            return 0
        top = self.WT[cx, cy]
        depth = float(np.clip(self.crush_k * speed * speed, 0.25, self.crush_max))
        kz = np.arange(NZ)
        zc = GZ0 + (kz + 0.5) * VS
        band = zc > top - depth
        occ = self.g.occ[fi, fj][:, band]                      # (m, nb)
        m_i, m_k = np.nonzero(occ)
        if len(m_i) == 0:
            return 0
        i = fi[m_i]
        j = fj[m_i]
        k = np.nonzero(band)[0][m_k]
        n = len(i)
        pos = Giant.index_to_world(i, j, k)
        c = np.array([cell_center(cx, cy)[0], cell_center(cx, cy)[1]])
        out = pos[:, :2] - c[None, :]
        out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-3)
        vel = np.zeros((n, 3))
        vel[:, :2] = out * self.rng.uniform(2.0, 7.0, (n, 1)) * self.splash
        vel[:, 2] = self.rng.uniform(1.0, 6.5, n) * self.splash
        self._destroy(i, j, k, vel, self.debris_frac)
        self._ev('crush', n, pos.mean(0))
        self._update_wt_cell(fi, fj, cx, cy)
        return n

    # ------------------------------------------------------------------------------------------
    def _land(self, a):
        """Anvil a has reached the top of its column: crush, stack, or slide off."""
        cx, cy = int(self.ix[a]), int(self.iy[a])
        speed = float(self.vz[a])
        base = max(self.WT[cx, cy], self.CH[cx, cy], 0.0)
        on_anvil = self.H[cx, cy] > base + 1e-6
        pos = np.array([self.xy[a, 0], self.xy[a, 1], self.z[a]])
        if on_anvil:
            # too steep here? tumble down the pile to where it comes to rest (block-game anvils don't slide,
            # but real piles do; resolved at once so the pile keeps up, and animated as a quick tumble)
            start = np.array([self.xy[a, 0], self.xy[a, 1], self.H[cx, cy]])
            hops = 0
            while hops < 60:
                sup = self.H[cx, cy]
                best, bxy = None, None
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                    nx_, ny_ = cx + dx, cy + dy
                    if not (0 <= nx_ < GN and 0 <= ny_ < GN):
                        continue
                    s_n = float(self.support(nx_, ny_)) + (0.3 if dx and dy else 0.0) + self.rng.uniform(0, 0.2)
                    if s_n < sup - self.slide_steep * CELL and (best is None or s_n < best):
                        best, bxy = s_n, (nx_, ny_)
                if bxy is None:
                    break
                cx, cy = bxy
                hops += 1
                if self.H[cx, cy] <= max(self.WT[cx, cy], self.CH[cx, cy], 0.0) + 1e-6:
                    break                                    # reached flesh, a fallen piece or the ground
            if hops:
                self.slides[a] += hops
                self.ix[a], self.iy[a] = cx, cy
                ncx, ncy = cell_center(cx, cy)
                self.xy[a] = (ncx, ncy)
                self.slide_from[a] = start
                self.slide_t[a] = self.t
                self._ev('slide', 1, pos)
                base = max(self.WT[cx, cy], self.CH[cx, cy], 0.0)
                if self.H[cx, cy] <= base + 1e-6:
                    # tumbled off onto flesh / a piece / the ground
                    z = base
                    kind = 'land_ground' if base <= 0.0 else 'land_flesh'
                    self._ev(kind, 1, np.array([ncx, ncy, z]))
                    if base <= 0.0:
                        self.ground_hits.append(np.array([[ncx, ncy, 0.0]]))
                    self.z[a] = z
                    self.vz[a] = 0.0
                    self.state[a] = REST
                    self.t_land[a] = self.t
                    self.stacks.setdefault((cx, cy), []).append(a)
                    self.H[cx, cy] = z + AN.HEIGHT
                    return
            sup = self.H[cx, cy]
            self._ev('land_anvil', 1, pos)
            self._vmax('vmax_anvil', speed)
            self.sparks.append((pos + np.array([0.0, 0.0, 0.0]), self.t))
            z = sup
        elif self.WT[cx, cy] >= self.CH[cx, cy] and self.WT[cx, cy] > 0.0:
            self._ev('land_flesh', 1, pos)
            self._vmax('vmax_flesh', speed)
            self._crush(a, speed)
            z = max(self.WT[cx, cy], self.CH[cx, cy], self.H[cx, cy], 0.0)
        elif self.CH[cx, cy] > 0.0:
            self._ev('land_flesh', 1, pos)
            self._vmax('vmax_flesh', speed)
            z = self.CH[cx, cy]
        else:
            self._ev('land_ground', 1, pos)
            self._vmax('vmax_ground', speed)
            self.ground_hits.append(np.array([[pos[0], pos[1], 0.0]]))
            z = 0.0
        if speed > 22.0 and self.rng.random() < 0.35:
            self.var[a] = min(2.0, self.var[a] + 1.0)              # chipped / damaged by the fall
        self.z[a] = z
        self.vz[a] = 0.0
        self.state[a] = REST
        self.t_land[a] = self.t
        self.stacks.setdefault((cx, cy), []).append(a)
        self.H[cx, cy] = z + AN.HEIGHT

    def _fall(self, dt):
        go = (self.state == WAIT) & (self.t0 <= self.t)
        if go.any():
            self.state[go] = FALL
            self._ev('release', int(go.sum()), np.c_[self.xy[go], self.z[go]][:: max(1, int(go.sum()) // 32)])
        fl = np.nonzero(self.state == FALL)[0]
        if len(fl) == 0:
            return
        v = self.vz[fl]
        self.z[fl] -= v * dt + 0.5 * (G - DRAG * v) * dt * dt
        self.vz[fl] = v + (G - DRAG * v) * dt
        sup = self.support(self.ix[fl], self.iy[fl])
        hit = self.z[fl] <= sup
        if not hit.any():
            return
        land = fl[hit]
        land = land[np.argsort(self.z[land], kind='stable')]    # lowest first so stacks build in order
        for a in land:
            if self.state[a] != FALL:
                continue
            if self.z[a] > float(self.support(self.ix[a], self.iy[a])):
                continue                                         # its column dropped meanwhile
            self._land(a)

    def _check_stacks(self):
        """Stacks whose foundation got crushed or broke away sink or drop."""
        for key in list(self.stacks.keys()):
            ids = self.stacks[key]
            if not ids:
                continue
            cx, cy = key
            base = max(self.WT[cx, cy], self.CH[cx, cy], 0.0)
            zs = self.z[ids]
            bottom = zs.min()
            gap = bottom - base
            if gap <= 0.02:
                continue
            ids = np.asarray(ids)
            if gap < 0.6:
                self.z[ids] -= gap                                  # settle into the dent
                self.H[cx, cy] -= gap
                self._ev('settle', len(ids))
            else:
                self.state[ids] = FALL
                self.vz[ids] = 0.0
                self.stacks[key] = []
                self.H[cx, cy] = 0.0
                self._ev('stack_drop', len(ids))

    def _weight(self):
        """Tall stacks resting on him slowly crush their way down."""
        if self.weight_crush <= 0 or self.t < self.next_weight:
            return
        self.next_weight = self.t + 0.12
        for (cx, cy), ids in self.stacks.items():
            if len(ids) < 3:
                continue
            base = max(self.CH[cx, cy], 0.0)
            if self.WT[cx, cy] <= base or self.rng.random() > self.weight_crush * (len(ids) - 2) / 4.0:
                continue
            fi, fj = self._footprint(cx, cy)
            if len(fi) == 0:
                continue
            k = self.v_top[fi, fj]
            live = self.v_any[fi, fj] & (GZ0 + (k + 1) * VS >= self.WT[cx, cy] - 1e-6)
            if not live.any():
                continue
            i, j, kk = fi[live], fj[live], k[live]
            # squashed under the stack: nothing can fly out from under the anvils
            self._destroy(i, j, kk, np.zeros((len(i), 3)), 0.0)
            self._ev('sink', len(i), Giant.index_to_world(i, j, kk).mean(0))
        self._update_wt()

    # ------------------------------------------------------------------------------------------
    def _booms(self):
        for b in self.booms:
            if b['fired'] or self.t < b['t']:
                continue
            b['fired'] = True
            o, d = b['o'], b['d']
            cand = np.nonzero(self.state == FALL)[0]
            if len(cand):
                p = np.c_[self.xy[cand], self.z[cand] + AN.HEIGHT * 0.5]
                rel = p - o
                s = rel @ d
                perp = rel - s[:, None] * d
                dist = np.linalg.norm(perp, axis=1)
                hit = (s > 0) & (s < 40.0) & (dist < 2.0 + 0.02 * s)     # within the rings' reach
                blast = cand[hit]
                if len(blast):
                    m = len(blast)
                    out = perp[hit]
                    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-3)
                    self.state[blast] = BLAST
                    self.p3[blast] = np.c_[self.xy[blast], self.z[blast]]
                    self.v3[blast] = out * self.rng.uniform(12, 24, (m, 1)) + d * self.rng.uniform(6, 16, (m, 1))
                    self.w3[blast] = self.rng.normal(0, 6.0, (m, 3))
                    self._ev('blast', m, self.p3[blast].mean(0))
            self._ev('boom', 1, o)

    def _free_step(self, dt):
        bl = np.nonzero(self.state == BLAST)[0]
        if len(bl) == 0:
            return
        self.v3[bl, 2] -= G * dt
        self.p3[bl] += self.v3[bl] * dt
        self.q[bl] = integrate_quat(self.q[bl], self.w3[bl], dt)
        cx, cy = cell_of(self.p3[bl, 0], self.p3[bl, 1])
        sup = np.maximum(self.support(cx, cy), self.ground_h(self.p3[bl, 0], self.p3[bl, 1]))
        down = self.p3[bl, 2] <= sup
        if down.any():
            d = bl[down]
            self.p3[d, 2] = sup[down] - 0.15
            self.state[d] = FREE
            self._ev('land_ground', len(d), self.p3[d])
            self.ground_hits.append(self.p3[d] * np.array([1.0, 1.0, 0.0]))

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
                gi = idx[settle]
                self.drest[gi] = True
                v[settle] = 0.0
                yaw = rng.uniform(0, 2 * np.pi, len(settle))
                self.dq[gi] = axis_angle_quat(np.tile([0.0, 0, 1.0], (len(settle), 1)), yaw)
                ix, iy = self._pile_idx(p[settle, 0], p[settle, 1])
                p[settle, 2] = gh[settle] + self.dscale[gi] * 0.5
                np.maximum.at(self.pile, (ix, iy), gh[settle] + self.dscale[gi] * 0.85)
        self.dv[idx] = v
        self.dp[idx] = p
        spin = ~self.drest[idx]
        if spin.any():
            si = idx[spin]
            self.dq[si] = integrate_quat(self.dq[si], self.dw[si], dt)

    def _structure(self):
        """Loose voxels crumble; pieces that lost their connection to the ground break away."""
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
        loose = o & (nb <= 1)
        if loose.any():
            i, j, k = np.nonzero(loose)
            vel = self.rng.normal(0, 1.0, (len(i), 3))
            vel[:, 2] = self.rng.uniform(-1.0, 1.0, len(i))
            self._destroy(i, j, k, vel, self.debris_frac)
            self._ev('crumble', len(i))
        if self.dirty < 40:
            return
        self.dirty = 0
        lab, n = ndimage.label(g.occ)
        if n <= 1:
            self._update_wt()
            return
        grounded = np.unique(lab[:, :, 0])
        grounded = grounded[grounded > 0]
        sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        for comp in range(1, n + 1):
            if comp in grounded:
                continue
            i, j, k = np.nonzero(lab == comp)
            if sizes[comp - 1] < 24:
                self._destroy(i, j, k, self.rng.normal(0, 1.0, (len(i), 3)), self.debris_frac)
                continue
            self._make_chunk(i, j, k)
        self._update_wt()

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
        if w is None:
            w = self.rng.normal(0, 0.35, 3)
            w[0] += self.rng.choice([-1, 1]) * 0.25
        vel = np.array([0.0, -0.4, 0.0]) if vel is None else np.array(vel, float)
        self.chunks.append({'id': cid, 'local': pos - com, 'col': col, 'pos': com.copy(), 'vel': vel,
                            'q': np.array([0.0, 0.0, 0.0, 1.0]), 'w': np.array(w, float), 'landed': False,
                            'rest': False, 'q_rest': None})
        self._ev('detach', len(i), com)

    def _shatter(self, c, wp):
        n = len(wp)
        r = wp - c['pos']
        v = c['vel'][None, :] + np.cross(c['w'][None, :], r)
        v = v * 0.35 + self.rng.normal(0, 2.2, (n, 3))
        v[:, 2] = np.abs(v[:, 2]) * 0.6 + self.rng.uniform(0, 3.0, n) * (self.rng.random(n) < 0.5)
        col = c['col'].copy()
        col['cy'][:, 3] = 0x3F
        self._add_debris(wp, v, col, VS)
        self._ev('shatter', n, c['pos'])

    @staticmethod
    def _face_down(q, local):
        dirs = np.array([[1.0, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
        ext = np.ptp(local, axis=0)[[0, 0, 1, 1, 2, 2]]
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
                    self._shatter(c, wp)
                    continue
                c['pos'], c['q'] = pos0, q0
                out = c['pos'][:2] / max(np.linalg.norm(c['pos'][:2]), 1e-6)
                c['vel'] = np.array([out[0] * 3.0, out[1] * 3.0, min(c['vel'][2], 0.0)])
                c['w'] *= 0.8
                wp = quat_rotate(np.tile(c['q'], (len(c['local']), 1)), c['local']) + c['pos']
            gh = self.ground_h(wp[:, 0], wp[:, 1])
            pen = float((gh + VS * 0.5 - wp[:, 2]).max())
            if pen > 0:
                if not big:
                    self._shatter(c, wp)
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
                    ix, iy = self._pile_idx(wp[:, 0], wp[:, 1])
                    np.maximum.at(self.pile, (ix, iy), wp[:, 2] + VS * 0.5)
                    cx, cy = cell_of(wp[:, 0], wp[:, 1])
                    np.maximum.at(self.CH, (cx, cy), wp[:, 2] + VS * 0.5)
                    self._ev('chunk_rest', len(wp), c['pos'])
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
            self._booms()
            self._fall(dt)
            self._free_step(dt)
            self._debris_step(dt)
            self._chunks_step(dt)
            self.t += dt
        wt0 = self.WT.copy()
        self._structure()
        self._weight()
        if self.dirty or not np.array_equal(wt0, self.WT):
            self._check_stacks()
        self.dust.step(np.concatenate(self.ground_hits) if self.ground_hits else np.zeros((0, 3)), scale / FPS)
        self.dt_frame = scale / FPS
        nfl = int((self.state == FALL).sum())
        if nfl:
            f = self.state == FALL
            self._ev('falling', nfl, np.c_[self.xy[f], self.z[f]][:: max(1, nfl // 64)])
        self.sparks = [s for s in self.sparks if self.t - s[1] < 0.12]
        self.frame += 1
        return self.events

    # ------------------------------------------------------------------------------------------
    def anvil_instances(self):
        """(M, 10) float32 pos3 quat4 scale1 variant1 fade1 for every visible anvil."""
        vis = (self.state != WAIT) | self.hover
        idx = np.nonzero(vis)[0]
        out = np.zeros((len(idx), 10), np.float32)
        p = np.c_[self.xy[idx], self.z[idx]]
        # anvils tumbling down the pile: a quick hop from where they hit to where they came to rest
        dur = np.clip(0.08 + 0.04 * self.slides[idx], 0.1, 0.5)
        u = np.clip((self.t - self.slide_t[idx]) / dur, 0.0, 1.0)
        sl = u < 1.0
        if sl.any():
            e = u[sl]
            a0 = self.slide_from[idx[sl]]
            b0 = p[sl]
            lerp = a0 * (1 - e[:, None]) + b0 * e[:, None]
            lerp[:, 2] += np.sin(np.pi * e) * 0.6                   # a little hop
            p[sl] = lerp
        out[:, 0:3] = p
        free = (self.state[idx] == BLAST) | (self.state[idx] == FREE)
        if free.any():
            out[free, 0:3] = self.p3[idx[free]]
        out[:, 3:7] = self.q[idx]
        out[:, 7] = 1.0
        out[:, 8] = self.var[idx]
        out[:, 9] = 1.0
        return out

    def rings(self):
        """(R, 8) float32 pos3 axis3 radius alpha: the rings of sonic booms travelling out of his chest."""
        out = []
        for b in self.booms:
            if not b['fired']:
                continue
            age0 = self.t - b['t']
            for i in range(28):
                s = 1.5 + i * 2.2
                born = s / 110.0
                a = age0 - born
                if a < 0 or a > 0.45:
                    continue
                alpha = (1.0 - a / 0.45) ** 1.5
                out.append([*(b['o'] + b['d'] * s), *b['d'], 1.6 + 2.2 * (a / 0.45) + 0.02 * s, alpha])
        return np.array(out, np.float32).reshape(-1, 8)

    def flashes(self):
        """Tiny bright sparks where anvils clang on anvils: (K, 5) pos3 size alpha."""
        if not self.sparks:
            return np.zeros((0, 5), np.float32)
        out = []
        for p, t in self.sparks:
            a = 1.0 - (self.t - t) / 0.12
            out.append([p[0], p[1], p[2] + 0.05, 0.9, max(a, 0.0) * 0.6])
        return np.array(out, np.float32)

    def instances(self):
        from warden import pack_static
        parts = [pack_static(self.g)]
        self.n_static = len(parts[0])
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
        fx = {'puffs': self.dust.puffs(), 'rings': self.rings(), 'flashes': self.flashes()}
        return vox, self.anvil_instances(), fx


# ---------------------------------------------------------------------------------------------
# formations
# ---------------------------------------------------------------------------------------------
def formation(xy, z, t0, vz=0.0, rng=None, hover=False, var=None):
    """Anvils at grid positions xy (N, 2) with base heights z, release times t0."""
    n = len(xy)
    rng = rng or np.random.default_rng(0)
    return {'x': np.asarray(xy, float)[:, 0], 'y': np.asarray(xy, float)[:, 1],
            'z': np.broadcast_to(np.asarray(z, float), (n,)).copy(),
            't0': np.broadcast_to(np.asarray(t0, float), (n,)).copy(),
            'vz': np.broadcast_to(np.asarray(vz, float), (n,)).copy(),
            'yaw': rng.integers(0, 4, n), 'var': rng.integers(0, 2, n) if var is None else np.full(n, var),
            'hover': np.full(n, bool(hover))}


def merge(*forms):
    return {k: np.concatenate([f[k] for f in forms]) for k in forms[0]}
