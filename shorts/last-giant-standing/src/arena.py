"""One continuous battle between the four voxel giants: arrows, a creeper that blows itself up, anvils, a storm of
TNT and the Warden's sonic boom. Damage carries over from round to round.

Vectorised numpy with a variable time step: every video frame advances the simulation by `scale / FPS`
seconds in ceil(SUB * scale) sub-steps, so slow motion is smooth and a render is deterministic for a given
schedule. The pieces come from the single-giant videos, generalised to several giants at their own offsets:

* arrows fly exact ballistic arcs, carve a thin channel into whichever giant they hit, lose speed per voxel
  (bone costs more) and stick; they fall out when the flesh around their tip is destroyed, ride along on
  pieces that break off, and sink into the ground at their impact angle when they miss;
* anvils drop straight down on a grid like the block game's falling blocks, crush a dent into flesh by
  impact speed, stack, tumble down steep piles; explosions throw them around;
* primed TNT hangs, falls and explodes on contact (giants, ground, anvil stacks) or when its fuse runs out:
  a ragged sphere of voxels is blown out of any giant it reaches, the ground is cratered (dirt clods), nearby
  TNT, anvils and arrows are thrown about;
* the creeper primes (flashing white and swelling) and explodes, taking a bite out of its neighbours;
* scheduled sonic booms blast falling TNT out of the sky, and scheduled severs knock a giant over;
* loose flesh crumbles, disconnected parts fall as rigid pieces (big ones tumble to rest, small ones shatter),
  debris piles up.
"""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

import anvil as AN
import arrows as AR
from explosion_vfx import VFX
from ground import Ground, REG, block_colours
from mathutil import quat_from_basis, integrate_quat, quat_rotate, axis_angle_quat, quat_mul, slerp
from models import VOXEL_DTYPE, make_giants
from vfx import Dust

FPS = 30
SUB = 4
G = 20.0                          # debris and pieces
# arrows
GA = 20.0
GRAV_A = np.array([0.0, 0.0, -GA])
TIP = AR.TIP - 0.04
UP = np.array([0.0, 0.0, 1.0])
# anvils
GN = 26.0
VT = 46.0
DRAG = GN / VT
CELL = AN.CELL
GH = 42
GNC = 2 * GH + 1
# TNT
GT = 17.0
R_EXP = 1.7
R_CRATER = 1.6
FLASH_PERIOD = 0.25
PUSH_R = 3.5
TNT_HALF = 0.5
# debris pile height field (absolute heights; craters lower it)
PILE_HALF = 64.0
PILE_RES = 0.25
PN = int(2 * PILE_HALF / PILE_RES)
CHUNK_KEEP = 400

A_WAIT, A_FLY, A_STUCK, A_GROUND, A_FALL, A_REST, A_CARRIED = range(7)
N_WAIT, N_FALL, N_REST, N_BLAST, N_FREE = range(5)
T_WAIT, T_FALL, T_GONE, T_BLOWN = range(4)

_c = np.array([-0.42, 0.42])
CONTACT_OFFS = np.array([[x, y, z] for x in _c for y in _c for z in _c] +
                        [[0.5, 0, 0], [-0.5, 0, 0], [0, 0.5, 0], [0, -0.5, 0], [0, 0, 0.5], [0, 0, -0.5],
                         [0, 0, 0]])
_d = np.arange(-2, 3)
_A, _B, _C = np.meshgrid(_d, _d, _d, indexing='ij')
ANCHOR_OFFS = np.stack([_A.ravel(), _B.ravel(), _C.ravel()], -1)
ANCHOR_OFFS = ANCHOR_OFFS[np.argsort((ANCHOR_OFFS ** 2).sum(1), kind='stable')]


def _normalize(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _ball(radius, vs):
    """Voxel offsets within `radius` world units (voxel size vs) and their distances."""
    r = int(np.ceil(radius / vs))
    d = np.arange(-r, r + 1)
    a, b, c = np.meshgrid(d, d, d, indexing='ij')
    dist = np.sqrt(a * a + b * b + c * c) * vs
    sel = dist <= radius
    return np.stack([a[sel], b[sel], c[sel]], -1), dist[sel]


def arrow_quat(d, side, roll):
    y = side * np.cos(roll)[:, None] + np.cross(d, side) * np.sin(roll)[:, None]
    return quat_from_basis(d, np.cross(d, y))


def yaw_quat(k):
    ang = np.asarray(k) * (np.pi / 2)
    z = np.zeros_like(ang, float)
    return np.stack([z, z, np.sin(ang / 2), np.cos(ang / 2)], -1)


def cell_of(x, y):
    return (np.clip(np.round(np.asarray(x) / CELL).astype(np.int64), -GH, GH) + GH,
            np.clip(np.round(np.asarray(y) / CELL).astype(np.int64), -GH, GH) + GH)


def cell_center(ix, iy):
    return (np.asarray(ix) - GH) * CELL, (np.asarray(iy) - GH) * CELL


def _empty(n, *shape, dtype=np.float64):
    return np.zeros((n,) + shape, dtype)


class Arena:
    def __init__(self, plan, seed=0):
        """plan: dict with optional 'arrows', 'anvils', 'tnt' formations, 'creeper' (prime / explode times,
        blast radius), 'booms' [(t, origin, dir)], 'severs' [(t, giant, box, push, spin)]; see timeline.py."""
        self.rng = np.random.default_rng(seed)
        self.giants = make_giants()
        self.NG = len(self.giants)
        feet = [(g.lo[0] - 0.6, g.hi[0] + 0.6, g.lo[1] - 0.6, g.hi[1] + 0.6) for g in self.giants]
        self.ground = Ground(feet)
        self.dust = Dust(seed + 5)
        self.vfx = VFX(seed + 7)
        self.t = 0.0
        self.frame = 0
        self.dt_frame = 1.0 / FPS
        self.pile = np.zeros((PN, PN))
        self.destroyed = np.zeros(self.NG, np.int64)
        self.dirty = np.zeros(self.NG, np.int64)
        self.events = {}
        self.ground_hits = []
        self.frame_explosions = []
        self.big_blasts = []
        self.sparks = []
        self.ground_changed = False
        self._packs = {}
        self.carve_r = 0.26
        self.balls = {}
        # debris
        self.dp = _empty(0, 3)
        self.dv = _empty(0, 3)
        self.dq = _empty(0, 4)
        self.dw = _empty(0, 3)
        self.dcol = np.zeros(0, VOXEL_DTYPE)
        self.drest = np.zeros(0, bool)
        self.dscale = np.zeros(0)
        self.chunks = []
        self.chunk_counter = 0
        self._init_arrows(plan.get('arrows'))
        self._init_anvils(plan.get('anvils'))
        self._init_tnt(plan.get('tnt'))
        cr = plan.get('creeper')
        self.creeper = dict(cr) if cr else None
        if self.creeper:
            self.creeper['done'] = False
        self.booms = [dict(t=float(b[0]), o=np.asarray(b[1], float), d=_normalize(np.asarray(b[2], float)),
                           r0=float(b[3]) if len(b) > 3 else 5.0, fired=False) for b in plan.get('booms', ())]
        self.severs = sorted(plan.get('severs', ()), key=lambda x: x[0])
        # anvil columns: which grid cell every voxel column of every giant belongs to
        self.v_cell = []
        for g in self.giants:
            vi, vj = np.meshgrid(np.arange(g.NX), np.arange(g.NY), indexing='ij')
            p = g.index_to_world(vi, vj, np.zeros_like(vi))
            self.v_cell.append(cell_of(p[..., 0], p[..., 1]))
        self.H = np.zeros((GNC, GNC))          # top of resting anvil stacks
        self.CH = np.zeros((GNC, GNC))         # top of settled pieces (anvil grid)
        self.CP = np.zeros((PN, PN))           # top of settled pieces (fine grid)
        self.stacks = {}
        self.H_changed = False
        self._update_wt()

    # ==========================================================================================
    # helpers
    # ==========================================================================================
    def _ev(self, name, n=1, pos=None):
        e = self.events.setdefault(name, [0, np.zeros(3), 0])
        e[0] += int(n)
        if pos is not None and n:
            p = np.asarray(pos, float).reshape(-1, 3)
            e[1] = e[1] + p.sum(0)
            e[2] += len(p)

    def _vmax(self, name, v):
        e = self.events.setdefault(name, [0, np.zeros(3), 0])
        e[0] = max(e[0], int(round(v)))

    def _pile_idx(self, x, y):
        ix = np.clip(((np.asarray(x) + PILE_HALF) / PILE_RES).astype(np.int64), 0, PN - 1)
        iy = np.clip(((np.asarray(y) + PILE_HALF) / PILE_RES).astype(np.int64), 0, PN - 1)
        return ix, iy

    def ground_h(self, x, y):
        """Resting surface for debris, arrows and TNT: the ground (craters) and the debris piles, anvil stacks
        and settled pieces on it."""
        ix, iy = self._pile_idx(x, y)
        cx, cy = cell_of(x, y)
        return np.maximum(np.maximum(self.pile[ix, iy], self.CP[ix, iy]), self.H[cx, cy])

    def floor_h(self, x, y):
        ix, iy = self._pile_idx(x, y)
        return self.pile[ix, iy]

    def support(self, cx, cy):
        x, y = cell_center(cx, cy)
        return np.maximum(np.maximum(self.H[cx, cy], self.WT[cx, cy]), np.maximum(self.CH[cx, cy], self.floor_h(x, y)))

    def occupied_any(self, p):
        """(n,) giant index occupying each point, -1 where empty."""
        p = np.asarray(p, float).reshape(-1, 3)
        out = np.full(len(p), -1, np.int64)
        for gi, g in enumerate(self.giants):
            near = g.near(p, 0.01)
            if near.any():
                idx = np.nonzero(near)[0]
                occ = g.occupied_at(p[idx])
                out[idx[occ & (out[idx] < 0)]] = gi
        return out

    def ball(self, radius, vs):
        key = (round(radius, 4), round(vs, 5))
        if key not in self.balls:
            self.balls[key] = _ball(radius, vs)
        return self.balls[key]

    # ==========================================================================================
    # debris and damage
    # ==========================================================================================
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

    def _destroy(self, gi, i, j, k, vel, frac=1.0):
        """Remove voxels (unique indices) from giant gi; a share frac of them flies off as debris."""
        if len(i) == 0:
            return
        g = self.giants[gi]
        g.occ[i, j, k] = False
        self.destroyed[gi] += len(i)
        self.dirty[gi] += len(i)
        if frac < 1.0:
            keep = self.rng.random(len(i)) < frac
            i, j, k, vel = i[keep], j[keep], k[keep], vel[keep]
            if len(i) == 0:
                return
        col = g.colours(i, j, k)
        col['cy'][:, 3] = 0x3F
        self._add_debris(g.index_to_world(i, j, k), vel, col, g.VS)

    def _add_chunk_field(self, c, cp, ch):
        wp = c['wp']
        top = wp[:, 2] + c['vs'] * 0.5
        ix, iy = self._pile_idx(wp[:, 0], wp[:, 1])
        np.maximum.at(cp, (ix, iy), top)
        if ch is not None:
            cx, cy = cell_of(wp[:, 0], wp[:, 1])
            np.maximum.at(ch, (cx, cy), top)

    def _rebuild_chunk_fields(self):
        self.CP[:] = 0.0
        self.CH[:] = 0.0
        for c in self.chunks:
            if c['rest']:
                self._add_chunk_field(c, self.CP, self.CH)

    def _wake_unsupported(self):
        """Whatever rests on something that has gone (anvils thrown away, stacks sinking, craters, pieces that
        fell) starts falling again: resting pieces, then debris and arrows."""
        changed = True
        woke = False
        while changed:
            changed = False
            for c in self.chunks:
                if not c['rest']:
                    continue
                others = np.zeros((PN, PN))
                for o in self.chunks:
                    if o['rest'] and o is not c:
                        self._add_chunk_field(o, others, None)
                wp = c['wp']
                ix, iy = self._pile_idx(wp[:, 0], wp[:, 1])
                sup = np.maximum(self.pile[ix, iy], others[ix, iy])
                if float((wp[:, 2] - c['vs'] * 0.5 - sup).min()) > 0.3:
                    c['rest'] = False
                    c['landed'] = True
                    c['vel'][:] = 0.0
                    c['w'][:] = 0.0
                    changed = woke = True
                    self._ev('chunk_wake', len(wp), c['pos'])
            if changed:
                self._rebuild_chunk_fields()
        r = np.nonzero(self.drest)[0]
        if len(r):
            gh = self.ground_h(self.dp[r, 0], self.dp[r, 1])
            up = self.dp[r, 2] - self.dscale[r] * 0.5 > gh + 0.2
            if up.any():
                w = r[up]
                self.drest[w] = False
                self.dv[w] = self.rng.normal(0, 0.4, (len(w), 3))
        ar = np.nonzero(self.a_state == A_REST)[0]
        if len(ar):
            gh = self.ground_h(self.a_p[ar, 0], self.a_p[ar, 1])
            up = self.a_p[ar, 2] > gh + 0.4
            if up.any():
                w = ar[up]
                self.a_state[w] = A_FALL
                self.a_v[w] = 0.0
                self.a_w[w] = self.rng.normal(0, 2.0, (len(w), 3))
        return woke

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
        inside = self.occupied_any(p) >= 0
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
                gh2 = self.ground_h(p[settle, 0], p[settle, 1])
                p[settle, 2] = gh2 + self.dscale[gi] * 0.5
                # only debris lying on the floor builds up the pile; what rests on anvils or pieces must not
                # leave a floor in the air when those are blown away
                fl = gh2 <= self.floor_h(p[settle, 0], p[settle, 1]) + 1e-6
                if fl.any():
                    ix, iy = self._pile_idx(p[settle[fl], 0], p[settle[fl], 1])
                    top = gh2[fl] + self.dscale[gi[fl]] * 0.85
                    np.maximum.at(self.pile, (ix, iy), top)
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        np.maximum.at(self.pile, (np.clip(ix + dx, 0, PN - 1), np.clip(iy + dy, 0, PN - 1)),
                                      top - self.dscale[gi[fl]] * 0.6)
        self.dv[idx] = v
        self.dp[idx] = p
        spin = ~self.drest[idx]
        if spin.any():
            si = idx[spin]
            self.dq[si] = integrate_quat(self.dq[si], self.dw[si], dt)

    # ==========================================================================================
    # structure: crumbling, pieces breaking off
    # ==========================================================================================
    def _structure(self):
        for gi, g in enumerate(self.giants):
            if self.dirty[gi] == 0:
                continue
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
                self._destroy(gi, i, j, k, vel, 0.8)
                self._ev('crumble', len(i))
            if self.dirty[gi] < 40:
                continue
            self.dirty[gi] = 0
            lab, n = self._label(gi)
            if n <= 1:
                continue
            grounded = np.unique(lab[:, :, 0])
            grounded = grounded[grounded > 0]
            sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
            for comp in range(1, n + 1):
                if comp in grounded:
                    continue
                i, j, k = np.nonzero(lab == comp)
                if sizes[comp - 1] < 24:
                    self._destroy(gi, i, j, k, self.rng.normal(0, 1.0, (len(i), 3)), 0.8)
                    continue
                self._make_chunk(gi, i, j, k)
        self._update_wt()

    def _label(self, gi):
        """Connected parts of giant gi. The creeper's legs only meet its body along an edge (like the game's
        model), so for it edge neighbours count as connected too."""
        g = self.giants[gi]
        st = ndimage.generate_binary_structure(3, 2) if g.kind == 'creeper' else None
        return ndimage.label(g.occ, structure=st)

    def _make_chunk(self, gi, i, j, k, vel=None, w=None):
        g = self.giants[gi]
        pos = g.index_to_world(i, j, k)
        com = pos.mean(0)
        col = g.colours(i, j, k)
        occ_local = np.zeros(g.occ.shape, bool)
        occ_local[i, j, k] = True
        vis = np.zeros(len(i), np.uint8)
        for bit, (di, dj, dk) in enumerate(((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))):
            ni, nj, nk = i + di, j + dj, k + dk
            inb = g.inb(ni, nj, nk)
            full = np.zeros(len(i), bool)
            full[inb] = occ_local[ni[inb], nj[inb], nk[inb]]
            vis |= (~full).astype(np.uint8) << bit
        col['cy'][:, 3] = vis
        g.occ[i, j, k] = False
        self.destroyed[gi] += len(i)
        cid = self.chunk_counter
        self.chunk_counter += 1
        # arrows stuck in this piece ride along with it
        st = np.nonzero((self.a_state == A_STUCK) & (self.a_ag == gi))[0]
        if len(st):
            a = self.a_anchor[st]
            ride = st[occ_local[a[:, 0], a[:, 1], a[:, 2]]]
            if len(ride):
                self.a_state[ride] = A_CARRIED
                self.a_carrier[ride] = cid
                self.a_lp[ride] = self.a_p[ride] - com
                self.a_lq[ride] = self.a_q[ride]
        if w is None:
            w = self.rng.normal(0, 0.35, 3)
            w[0] += self.rng.choice([-1, 1]) * 0.25
        vel = np.array([0.0, -0.4, 0.0]) if vel is None else np.array(vel, float)
        self.chunks.append({'id': cid, 'g': gi, 'vs': g.VS, 'local': pos - com, 'col': col, 'pos': com.copy(),
                            'vel': vel, 'q': np.array([0.0, 0.0, 0.0, 1.0]), 'w': np.array(w, float),
                            'landed': False, 'rest': False, 'q_rest': None})
        self._ev('detach', len(i), com)

    def sever(self, gi, box, push, spin, hinge=None):
        """Blow out a joint (world box x0, x1, y0, y1, z0, z1) of giant gi; everything above it that loses its
        connection to the ground falls away as rigid pieces with the given push and spin. With `hinge`
        (dict pivot, axis, omega) the biggest piece topples over that pivot like a felled tree first."""
        g = self.giants[gi]
        x0, x1, y0, y1, z0, z1 = box
        a0 = np.array(g.world_to_index(np.array([x0, y0, z0])))
        a1 = np.array(g.world_to_index(np.array([x1, y1, z1])))
        a0 = np.maximum(a0, 0)
        a1 = np.minimum(a1, np.array([g.NX - 1, g.NY - 1, g.NZ - 1]))
        cut = np.zeros_like(g.occ)
        cut[a0[0]:a1[0] + 1, a0[1]:a1[1] + 1, a0[2]:a1[2] + 1] = True
        i, j, k = np.nonzero(cut & g.occ)
        push = np.asarray(push, float)
        if len(i):
            n = len(i)
            vel = self.rng.normal(0, 2.5, (n, 3)) + push * 0.5
            vel[:, 2] += self.rng.uniform(0.0, 2.0, n)
            self._destroy(gi, i, j, k, vel, 0.7)
            self._ev('sever', n, g.index_to_world(i, j, k).mean(0))
        lab, n = self._label(gi)
        grounded = np.unique(lab[:, :, 0])
        sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1)) if n else []
        free = [comp for comp in range(1, n + 1) if comp not in grounded]
        biggest = max(free, key=lambda cmp: sizes[cmp - 1]) if free else None
        for comp in free:
            ci, cj, ck = np.nonzero(lab == comp)
            if len(ci) < 24:
                self._destroy(gi, ci, cj, ck, self.rng.normal(0, 1.0, (len(ci), 3)), 0.8)
                continue
            self._make_chunk(gi, ci, cj, ck, vel=push, w=spin)
            if hinge is not None and comp == biggest:
                c = self.chunks[-1]
                piv = np.asarray(hinge['pivot'], float)
                r0 = c['pos'] - piv
                c['hinge'] = {'pivot': piv, 'axis': _normalize(np.asarray(hinge['axis'], float)),
                              'theta': 0.0, 'omega': float(hinge.get('omega', 0.4)), 'r0': r0,
                              'I': 4.0 / 3.0 * float(r0 @ r0), 'release': float(hinge.get('release', 1.45))}
                c['ignore_g'] = gi
        self.dirty[gi] = 0
        self._update_wt()

    def _shatter(self, c, wp, rid):
        n = len(wp)
        r = wp - c['pos']
        v = c['vel'][None, :] + np.cross(c['w'][None, :], r)
        v = v * 0.35 + self.rng.normal(0, 2.2, (n, 3))
        v[:, 2] = np.abs(v[:, 2]) * 0.6 + self.rng.uniform(0, 3.0, n) * (self.rng.random(n) < 0.5)
        col = c['col'].copy()
        col['cy'][:, 3] = 0x3F
        self._add_debris(wp, v, col, c['vs'])
        if len(rid):
            self._a_drop(rid)
            self.a_v[rid] += c['vel'] * 0.35
        self._ev('shatter', n, c['pos'])

    def _carry_arrows(self, c, rid):
        if len(rid):
            qq = np.tile(c['q'], (len(rid), 1))
            self.a_p[rid] = quat_rotate(qq, self.a_lp[rid]) + c['pos']
            self.a_q[rid] = quat_mul(qq, self.a_lq[rid])
            self.a_d[rid] = quat_rotate(self.a_q[rid], np.tile([1.0, 0.0, 0.0], (len(rid), 1)))

    def _hinge_step(self, c, dt, rid):
        """A piece toppling over a pivot on its stump (gravity torque about the hinge axis); once it has tipped
        far enough or touches the floor it flies on as a free rigid body with the velocity it has."""
        h = c['hinge']
        R = axis_angle_quat(h['axis'], h['theta'])
        r = quat_rotate(R, h['r0'])
        alpha = float(np.cross(r, np.array([0.0, 0.0, -G])) @ h['axis']) / h['I']
        h['omega'] += alpha * dt
        h['theta'] += h['omega'] * dt
        R = axis_angle_quat(h['axis'], h['theta'])
        c['q'] = R / np.linalg.norm(R)
        c['pos'] = h['pivot'] + quat_rotate(R, h['r0'])
        c['w'] = h['axis'] * h['omega']
        c['vel'] = np.cross(c['w'], c['pos'] - h['pivot'])
        wp = quat_rotate(np.tile(c['q'], (len(c['local']), 1)), c['local']) + c['pos']
        self._chunk_knock(c, wp)
        floor = self._piece_floor(wp[:, 0], wp[:, 1])
        if h['theta'] >= h['release'] or bool((wp[:, 2] - c['vs'] * 0.5 < floor).any()):
            del c['hinge']
            self._ev('topple', len(wp), c['pos'])
        self._carry_arrows(c, rid)

    def _piece_floor(self, x, y):
        """What a falling piece lands on: the ground, debris and resting pieces (anvils it knocks away)."""
        ix, iy = self._pile_idx(x, y)
        return np.maximum(self.pile[ix, iy], self.CP[ix, iy])

    def _chunk_knock(self, c, wp):
        """Resting anvils in the way of a moving piece are knocked away with it."""
        rs = np.nonzero(self.n_state == N_REST)[0]
        if len(rs) == 0:
            return
        pa = np.c_[self.n_xy[rs], self.n_z[rs] + AN.HEIGHT * 0.5]
        lo = wp.min(0) - 1.0
        hi = wp.max(0) + 1.0
        inb = np.all((pa >= lo) & (pa <= hi), axis=1)
        if not inb.any():
            return
        cand = rs[inb]
        d, _ = cKDTree(wp[::2]).query(pa[inb], distance_upper_bound=0.85)
        hit = np.isfinite(d)
        if not hit.any():
            return
        k = cand[hit]
        p = pa[inb][hit]
        self._unstack(k)
        self.n_p3[k] = p - np.array([0.0, 0.0, AN.HEIGHT * 0.5])
        v = c['vel'][None, :] + np.cross(c['w'][None, :], p - c['pos'])
        out = _normalize(np.c_[p[:, :2] - c['pos'][:2], np.zeros(len(k))])
        self.n_v3[k] = v * 0.8 + out * self.rng.uniform(1.0, 3.0, (len(k), 1)) + np.array([0.0, 0.0, 1.5])
        self.n_w3[k] = self.rng.normal(0, 3.0, (len(k), 3))
        self.n_state[k] = N_BLAST
        self._ev('anvil_knock', len(k), p)

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
        keep = []
        for c in self.chunks:
            if c['rest']:
                keep.append(c)
                continue
            rid = np.nonzero((self.a_state == A_CARRIED) & (self.a_carrier == c['id']))[0]
            big = len(c['local']) >= CHUNK_KEEP
            if 'hinge' in c:
                self._hinge_step(c, dt, rid)
                keep.append(c)
                continue
            pos0, q0 = c['pos'].copy(), c['q'].copy()
            c['vel'][2] -= G * dt
            c['pos'] = c['pos'] + c['vel'] * dt
            c['q'] = integrate_quat(c['q'][None], c['w'][None], dt)[0]
            if c['landed']:
                c['q'] = slerp(c['q'][None], c['q_rest'][None], min(1.0, 5.0 * dt))[0]
            wp = quat_rotate(np.tile(c['q'], (len(c['local']), 1)), c['local']) + c['pos']
            occ = self.occupied_any(wp[::3])
            if 'ignore_g' in c:
                occ[occ == c['ignore_g']] = -1
            if (occ >= 0).sum() > 3:
                if not big:
                    self._shatter(c, wp, rid)
                    continue
                c['pos'], c['q'] = pos0, q0
                out = c['pos'][:2] - self.giants[c['g']].center[:2]
                out /= max(np.linalg.norm(out), 1e-6)
                c['vel'] = np.array([out[0] * 3.0, out[1] * 3.0, min(c['vel'][2], 0.0)])
                c['w'] *= 0.8
                wp = quat_rotate(np.tile(c['q'], (len(c['local']), 1)), c['local']) + c['pos']
            if len(c['local']) >= 60:
                self._chunk_knock(c, wp)
            gh = self._piece_floor(wp[:, 0], wp[:, 1])
            pen = float((gh + c['vs'] * 0.5 - wp[:, 2]).max())
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
                    gh = self._piece_floor(wp[:, 0], wp[:, 1])
                    c['pos'][2] += float((gh + c['vs'] * 0.5 - wp[:, 2]).max())
                    wp = quat_rotate(np.tile(c['q'], (len(c['local']), 1)), c['local']) + c['pos']
                    c['rest'] = True
                    c['wp'] = wp
                    self._add_chunk_field(c, self.CP, self.CH)
                    self._ev('chunk_rest', len(wp), c['pos'])
            self._carry_arrows(c, rid)
            keep.append(c)
        self.chunks = keep

    # ==========================================================================================
    # arrows
    # ==========================================================================================
    def _init_arrows(self, f):
        n = 0 if f is None else len(f['pos'])
        self.a_n = n
        self.a_p = _empty(n, 3) if n == 0 else f['pos'].astype(np.float64).copy()
        self.a_v0 = _empty(n, 3) if n == 0 else f['vel'].astype(np.float64).copy()
        self.a_v = self.a_v0.copy()
        self.a_launch = np.zeros(n) if n == 0 else f['launch'].astype(np.float64)
        self.a_var = np.zeros(n) if n == 0 else f['var'].astype(np.float32)
        self.a_state = np.full(n, A_WAIT, np.int8)
        side = np.cross(self.a_v0, UP) if n else _empty(0, 3)
        if n:
            side[np.linalg.norm(side, axis=1) < 1e-6] = (1.0, 0.0, 0.0)
        self.a_side = _normalize(side) if n else side
        self.a_roll = self.rng.uniform(-0.5, 0.5, n) + np.where(self.rng.random(n) < 0.5, 0.0, np.pi / 4)
        self.a_d = _normalize(self.a_v0) if n else _empty(0, 3)
        self.a_q = arrow_quat(self.a_d, self.a_side, self.a_roll) if n else _empty(0, 4)
        self.a_w = _empty(n, 3)
        self.a_ag = np.full(n, -1, np.int64)
        self.a_anchor = np.full((n, 3), -1, np.int64)
        self.a_thit = np.full(n, -10.0)
        self.a_hit = np.zeros(n, bool)
        self.a_carrier = np.full(n, -1, np.int64)
        self.a_lp = _empty(n, 3)
        self.a_lq = _empty(n, 4)
        self.a_cost = 2.2
        self.a_bone = 6.0
        self.a_embed = 3.0

    def _find_anchor(self, tip):
        """Nearest occupied voxel of any giant within 2 voxels of each tip."""
        n = len(tip)
        gid = np.full(n, -1, np.int64)
        out = np.full((n, 3), -1, np.int64)
        for gi, g in enumerate(self.giants):
            todo = np.nonzero((gid < 0) & g.near(tip, 1.0))[0]
            if len(todo) == 0:
                continue
            ti, tj, tk = g.world_to_index(tip[todo])
            found = np.zeros(len(todo), bool)
            for off in ANCHOR_OFFS:
                rem = ~found
                if not rem.any():
                    break
                ci, cj, ck = ti[rem] + off[0], tj[rem] + off[1], tk[rem] + off[2]
                occ = g.occupied(ci, cj, ck)
                if occ.any():
                    sub = np.nonzero(rem)[0][occ]
                    out[todo[sub]] = np.stack([ci[occ], cj[occ], ck[occ]], -1)
                    found[sub] = True
            gid[todo[found]] = gi
        return gid, out

    def _a_stick(self, idx, pos, d):
        self.a_p[idx] = pos
        self.a_d[idx] = d
        self.a_v[idx] = 0.0
        self.a_q[idx] = arrow_quat(d, self.a_side[idx], self.a_roll[idx])
        gid, anc = self._find_anchor(pos + d * TIP)
        self.a_ag[idx] = gid
        self.a_anchor[idx] = anc
        self.a_state[idx] = A_STUCK
        self.a_thit[idx] = self.t
        lost = idx[gid < 0]
        if len(lost):
            self._a_drop(lost)
        self._ev('stick', len(idx) - len(lost), pos)

    def _a_drop(self, idx):
        n = len(idx)
        if n == 0:
            return
        self.a_state[idx] = A_FALL
        self.a_carrier[idx] = -1
        self.a_v[idx] = -self.a_d[idx] * self.rng.uniform(0.3, 1.5, (n, 1)) + self.rng.normal(0, 0.6, (n, 3))
        self.a_w[idx] = self.rng.normal(0, 3.0, (n, 3))
        self._ev('fall', n)

    def _a_fly(self, dt):
        rng = self.rng
        go = (self.a_state == A_WAIT) & (self.a_launch <= self.t)
        if go.any():
            self.a_state[go] = A_FLY
            self._ev('launch', int(go.sum()), self.a_p[go][:: max(1, int(go.sum()) // 64)])
        fl = np.nonzero(self.a_state == A_FLY)[0]
        if len(fl) == 0:
            return
        p0 = self.a_p[fl]
        v = self.a_v[fl]
        step = v * dt + 0.5 * GRAV_A * dt * dt
        v_new = v + GRAV_A * dt
        d = _normalize(v_new)
        speed = np.linalg.norm(v_new, axis=1)
        frac = np.ones(len(fl))
        stopped = np.zeros(len(fl), bool)
        first = np.zeros(len(fl), bool)
        spd = speed.copy()
        alive = np.ones(len(fl), bool)
        for gi, g in enumerate(self.giants):
            pad = TIP + 1.0 + np.abs(step)
            near = np.nonzero(np.all((p0 > g.lo - pad) & (p0 < g.hi + pad), axis=1) & alive)[0]
            if len(near) == 0:
                continue
            carve = self.ball(self.carve_r, g.VS)[0]
            L = np.linalg.norm(step[near], axis=1)
            nsub = max(1, int(np.ceil(L.max() / (g.VS * 0.5))))
            all_i, all_j, all_k, all_src = [], [], [], []
            for s in range(1, nsub + 1):
                f = s / nsub
                tip = p0[near] + step[near] * f + d[near] * TIP
                ii, jj, kk = g.world_to_index(tip)
                occ = g.occupied(ii, jj, kk) & alive[near]
                if not occ.any():
                    continue
                hs = np.nonzero(occ)[0]
                ci = ii[hs, None] + carve[None, :, 0]
                cj = jj[hs, None] + carve[None, :, 1]
                ck = kk[hs, None] + carve[None, :, 2]
                inb = g.inb(ci, cj, ck)
                ci, cj, ck = np.where(inb, ci, 0), np.where(inb, cj, 0), np.where(inb, ck, 0)
                o = inb & g.occ[ci, cj, ck]
                nb = o & g.bone[ci, cj, ck]
                cost = o.sum(1) * self.a_cost * (g.VS / 0.25) ** 3 + nb.sum(1) * (self.a_bone - self.a_cost)
                all_i.append(ci[o])
                all_j.append(cj[o])
                all_k.append(ck[o])
                all_src.append(np.repeat(near[hs], o.sum(1)))
                g.occ[ci[o], cj[o], ck[o]] = False
                newly = near[hs][~first[near[hs]] & ~self.a_hit[fl[near[hs]]]]
                first[newly] = True
                spd[near[hs]] -= cost
                stop = near[hs][spd[near[hs]] < self.a_embed]
                if len(stop):
                    alive[stop] = False
                    frac[stop] = f
                    stopped[stop] = True
            if all_i:
                ci = np.concatenate(all_i)
                cj = np.concatenate(all_j)
                ck = np.concatenate(all_k)
                src = np.concatenate(all_src)
                flat = np.ravel_multi_index((ci, cj, ck), g.occ.shape)
                flat, fi = np.unique(flat, return_index=True)
                ci, cj, ck = np.unravel_index(flat, g.occ.shape)
                sd = d[src[fi]]
                m = len(ci)
                back = -sd * rng.uniform(1.0, 6.0, (m, 1))
                perp = rng.normal(0, 1.8, (m, 3))
                perp -= sd * np.sum(perp * sd, 1, keepdims=True)
                vel = back + perp
                vel[:, 2] += rng.uniform(-0.5, 2.5, m)
                g.occ[ci, cj, ck] = True
                self._destroy(gi, ci, cj, ck, vel, 0.7)
                self._ev('carve', m, g.index_to_world(ci, cj, ck)[:: max(1, m // 48)])
                nbone = int(g.bone[ci, cj, ck].sum())
                if nbone:
                    self._ev('bone', nbone)
        if first.any():
            hi_ = fl[first]
            self.a_hit[hi_] = True
            self._ev('impact', len(hi_), p0[first] + d[first] * TIP)
        keep = ~stopped
        v_new[keep] = d[keep] * np.maximum(spd[keep], 0.0)[:, None]
        newp = p0 + step * frac[:, None]
        self.a_p[fl] = newp
        self.a_v[fl] = v_new
        self.a_d[fl] = d
        if stopped.any():
            self._a_stick(fl[stopped], newp[stopped], d[stopped])
        fly = fl[~stopped]
        if len(fly):
            tip = self.a_p[fly] + self.a_d[fly] * TIP
            gh = self.ground_h(tip[:, 0], tip[:, 1])
            hit = tip[:, 2] < gh
            if hit.any():
                self._a_ground(fly[hit], tip[hit], gh[hit])

    def _a_ground(self, idx, tip, gh):
        d = self.a_d[idx]
        dz = d[:, 2]
        steep = dz < -0.12
        s = idx[steep]
        if len(s):
            back = (gh[steep] - tip[steep, 2]) / np.maximum(-dz[steep], 1e-3)
            back = np.where(back > 0.8, 0.3, back)
            cross = tip[steep] - d[steep] * back[:, None]
            tip_final = cross + d[steep] * 0.5
            self.a_p[s] = tip_final - d[steep] * TIP
            self.a_v[s] = 0.0
            self.a_q[s] = arrow_quat(d[steep], self.a_side[s], self.a_roll[s])
            self.a_state[s] = A_GROUND
            self.a_thit[s] = self.t
            self.ground_hits.append(cross)
            self._ev('ground_hit', len(s), cross[:: max(1, len(s) // 48)])
        k = idx[~steep]
        if len(k):
            self.a_q[k] = arrow_quat(d[~steep], self.a_side[k], self.a_roll[k])
            self.a_v[k] *= 0.3
            self.a_v[k, 2] = np.abs(self.a_v[k, 2]) * 0.2
            self.a_p[k, 2] += gh[~steep] - tip[~steep, 2] + 0.05
            self.a_state[k] = A_FALL
            self.a_w[k] = self.rng.normal(0, 4.0, (len(k), 3))
            self._ev('ground_hit', len(k), tip[~steep])

    def _a_check_anchors(self):
        st = np.nonzero(self.a_state == A_STUCK)[0]
        if len(st) == 0:
            return
        ok = np.zeros(len(st), bool)
        for gi, g in enumerate(self.giants):
            m = self.a_ag[st] == gi
            if m.any():
                a = self.a_anchor[st[m]]
                ok[m] = g.occupied(a[:, 0], a[:, 1], a[:, 2])
        lost = st[~ok]
        if len(lost) == 0:
            return
        gid, anc = self._find_anchor(self.a_p[lost] + self.a_d[lost] * TIP)
        f = gid >= 0
        self.a_ag[lost[f]] = gid[f]
        self.a_anchor[lost[f]] = anc[f]
        if (~f).any():
            self._a_drop(lost[~f])

    def _a_fall(self, dt):
        fa = np.nonzero(self.a_state == A_FALL)[0]
        if len(fa) == 0:
            return
        self.a_v[fa] += GRAV_A * dt
        self.a_v[fa] *= (1.0 - 0.2 * dt)
        newp = self.a_p[fa] + self.a_v[fa] * dt
        inside = self.occupied_any(newp) >= 0
        if inside.any():
            ib = fa[inside]
            self.a_v[ib, :2] = self.a_v[ib, :2] * -0.3 + self.rng.normal(0, 0.5, (len(ib), 2))
            self.a_v[ib, 1] -= 1.0
            self.a_v[ib, 2] *= 0.2
            newp[inside] = self.a_p[ib] + self.a_v[ib] * dt
        self.a_p[fa] = newp
        self.a_q[fa] = integrate_quat(self.a_q[fa], self.a_w[fa], dt)
        dirs = quat_rotate(self.a_q[fa], np.tile([1.0, 0.0, 0.0], (len(fa), 1)))
        low = np.minimum(self.a_p[fa, 2] + dirs[:, 2] * TIP, self.a_p[fa, 2] - dirs[:, 2] * TIP)
        gh = self.ground_h(self.a_p[fa, 0], self.a_p[fa, 1])
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
            ang = np.pi / 4
            zax = UP * np.cos(ang) + np.cross(dflat, UP) * np.sin(ang)
            self.a_q[li] = quat_from_basis(dflat, zax)
            self.a_p[li, 2] = gh[land] + 0.06
            self.a_v[li] = 0.0
            self.a_state[li] = A_REST
            self._ev('arrow_land', n, self.a_p[li])

    def arrow_instances(self):
        vis = self.a_state != A_WAIT
        idx = np.nonzero(vis)[0]
        q = self.a_q[idx].copy()
        p = self.a_p[idx].copy()
        fl = self.a_state[idx] == A_FLY
        if fl.any():
            f = idx[fl]
            q[fl] = arrow_quat(self.a_d[f], self.a_side[f], self.a_roll[f])
        age = self.t - self.a_thit[idx]
        quiv = ((self.a_state[idx] == A_STUCK) | (self.a_state[idx] == A_GROUND)) & (age < 0.7)
        if quiv.any():
            k = idx[quiv]
            a = age[quiv]
            ang = 0.1 * np.exp(-a / 0.13) * np.sin(2 * np.pi * 13.0 * a)
            dq = axis_angle_quat(self.a_side[k], ang)
            tip = self.a_p[k] + self.a_d[k] * TIP
            p[quiv] = tip + quat_rotate(dq, self.a_p[k] - tip)
            q[quiv] = quat_mul(dq, q[quiv])
        out = np.zeros((len(idx), 10), np.float32)
        out[:, 0:3] = p
        out[:, 3:7] = q
        out[:, 7] = 1.0
        out[:, 8] = self.a_var[idx]
        out[:, 9] = 1.0
        return out

    def arrow_streaks(self):
        f = np.nonzero(self.a_state == A_FLY)[0]
        if len(f) == 0:
            return np.zeros((0, 8), np.float32)
        spd = np.linalg.norm(self.a_v[f], axis=1)
        f = f[spd > 6.0]
        if len(f) == 0:
            return np.zeros((0, 8), np.float32)
        head = self.a_p[f] - self.a_d[f] * TIP * 0.8
        tail = head - self.a_v[f] * (0.5 * self.dt_frame)
        out = np.zeros((len(f), 8), np.float32)
        out[:, 0:3] = head
        out[:, 3:6] = tail
        out[:, 6] = 0.12
        out[:, 7] = 0.35
        return out

    # ==========================================================================================
    # anvils
    # ==========================================================================================
    def _init_anvils(self, f):
        n = 0 if f is None else len(f['x'])
        self.n_n = n
        if n == 0:
            f = {k: np.zeros(0) for k in ('x', 'y', 'z', 't0', 'vz', 'yaw', 'var', 'hover')}
        self.n_ix, self.n_iy = cell_of(f['x'], f['y'])
        cx, cy = cell_center(self.n_ix, self.n_iy)
        self.n_xy = np.stack([cx, cy], -1).astype(np.float64).reshape(-1, 2)
        self.n_z = np.asarray(f['z'], np.float64).copy()
        self.n_vz = np.asarray(f['vz'], np.float64).copy()
        self.n_t0 = np.asarray(f['t0'], np.float64)
        self.n_var = np.asarray(f['var'], np.float64).copy()
        self.n_hover = np.asarray(f['hover'], bool)
        self.n_appear = np.asarray(f.get('appear', np.zeros(n)), np.float64)
        self.n_state = np.full(n, N_WAIT, np.int8)
        self.n_q = yaw_quat(np.asarray(f['yaw'], np.int64)).reshape(-1, 4)
        self.n_tland = np.full(n, -10.0)
        self.n_slides = np.zeros(n, np.int64)
        self.n_from = np.c_[self.n_xy, self.n_z] if n else _empty(0, 3)
        self.n_tslide = np.full(n, -10.0)
        self.n_p3 = _empty(n, 3)
        self.n_v3 = _empty(n, 3)
        self.n_w3 = _empty(n, 3)
        self.n_crush_k = 0.0010
        self.n_crush_max = 2.2
        self.n_steep = 1.5

    def _update_wt(self):
        """Top of the giants in every anvil column, and which giant it belongs to."""
        wt = np.zeros((GNC, GNC))
        wg = np.full((GNC, GNC), -1, np.int64)
        self.v_top = []
        self.v_any = []
        for gi, g in enumerate(self.giants):
            occ = g.occ
            anyv = occ.any(2)
            top = g.NZ - 1 - np.argmax(occ[:, :, ::-1], axis=2)
            topz = np.where(anyv, g.origin[2] + (top + 1) * g.VS, 0.0)
            self.v_top.append(top)
            self.v_any.append(anyv)
            cx, cy = self.v_cell[gi]
            cur = np.zeros((GNC, GNC))
            np.maximum.at(cur, (cx.ravel(), cy.ravel()), topz.ravel())
            higher = cur > wt
            wg[higher] = gi
            wt = np.maximum(wt, cur)
        self.WT = wt
        self.WTg = wg

    def _footprint(self, gi, cx, cy):
        sel = (self.v_cell[gi][0] == cx) & (self.v_cell[gi][1] == cy)
        return np.nonzero(sel)

    def _crush(self, a, speed):
        cx, cy = int(self.n_ix[a]), int(self.n_iy[a])
        gi = int(self.WTg[cx, cy])
        if gi < 0:
            return 0
        g = self.giants[gi]
        fi, fj = self._footprint(gi, cx, cy)
        if len(fi) == 0:
            return 0
        top = self.WT[cx, cy]
        depth = float(np.clip(self.n_crush_k * speed * speed, 0.25, self.n_crush_max))
        zc = g.origin[2] + (np.arange(g.NZ) + 0.5) * g.VS
        band = zc > top - depth
        occ = g.occ[fi, fj][:, band]
        m_i, m_k = np.nonzero(occ)
        if len(m_i) == 0:
            return 0
        i, j = fi[m_i], fj[m_i]
        k = np.nonzero(band)[0][m_k]
        n = len(i)
        pos = g.index_to_world(i, j, k)
        c = np.array(cell_center(cx, cy))
        out = _normalize(pos[:, :2] - c[None, :])
        vel = np.zeros((n, 3))
        vel[:, :2] = out * self.rng.uniform(2.0, 7.0, (n, 1))
        vel[:, 2] = self.rng.uniform(1.0, 6.5, n)
        self._destroy(gi, i, j, k, vel, 0.6)
        self._ev('crush', n, pos.mean(0))
        self._update_wt()
        return n

    def _n_land(self, a):
        cx, cy = int(self.n_ix[a]), int(self.n_iy[a])
        speed = float(self.n_vz[a])
        pos = np.array([self.n_xy[a, 0], self.n_xy[a, 1], self.n_z[a]])
        fx_, fy_ = cell_center(cx, cy)
        base = max(self.WT[cx, cy], self.CH[cx, cy], float(self.floor_h(fx_, fy_)))
        on_anvil = self.H[cx, cy] > base + 1e-6
        if on_anvil:
            start = np.array([self.n_xy[a, 0], self.n_xy[a, 1], self.H[cx, cy]])
            hops = 0
            while hops < 60:
                sup = self.H[cx, cy]
                best, bxy = None, None
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                    nx_, ny_ = cx + dx, cy + dy
                    if not (0 <= nx_ < GNC and 0 <= ny_ < GNC):
                        continue
                    s_n = float(self.support(nx_, ny_)) + (0.3 if dx and dy else 0.0) + self.rng.uniform(0, 0.2)
                    if s_n < sup - self.n_steep * CELL and (best is None or s_n < best):
                        best, bxy = s_n, (nx_, ny_)
                if bxy is None:
                    break
                cx, cy = bxy
                hops += 1
                fx_, fy_ = cell_center(cx, cy)
                if self.H[cx, cy] <= max(self.WT[cx, cy], self.CH[cx, cy], float(self.floor_h(fx_, fy_))) + 1e-6:
                    break
            if hops:
                self.n_slides[a] += hops
                self.n_ix[a], self.n_iy[a] = cx, cy
                ncx, ncy = cell_center(cx, cy)
                self.n_xy[a] = (ncx, ncy)
                self.n_from[a] = start
                self.n_tslide[a] = self.t
                self._ev('slide', 1, pos)
                base = max(self.WT[cx, cy], self.CH[cx, cy], float(self.floor_h(ncx, ncy)))
                if self.H[cx, cy] <= base + 1e-6:
                    self._ev('land_ground' if self.WT[cx, cy] <= base else 'land_flesh', 1, np.array([ncx, ncy, base]))
                    if self.WT[cx, cy] <= base:
                        self.ground_hits.append(np.array([[ncx, ncy, base]]))
                    self._n_rest(a, cx, cy, base)
                    return
            self._ev('land_anvil', 1, pos)
            self._vmax('vmax_anvil', speed)
            self.sparks.append((pos.copy(), self.t))
            z = self.H[cx, cy]
        elif self.WT[cx, cy] >= self.CH[cx, cy] and self.WT[cx, cy] > float(self.floor_h(fx_, fy_)):
            self._ev('land_flesh', 1, pos)
            self._vmax('vmax_flesh', speed)
            self._crush(a, speed)
            fx_, fy_ = cell_center(cx, cy)
            z = max(self.WT[cx, cy], self.CH[cx, cy], self.H[cx, cy], float(self.floor_h(fx_, fy_)))
        elif self.CH[cx, cy] > float(self.floor_h(fx_, fy_)):
            self._ev('land_flesh', 1, pos)
            z = self.CH[cx, cy]
        else:
            self._ev('land_ground', 1, pos)
            self._vmax('vmax_ground', speed)
            z = float(self.floor_h(fx_, fy_))
            self.ground_hits.append(np.array([[pos[0], pos[1], z]]))
        if speed > 22.0 and self.rng.random() < 0.35:
            self.n_var[a] = min(2.0, self.n_var[a] + 1.0)
        self._n_rest(a, cx, cy, z)

    def _n_rest(self, a, cx, cy, z):
        self.n_z[a] = z
        self.n_vz[a] = 0.0
        self.n_state[a] = N_REST
        self.n_tland[a] = self.t
        self.stacks.setdefault((cx, cy), []).append(a)
        self.H[cx, cy] = z + AN.HEIGHT

    def _n_fall(self, dt):
        go = (self.n_state == N_WAIT) & (self.n_t0 <= self.t)
        if go.any():
            self.n_state[go] = N_FALL
            self._ev('release', int(go.sum()), np.c_[self.n_xy[go], self.n_z[go]][:: max(1, int(go.sum()) // 32)])
        fl = np.nonzero(self.n_state == N_FALL)[0]
        if len(fl) == 0:
            return
        v = self.n_vz[fl]
        self.n_z[fl] -= v * dt + 0.5 * (GN - DRAG * v) * dt * dt
        self.n_vz[fl] = v + (GN - DRAG * v) * dt
        sup = self.support(self.n_ix[fl], self.n_iy[fl])
        hit = self.n_z[fl] <= sup
        if not hit.any():
            return
        land = fl[hit]
        land = land[np.argsort(self.n_z[land], kind='stable')]
        for a in land:
            if self.n_state[a] != N_FALL:
                continue
            if self.n_z[a] > float(self.support(self.n_ix[a], self.n_iy[a])):
                continue
            self._n_land(a)

    def _check_stacks(self):
        for key in list(self.stacks.keys()):
            ids = self.stacks[key]
            if not ids:
                continue
            cx, cy = key
            fx_, fy_ = cell_center(cx, cy)
            base = max(self.WT[cx, cy], self.CH[cx, cy], float(self.floor_h(fx_, fy_)))
            ids = np.asarray([i for i in ids if self.n_state[i] == N_REST])
            if len(ids) == 0:
                self.stacks[key] = []
                self.H[cx, cy] = 0.0
                self.H_changed = True
                continue
            gap = self.n_z[ids].min() - base
            if gap <= 0.02:
                continue
            self.H_changed = True
            if gap < 0.6:
                self.n_z[ids] -= gap
                self.H[cx, cy] -= gap
                self._ev('settle', len(ids))
            else:
                self.n_state[ids] = N_FALL
                self.n_vz[ids] = 0.0
                self.stacks[key] = []
                self.H[cx, cy] = 0.0
                self._ev('stack_drop', len(ids))

    def _unstack(self, idx):
        """Anvils idx leave their stacks (blown or knocked away); anvils that were resting on them fall onto
        whatever is left underneath."""
        touched = set()
        for a in idx:
            key = (int(self.n_ix[a]), int(self.n_iy[a]))
            ids = self.stacks.get(key)
            if ids and a in ids:
                ids.remove(a)
                touched.add(key)
        for key in touched:
            ids = sorted(self.stacks[key], key=lambda i: self.n_z[i])
            keep = []
            for i in ids:
                if keep and self.n_z[i] > self.n_z[keep[-1]] + AN.HEIGHT + 0.05:
                    break
                keep.append(i)
            drop = ids[len(keep):]
            if drop:
                drop = np.asarray(drop)
                self.n_state[drop] = N_FALL
                self.n_vz[drop] = 0.0
            self.stacks[key] = keep
            self.H[key] = (self.n_z[keep[-1]] + AN.HEIGHT) if keep else 0.0
            self.H_changed = True

    def _n_free(self, dt):
        bl = np.nonzero(self.n_state == N_BLAST)[0]
        if len(bl) == 0:
            return
        self.n_v3[bl, 2] -= GN * dt
        self.n_p3[bl] += self.n_v3[bl] * dt
        self.n_q[bl] = integrate_quat(self.n_q[bl], self.n_w3[bl], dt)
        gh = self.ground_h(self.n_p3[bl, 0], self.n_p3[bl, 1])
        down = (self.n_p3[bl, 2] <= gh) & (self.n_v3[bl, 2] < 0)
        if down.any():
            d = bl[down]
            self.n_p3[d, 2] = gh[down] - 0.15
            self.n_state[d] = N_FREE
            self._ev('land_ground', len(d), self.n_p3[d])
            self.ground_hits.append(self.n_p3[d] * np.array([1.0, 1.0, 0.0]) + np.c_[np.zeros((len(d), 2)), gh[down]])

    def anvil_instances(self):
        vis = (self.n_state != N_WAIT) | (self.n_hover & (self.t >= self.n_appear))
        idx = np.nonzero(vis)[0]
        out = np.zeros((len(idx), 11), np.float32)
        p = np.c_[self.n_xy[idx], self.n_z[idx]]
        dur = np.clip(0.08 + 0.04 * self.n_slides[idx], 0.1, 0.5)
        u = np.clip((self.t - self.n_tslide[idx]) / dur, 0.0, 1.0)
        sl = u < 1.0
        if sl.any():
            e = u[sl]
            lerp = self.n_from[idx[sl]] * (1 - e[:, None]) + p[sl] * e[:, None]
            lerp[:, 2] += np.sin(np.pi * e) * 0.6
            p[sl] = lerp
        free = (self.n_state[idx] == N_BLAST) | (self.n_state[idx] == N_FREE)
        if free.any():
            p[free] = self.n_p3[idx[free]]
        out[:, 0:3] = p
        out[:, 3:7] = self.n_q[idx]
        out[:, 7] = 1.0
        out[:, 8] = self.n_var[idx]
        # hanging anvils fade in when they appear
        out[:, 9] = np.where(self.n_state[idx] == N_WAIT, np.clip((self.t - self.n_appear[idx]) / 0.3, 0.0, 1.0), 1.0)
        return out

    # ==========================================================================================
    # TNT and explosions
    # ==========================================================================================
    def _init_tnt(self, f):
        n = 0 if f is None else len(f['pos'])
        self.t_n = n
        self.t_p = _empty(n, 3) if n == 0 else f['pos'].astype(np.float64).copy()
        self.t_v = _empty(n, 3) if n == 0 else f['vel'].astype(np.float64).copy()
        self.t_t0 = np.zeros(n) if n == 0 else f['t0'].astype(np.float64)
        self.t_fuse = np.zeros(n) if n == 0 else f['fuse'].astype(np.float64)
        self.t_hover = np.zeros(n, bool) if n == 0 else f['hover'].astype(bool)
        self.t_appear = np.zeros(n) if n == 0 else np.asarray(f.get('appear', np.zeros(n)), np.float64)
        self.t_state = np.full(n, T_WAIT, np.int8)
        self.t_phase = self.rng.uniform(0.0, 2.0 * FLASH_PERIOD, n)      # they twinkle, not strobe
        self.t_flash_amp = 0.3 / (1.0 + 0.5 * np.log10(max(n, 1)))
        self.t_spin = self.rng.normal(0, 1.0, (n, 3))
        self.t_q = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))

    def _tnt_step(self, dt):
        go = (self.t_state == T_WAIT) & (self.t_t0 <= self.t)
        if go.any():
            self.t_state[go] = T_FALL
            self._ev('tnt_release', int(go.sum()), self.t_p[go][:: max(1, int(go.sum()) // 32)])
        fl = np.nonzero((self.t_state == T_FALL) | (self.t_state == T_BLOWN))[0]
        # fuses run out anywhere (also while hanging)
        wait_fuse = np.nonzero((self.t_state == T_WAIT) & (self.t >= self.t_fuse))[0]
        if len(wait_fuse):
            self.t_state[wait_fuse] = T_GONE
            self._explode(self.t_p[wait_fuse].copy())
        if len(fl) == 0:
            return
        v = self.t_v[fl]
        v[:, 2] -= GT * dt
        v *= (1.0 - 0.3 * dt)
        newp = self.t_p[fl] + v * dt
        self.t_v[fl] = v
        self.t_p[fl] = newp
        blown = self.t_state[fl] == T_BLOWN
        if blown.any():
            b = fl[blown]
            self.t_q[b] = integrate_quat(self.t_q[b], self.t_spin[b] * 6.0, dt)
        contact = np.zeros(len(fl), bool)
        for gi, g in enumerate(self.giants):
            near = np.nonzero(g.near(newp, 1.0))[0]
            if len(near) == 0:
                continue
            c = np.zeros(len(near), bool)
            for off in CONTACT_OFFS:
                c |= g.occupied_at(newp[near] + off)
            contact[near] |= c
        gh = self.ground_h(newp[:, 0], newp[:, 1])
        on_ground = newp[:, 2] - TNT_HALF < gh
        fuse = self.t >= self.t_fuse[fl]
        boom = contact | on_ground | fuse
        if boom.any():
            b = fl[boom]
            C = newp[boom].copy()
            C[:, 2] = np.maximum(C[:, 2], gh[boom] + TNT_HALF)
            self.t_state[b] = T_GONE
            n_c = int(contact[boom].sum())
            if n_c:
                self._ev('tnt_hit', n_c)
            self._explode(C)

    def _explode(self, C, R=R_EXP, crater=R_CRATER, vaporize=0.72, push=2.5, speed=(3.5, 10.0)):
        """Detonate explosions at centres C (E, 3)."""
        rng = self.rng
        E = len(C)
        if E == 0:
            return
        self.frame_explosions.append(C.copy())
        self._ev('explode', E, C[:: max(1, E // 64)])
        # 1. blast the giants
        for gi, g in enumerate(self.giants):
            near = g.near(C, R * 1.2)
            if not near.any():
                continue
            Cn = C[near]
            S, Sd = self.ball(R * 1.12, g.VS)
            ic, jc, kc = g.world_to_index(Cn)
            ci = ic[:, None] + S[None, :, 0]
            cj = jc[:, None] + S[None, :, 1]
            ck = kc[:, None] + S[None, :, 2]
            inb = g.inb(ci, cj, ck)
            thr = R * rng.uniform(0.8, 1.12, inb.shape)
            ci0, cj0, ck0 = np.where(inb, ci, 0), np.where(inb, cj, 0), np.where(inb, ck, 0)
            hit = inb & (Sd[None, :] <= thr) & g.occ[ci0, cj0, ck0]
            if not hit.any():
                continue
            src = np.nonzero(hit)[0]
            vi, vj, vk = ci0[hit], cj0[hit], ck0[hit]
            flat = np.ravel_multi_index((vi, vj, vk), g.occ.shape)
            flat, first = np.unique(flat, return_index=True)
            vi, vj, vk = np.unravel_index(flat, g.occ.shape)
            src = src[first]
            self._blast_voxels(gi, vi, vj, vk, Cn[src], R, vaporize, speed)
        # 2. crater the ground
        low = C[:, 2] < crater + 0.6
        if low.any():
            changes = self.ground.blast(C[low], crater)
            if changes:
                self._ground_changed(changes, C[low])
        # 3. push TNT, throw anvils and knock arrows about
        self._impulse(C, PUSH_R, push)

    def _blast_voxels(self, gi, vi, vj, vk, c, R, vaporize, speed):
        """Voxels (vi, vj, vk) of giant gi blown out by explosions centred at c (one centre per voxel)."""
        rng = self.rng
        g = self.giants[gi]
        n = len(vi)
        p = g.index_to_world(vi, vj, vk)
        away = p - c
        dist = np.linalg.norm(away, axis=1)
        away /= np.maximum(dist, 1e-6)[:, None]
        axis = np.stack([np.full(n, g.center[0]), np.full(n, g.center[1]), p[:, 2]], -1)
        out = c - axis
        out[:, 2] *= 0.3
        out = _normalize(out)
        dirv = out * 1.0 + away * 0.35 + rng.normal(0, 0.5, (n, 3))
        dirv[:, 2] += 0.45
        dirv = _normalize(dirv)
        lo_s, hi_s = speed
        spd = rng.uniform(lo_s, hi_s, n) * np.sqrt(np.clip(1.2 - dist / R, 0.15, 1.0))
        vel = dirv * spd[:, None]
        vel[:, 2] += rng.uniform(0.5, 3.5, n)
        gone = rng.random(n) < vaporize * np.clip(1.3 - dist / R, 0.3, 1.0)
        if gone.any():
            g.occ[vi[gone], vj[gone], vk[gone]] = False
            self.destroyed[gi] += int(gone.sum())
            self.dirty[gi] += int(gone.sum())
        keep = ~gone
        self._destroy(gi, vi[keep], vj[keep], vk[keep], vel[keep])
        self._ev('blast', n, p[:: max(1, n // 64)])
        core = int(g.bone[vi, vj, vk].sum())
        if core:
            self._ev('powder' if g.kind == 'creeper' else 'bone', core)

    def _impulse(self, C, radius, strength):
        rng = self.rng
        # TNT
        fl = np.nonzero((self.t_state == T_FALL) | (self.t_state == T_WAIT) | (self.t_state == T_BLOWN))[0]
        for c in C[:: max(1, len(C) // 48)]:
            if len(fl):
                dv = self.t_p[fl] - c
                d = np.linalg.norm(dv, axis=1)
                s = d < radius
                if s.any():
                    k = fl[s]
                    imp = strength * (1.0 - d[s] / radius) ** 2
                    self.t_v[k] += dv[s] / np.maximum(d[s], 1e-6)[:, None] * imp[:, None]
                    self.t_state[k] = np.where(self.t_state[k] == T_WAIT, T_FALL, self.t_state[k])
            # anvils resting or falling close to the blast are thrown
            an = np.nonzero((self.n_state == N_REST) | (self.n_state == N_FALL) | (self.n_state == N_FREE))[0]
            if len(an):
                pa = np.c_[self.n_xy[an], self.n_z[an] + AN.HEIGHT * 0.5]
                fr = self.n_state[an] == N_FREE
                if fr.any():
                    pa[fr] = self.n_p3[an[fr]]
                dv = pa - c
                d = np.linalg.norm(dv, axis=1)
                s = d < radius * 0.8
                if s.any():
                    k = an[s]
                    self._unstack(k[self.n_state[k] == N_REST])
                    self.n_p3[k] = pa[s] - np.array([0.0, 0.0, AN.HEIGHT * 0.5])
                    dirv = _normalize(dv[s] + np.array([0.0, 0.0, 1.2]))
                    self.n_v3[k] = dirv * rng.uniform(5.0, 12.0, (len(k), 1)) * (1.0 - d[s] / radius)[:, None] * 1.6
                    self.n_w3[k] = rng.normal(0, 5.0, (len(k), 3))
                    self.n_state[k] = N_BLAST
                    self._ev('anvil_thrown', len(k), pa[s])
            # arrows lying about are knocked flying
            ar = np.nonzero((self.a_state == A_GROUND) | (self.a_state == A_REST))[0]
            if len(ar):
                dv = self.a_p[ar] - c
                d = np.linalg.norm(dv, axis=1)
                s = d < radius
                if s.any():
                    k = ar[s]
                    self.a_state[k] = A_FALL
                    self.a_v[k] = _normalize(dv[s] + np.array([0.0, 0.0, 1.5])) * rng.uniform(4.0, 10.0, (len(k), 1))
                    self.a_w[k] = rng.normal(0, 8.0, (len(k), 3))

    def _ground_changed(self, changes, centres):
        rng = self.rng
        ch = np.array(changes)
        n_blocks = int((ch[:, 2] - ch[:, 3]).sum())
        self._ev('ground', n_blocks, np.stack([ch[:, 0] - REG + 0.5, ch[:, 1] - REG + 0.5, ch[:, 3]], -1)[:: max(1, len(ch) // 32)])
        pos, vel, cols = [], [], []
        for ix, iy, old, new in changes:
            for k in range(new, old):
                if rng.random() > 0.3:
                    continue
                p = np.array([ix - REG + rng.uniform(0.2, 0.8), iy - REG + rng.uniform(0.2, 0.8), k + 0.5])
                c = centres[np.argmin(np.linalg.norm(centres - p, axis=1))]
                d = p - c
                d[2] = abs(d[2]) + 0.6
                d /= max(np.linalg.norm(d), 1e-6)
                pos.append(p)
                vel.append(d * rng.uniform(7.0, 15.0) + rng.normal(0, 1.5, 3))
                cols.append(block_colours(k))
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
        self.ground_changed = True
        changed = np.zeros((2 * REG, 2 * REG), bool)
        span = int(round(1.0 / PILE_RES))
        for ix, iy, old, new in changes:
            px0, py0 = self._pile_idx(np.array([ix - REG + 0.01]), np.array([iy - REG + 0.01]))
            self.pile[px0[0]:px0[0] + span, py0[0]:py0[0] + span] = new
            changed[ix, iy] = True
        # wake debris, arrows and anvils resting on the changed ground
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
                    self.dv[w] = rng.normal(0, 0.5, (len(w), 3))
        ar = np.nonzero((self.a_state == A_GROUND) | (self.a_state == A_REST))[0]
        if len(ar):
            cx = np.floor(self.a_p[ar, 0]).astype(np.int64) + REG
            cy = np.floor(self.a_p[ar, 1]).astype(np.int64) + REG
            ok = (cx >= 0) & (cx < 2 * REG) & (cy >= 0) & (cy < 2 * REG)
            wake = np.zeros(len(ar), bool)
            wake[ok] = changed[cx[ok], cy[ok]]
            if wake.any():
                w = ar[wake]
                self.a_state[w] = A_FALL
                self.a_v[w] = rng.normal(0, 1.0, (len(w), 3))
                self.a_w[w] = rng.normal(0, 3.0, (len(w), 3))
        fr = np.nonzero(self.n_state == N_FREE)[0]
        if len(fr):
            gh = self.ground_h(self.n_p3[fr, 0], self.n_p3[fr, 1])
            drop = self.n_p3[fr, 2] > gh + 0.3
            if drop.any():
                d = fr[drop]
                self.n_state[d] = N_BLAST
                self.n_v3[d] = 0.0

    def tnt_instances(self):
        vis = ((self.t_state == T_WAIT) & self.t_hover & (self.t >= self.t_appear)) | (self.t_state == T_FALL) | \
            (self.t_state == T_BLOWN)
        idx = np.nonzero(vis)[0]
        out = np.zeros((len(idx), 11), np.float32)
        if len(idx) == 0:
            return out
        out[:, 0:3] = self.t_p[idx]
        out[:, 3:7] = self.t_q[idx]
        left = self.t_fuse[idx] - self.t
        swell = np.clip(1.0 - left / 0.5, 0, 1) ** 4
        out[:, 7] = 1.0 + 0.3 * swell
        out[:, 9] = np.where(self.t_state[idx] == T_WAIT, np.clip((self.t - self.t_appear[idx]) / 0.3, 0.0, 1.0), 1.0)
        ph = np.floor((self.t + self.t_phase[idx]) / FLASH_PERIOD).astype(np.int64)
        out[:, 10] = np.where(ph % 2 == 1, self.t_flash_amp, 0.0)
        return out

    # ==========================================================================================
    # the creeper, sonic booms, severs
    # ==========================================================================================
    def creeper_fx(self):
        """(white flash 0..1, swell) of the priming creeper."""
        c = self.creeper
        if not c or c['done'] or self.t < c['prime']:
            return 0.0, 1.0
        u = np.clip((self.t - c['prime']) / (c['explode'] - c['prime']), 0.0, 1.0)
        # flashes faster and faster, swells towards the end (like the game)
        period = 0.42 - 0.3 * u
        phase = (self.t - c['prime']) / period
        white = 0.62 if (phase % 1.0) < 0.5 else 0.0
        return white * (0.45 + 0.55 * u), 1.0 + 0.32 * u ** 3

    def _creeper_step(self):
        c = self.creeper
        if not c or c['done']:
            return
        if not c.get('hissing') and self.t >= c['prime']:
            c['hissing'] = True
            self._ev('hiss', 1, self.giants[c['giant']].center)
        if self.t < c['explode']:
            return
        c['done'] = True
        gi = c['giant']
        g = self.giants[gi]
        rng = self.rng
        centre = g.center.copy()
        centre[2] = g.hi[2] * 0.5
        # the creeper itself goes up in the blast
        i, j, k = np.nonzero(g.occ)
        n = len(i)
        p = g.index_to_world(i, j, k)
        d = _normalize(p - centre + rng.normal(0, 0.3, (n, 3)))
        vel = d * rng.uniform(5.0, 15.0, (n, 1))
        vel[:, 2] += rng.uniform(2.0, 7.0, n)
        gone = rng.random(n) < c.get('vaporize', 0.88)
        g.occ[i[gone], j[gone], k[gone]] = False
        self.destroyed[gi] += int(gone.sum())
        keep = ~gone
        self._destroy(gi, i[keep], j[keep], k[keep], vel[keep])
        g.alive = False
        self.ground.release(gi)
        # its neighbours lose whatever is inside the blast
        R = c['radius']
        for oi, o in enumerate(self.giants):
            if oi == gi or not o.occ.any():
                continue
            if np.any(o.lo > centre + R) or np.any(o.hi < centre - R):
                continue
            vi, vj, vk = np.nonzero(o.occ)
            q = o.index_to_world(vi, vj, vk)
            dist = np.linalg.norm(q - centre, axis=1)
            thr = R * (0.82 + 0.3 * np.abs(np.sin(q[:, 2] * 1.7 + q[:, 1] * 2.3)))
            hit = dist < thr
            if hit.any():
                self._blast_voxels(oi, vi[hit], vj[hit], vk[hit], np.tile(centre, (int(hit.sum()), 1)), R, 0.8,
                                   (4.0, 10.0))
        changes = self.ground.blast(np.array([[centre[0], centre[1], 1.5]]), c.get('crater', 7.0))
        if changes:
            self._ground_changed(changes, np.array([[centre[0], centre[1], 1.5]]))
        self._impulse(np.array([centre]), R * 1.3, 14.0)
        self.frame_explosions.append(np.array([centre]))
        self.big_blasts.append((centre + np.array([0.0, 0.0, 2.0]), c.get('fx_scale', 3.2)))
        self._ev('creeper_boom', 1, centre)
        self.dirty[:] += 40

    def _booms(self):
        for b in self.booms:
            if b['fired'] or self.t < b['t']:
                continue
            b['fired'] = True
            o, d = b['o'], b['d']
            fl = np.nonzero((self.t_state == T_FALL) | (self.t_state == T_WAIT))[0]
            if len(fl):
                rel = self.t_p[fl] - o
                s = rel @ d
                perp = rel - s[:, None] * d
                dist = np.linalg.norm(perp, axis=1)
                hit = (s > 0) & (s < b.get('reach', 70.0)) & (dist < b['r0'] + 0.06 * s)
                k = fl[hit]
                if len(k):
                    # flung high up and a little outwards: they go off in the sky like fireworks
                    out = _normalize(perp[hit] + self.rng.normal(0, 0.3, (len(k), 3)))
                    self.t_v[k] = out * self.rng.uniform(3, 8, (len(k), 1)) + d * self.rng.uniform(16, 30, (len(k), 1))
                    self.t_state[k] = T_BLOWN
                    self.t_fuse[k] = self.t + self.rng.uniform(0.5, 1.4, len(k))
                    self._ev('boom_blast', len(k), self.t_p[k].mean(0))
            self._ev('boom', 1, o)

    def rings(self):
        out = []
        for b in self.booms:
            if not b['fired']:
                continue
            age0 = self.t - b['t']
            for i in range(28):
                s = 1.5 + i * 2.2
                a = age0 - s / 110.0
                if a < 0 or a > 0.45:
                    continue
                alpha = (1.0 - a / 0.45) ** 1.5
                out.append([*(b['o'] + b['d'] * s), *b['d'], 1.6 + 2.2 * (a / 0.45) + 0.02 * s, alpha])
        return np.array(out, np.float32).reshape(-1, 8)

    # ==========================================================================================
    def step_frame(self, scale=1.0):
        """Advance one video frame; scale < 1 is slow motion."""
        self.events = {}
        self.ground_hits = []
        self.frame_explosions = []
        self.big_blasts = []
        n_sub = max(1, int(np.ceil(SUB * scale - 1e-9)))
        dt = scale / FPS / n_sub
        wt0 = self.WT.copy()
        for _ in range(n_sub):
            while self.severs and self.severs[0][0] <= self.t:
                sv = self.severs.pop(0)
                self.sever(*sv[1:])
            self._booms()
            self._creeper_step()
            if self.a_n:
                self._a_fly(dt)
                self._a_fall(dt)
            if self.n_n:
                self._n_fall(dt)
                self._n_free(dt)
            if self.t_n:
                self._tnt_step(dt)
            self._debris_step(dt)
            self._chunks_step(dt)
            self.t += dt
        self._structure()
        if self.a_n:
            self._a_check_anchors()
        if self.n_n and (self.dirty.any() or not np.array_equal(wt0, self.WT) or self.ground_changed):
            self._check_stacks()
        if (self.H_changed or self.ground_changed) and self._wake_unsupported() and self.n_n:
            self._check_stacks()
        self.H_changed = False
        self.ground_changed = False
        self.dust.step(np.concatenate(self.ground_hits) if self.ground_hits else np.zeros((0, 3)), scale / FPS)
        C = np.concatenate(self.frame_explosions) if self.frame_explosions else np.zeros((0, 3))
        self.vfx.step(C, scale / FPS, self.big_blasts)
        self.dt_frame = scale / FPS
        for name, st in (('flying', self.a_state == A_FLY), ('falling', self.n_state == N_FALL),
                         ('tnt_falling', self.t_state == T_FALL)):
            k = int(st.sum())
            if k:
                p = (self.a_p if name == 'flying' else (self.t_p if name == 'tnt_falling' else
                                                        np.c_[self.n_xy, self.n_z]))[st]
                self._ev(name, k, p[:: max(1, k // 64)])
        hang = int(((self.t_state == T_WAIT) & self.t_hover & (self.t >= self.t_appear)).sum())
        if hang:
            self._ev('tnt_hanging', hang)
        self.sparks = [s for s in self.sparks if self.t - s[1] < 0.12]
        self.frame += 1
        return self.events

    # ==========================================================================================
    def instances(self):
        """Voxels (standing giants first, then pieces and debris), their instance ranges, props, arrows and
        effects for the renderer."""
        parts = []
        ranges = []
        n0 = 0
        for gi, g in enumerate(self.giants):
            key = int(self.destroyed[gi])
            if self._packs.get(gi, (None,))[0] != key:
                self._packs[gi] = (key, g.pack_static() if g.occ.any() else np.zeros(0, VOXEL_DTYPE))
            inst = self._packs[gi][1]
            parts.append(inst)
            ranges.append((n0, n0 + len(inst)))
            n0 += len(inst)
        for c in self.chunks:
            n = len(c['local'])
            inst = c['col'].copy()
            inst['pos'] = quat_rotate(np.tile(c['q'], (n, 1)), c['local']) + c['pos']
            inst['quat'] = c['q']
            inst['scale'] = c['vs']
            parts.append(inst)
        if len(self.dp):
            inst = self.dcol.copy()
            inst['pos'] = self.dp
            inst['quat'] = self.dq
            inst['scale'] = self.dscale
            parts.append(inst)
        vox = np.concatenate(parts)
        props = {}
        if self.n_n:
            props['anvil'] = self.anvil_instances()
        if self.t_n:
            props['tnt'] = self.tnt_instances()
        arrows = self.arrow_instances() if self.a_n else None
        puffs = [self.dust.puffs(), self.vfx.puffs()]
        fx = {'puffs': np.concatenate(puffs), 'flashes': np.concatenate([self.vfx.flashes(), self._spark_flashes()]),
              'lights': self.vfx.lights(), 'streaks': self.arrow_streaks() if self.a_n else None,
              'rings': self.rings()}
        return vox, ranges, props, arrows, fx

    def _spark_flashes(self):
        if not self.sparks:
            return np.zeros((0, 5), np.float32)
        out = []
        for p, t in self.sparks:
            a = 1.0 - (self.t - t) / 0.12
            out.append([p[0], p[1], p[2] + 0.05, 0.9, max(a, 0.0) * 0.6])
        return np.array(out, np.float32)
