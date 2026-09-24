"""The hydraulic press test, simulated: blocks are placed on the platen one after another and the ram comes down
on each. Every material does its own thing: grass crumbles, glass shatters, a melon bursts, slime squashes flat and
throws the ram back up (then gets splatted), TNT primes and blows up, a chest bursts and spills its loot, diamond
and obsidian crack stage by stage before they give, and bedrock does not give at all: the press breaks instead.

Broken blocks become rigid voxel chunks and single-voxel crumbs that fly out, collide with the platen, the press
base, the floor, the columns and the ram (which squeezes whatever is under it outwards), and settle. Small stuff
(juice, sparkles, purple portal bits, oil) are point particles; sparks are streaks. Stepped per video frame in
sub-steps, so slow motion stays smooth and a render is deterministic for a given schedule.
"""
import numpy as np

import items as IT
import pixelart as PA
import scene as SC
from explosion_vfx import VFX
from mathutil import quat_rotate, integrate_quat
from vfx import Dust

FPS = 30
SUB = 4
GRAV = 36.0
H = IT.B
Z0 = SC.Z0
REST = SC.RAM_REST
ZC = Z0 + H                     # ram bottom at first contact


def _ease(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def support_height(x, y, block=None):
    """Top of whatever is under (x, y): the platen, the press base, or the floor; block: the top of an intact
    block standing on the platen (its 4 x 4 footprint), if there is one."""
    x = np.asarray(x)
    y = np.asarray(y)
    h = np.zeros(np.shape(x))
    bx0, bx1, by0, by1, _, bz1 = SC.BASE
    h = np.where((x > bx0) & (x < bx1) & (y > by0) & (y < by1), bz1, h)
    px0, px1, py0, py1, _, pz1 = SC.PLATEN
    h = np.where((x > px0) & (x < px1) & (y > py0) & (y < py1), pz1, h)
    for (cx0, cx1) in SC.COLS:
        h = np.where((x > cx0 - 0.5) & (x < cx1 + 0.5) & (y > SC.COL_Y[0] - 0.5) & (y < SC.COL_Y[1] + 0.5), 2.0, h)
    if block is not None:
        h = np.where((np.abs(x) < H / 2) & (np.abs(y) < H / 2), np.maximum(h, block), h)
    return h


# ---------------------------------------------------------------------------------------------
# materials
# ---------------------------------------------------------------------------------------------
MATS = {
    'grass': dict(s_break=0.6, push=0.5, chunks=14, crumb=0.6, speed=(4.0, 10.0), up=(0.5, 3.0), bulge=0.22,
                  dust=True, tons=6),
    'glass': dict(s_break=0.985, push=0.1, chunks=46, crumb=0.3, speed=(8.0, 17.0), up=(1.0, 6.0), bulge=0.0,
                  sparkle=40, tons=2),
    'melon': dict(s_break=0.76, push=0.42, chunks=12, crumb=0.45, speed=(6.0, 13.0), up=(1.0, 5.0), bulge=0.3,
                  juice=320, tons=9),
    'slime': dict(s_break=0.3, push=0.5, chunks=12, crumb=0.65, speed=(9.0, 17.0), up=(1.0, 6.0), bulge=0.5,
                  goo=200, tons=20),
    'tnt': dict(s_break=0.8, push=0.38, chunks=24, crumb=0.5, speed=(16.0, 30.0), up=(2.0, 12.0), bulge=0.12,
                prime=0.8, tons=15),
    'chest': dict(s_break=0.72, push=0.45, chunks=10, crumb=0.3, speed=(5.0, 11.0), up=(2.0, 7.0), bulge=0.15,
                  loot=14, tons=25),
    'diamond': dict(s_break=0.97, push=1.15, chunks=38, crumb=0.2, speed=(9.0, 16.0), up=(1.0, 7.0), bulge=0.0,
                    crack=True, sparkle=90, pop=4, tons=400),
    'obsidian': dict(s_break=0.985, push=1.8, chunks=22, crumb=0.2, speed=(4.0, 9.0), up=(0.5, 3.5), bulge=0.0,
                     crack=True, purple=160, sparks=True, tons=2500),
    'bedrock': dict(unbreakable=True, strain=3.2, tons=100000),
}


def event_offsets(test):
    """When things happen in a test, in seconds after its t0 (mirrors the scripts below): contact, break (the block
    gives; for bedrock the press does), and per material: bounce (slime), prime (TNT), hoses (bedrock)."""
    kind = test['kind']
    m = MATS[kind]
    d1 = test.get('wait', 0.35) + test.get('descend', 0.32)
    ev = {'contact': d1}
    if kind == 'slime':
        ev['bounce'] = d1 + 0.55 + 0.35
        ev['break'] = ev['bounce'] + 0.75 + 0.12
    elif kind == 'tnt':
        ev['prime'] = d1 + m['push']
        ev['break'] = ev['prime'] + m['prime']
    elif kind == 'bedrock':
        ev['hoses'] = d1 + 0.55 * m['strain']
        ev['break'] = d1 + m['strain']
    else:
        ev['break'] = d1 + test.get('push', m['push'])
    return ev


# ---------------------------------------------------------------------------------------------
# rigid fragments
# ---------------------------------------------------------------------------------------------
class Frags:
    """Rigid chunks of voxels (a crumb is a chunk of one voxel)."""

    def __init__(self):
        self.p = np.zeros((0, 3))
        self.v = np.zeros((0, 3))
        self.q = np.zeros((0, 4))
        self.w = np.zeros((0, 3))
        self.rest = np.zeros(0, bool)
        self.tmpl = np.zeros(0, IT.VOXEL_DTYPE)
        self.owner = np.zeros(0, np.int64)
        self.rel = np.zeros((0, 3))
        self.half = np.zeros(0)                 # half edge of each voxel

    def __len__(self):
        return len(self.p)

    def add(self, vox, idx, group, wpos, vel, spin, rng, scale=None):
        """vox: the VoxelSet; idx: its voxels that break loose; group: chunk label per voxel (same label = same
        rigid piece); wpos: their world positions now; vel(centres) -> velocities; spin: angular speed scale."""
        if len(idx) == 0:
            return
        labels, inv = np.unique(group, return_inverse=True)
        m = len(labels)
        cnt = np.bincount(inv, minlength=m).astype(float)
        cen = np.zeros((m, 3))
        np.add.at(cen, inv, wpos)
        cen /= cnt[:, None]
        base = len(self.p)
        self.p = np.concatenate([self.p, cen])
        self.v = np.concatenate([self.v, vel(cen)])
        self.q = np.concatenate([self.q, np.tile([0.0, 0.0, 0.0, 1.0], (m, 1))])
        w = rng.normal(0, 1, (m, 3)) * spin / np.sqrt(np.maximum(cnt, 1.0))[:, None] ** 0.35
        self.w = np.concatenate([self.w, w])
        self.rest = np.concatenate([self.rest, np.zeros(m, bool)])
        alive = np.zeros(vox.n, bool)
        alive[idx] = True
        gfull = np.full(vox.n, -1, np.int64)
        gfull[idx] = inv + 1
        vis = vox.visibility(alive, gfull)
        inst = vox.instances(idx, vis=vis)
        if scale is not None:
            inst['scale'] = scale
        self.tmpl = np.concatenate([self.tmpl, inst])
        self.owner = np.concatenate([self.owner, base + inv])
        self.rel = np.concatenate([self.rel, wpos - cen[inv]])
        self.half = np.concatenate([self.half, 0.5 * inst['scale'].astype(float)])

    def world(self):
        qo = self.q[self.owner]
        return self.p[self.owner] + quat_rotate(qo, self.rel), qo

    def instances(self):
        if len(self.tmpl) == 0:
            return np.zeros(0, IT.VOXEL_DTYPE)
        out = self.tmpl.copy()
        pw, qo = self.world()
        out['pos'] = pw
        out['quat'] = qo
        return out

    def remove_on(self, x0, x1, y0, y1, z0, z1):
        """Clear what lies in a box (the platen before the next block goes on)."""
        if len(self.p) == 0:
            return
        c = self.p
        gone = (c[:, 0] > x0) & (c[:, 0] < x1) & (c[:, 1] > y0) & (c[:, 1] < y1) & (c[:, 2] > z0) & (c[:, 2] < z1)
        if not gone.any():
            return
        keep_c = ~gone
        remap = np.cumsum(keep_c) - 1
        kv = keep_c[self.owner]
        for a in ('p', 'v', 'q', 'w', 'rest'):
            setattr(self, a, getattr(self, a)[keep_c])
        self.tmpl = self.tmpl[kv]
        self.rel = self.rel[kv]
        self.half = self.half[kv]
        self.owner = remap[self.owner[kv]]

    def step(self, dt, ram, events, block=None):
        """Integrate and collide with the world. ram: (z_bottom, vz, present); block: top of an intact block."""
        n = len(self.p)
        if n == 0:
            return
        act = ~self.rest
        zr, vzr, ram_on = ram
        # the ram wakes up what it comes down on
        if ram_on:
            near = self.rest & (np.abs(self.p[:, 0]) < 6.0) & (np.abs(self.p[:, 1]) < 6.0) & (self.p[:, 2] > zr - 3.0)
            act |= near
            self.rest &= ~near
        if not act.any():
            return
        a = np.nonzero(act)[0]
        self.v[a, 2] -= GRAV * dt
        self.v[a] *= (1.0 - 0.25 * dt)
        self.p[a] += self.v[a] * dt
        self.q[a] = integrate_quat(self.q[a], self.w[a], dt)
        # voxel-level contact against the support below and the ram above
        vm = act[self.owner]
        own = self.owner[vm]
        pw = self.p[own] + quat_rotate(self.q[own], self.rel[vm])
        hv = self.half[vm]
        sup = support_height(pw[:, 0], pw[:, 1], block)
        pen = sup - (pw[:, 2] - hv)
        up = np.zeros(n)
        np.maximum.at(up, own, pen)
        supmax = np.full(n, -1e9)
        np.maximum.at(supmax, own, sup)
        down = np.zeros(n)
        if ram_on:
            under = (np.abs(pw[:, 0]) < SC.RAM_W / 2) & (np.abs(pw[:, 1]) < SC.RAM_W / 2) & (pw[:, 2] < zr + 1.0)
            pr = np.where(under, (pw[:, 2] + hv) - zr, 0.0)
            np.maximum.at(down, own, pr)
        hit = act & (up > 0.0)
        if hit.any():
            h = np.nonzero(hit)[0]
            vz = self.v[h, 2].copy()
            self.p[h, 2] += up[h]
            fast = vz < -3.0
            self.v[h, 2] = np.where(fast, -vz * 0.28, 0.0)
            self.v[h, :2] *= 0.72
            self.w[h] *= 0.6
            if fast.any():
                sp = -vz[fast]
                e = events.setdefault('land', [0, 0.0])
                e[0] += int(fast.sum())
                e[1] = max(e[1], float(sp.max()))
            calm = (~fast) & (np.linalg.norm(self.v[h], axis=1) < 1.2)
            # hanging over an edge with its middle: it tips off instead of resting there
            over = support_height(self.p[h, 0], self.p[h, 1], block) < supmax[h] - 0.5
            tip = calm & over
            if tip.any():
                ht = h[tip]
                self.v[ht, :2] += _unit(self.p[ht, :2] + 1e-3) * 3.0
                self.w[ht] += np.random.default_rng(len(ht)).normal(0, 2.0, (len(ht), 3))
            calm &= ~over
            self.rest[h[calm]] = True
            self.v[h[calm]] = 0.0
            self.w[h[calm]] = 0.0
        sq = act & (down > 0.0)
        if sq.any():
            s = np.nonzero(sq)[0]
            self.p[s, 2] -= down[s]
            self.v[s, 2] = np.minimum(self.v[s, 2], vzr)
            # squeezed between the ram and the platen: squirts out sideways
            trapped = s[up[s] > 0.0]
            if len(trapped):
                out = _unit(self.p[trapped, :2] + 1e-3)
                self.v[trapped, :2] += out * (6.0 + 1.5 * abs(vzr))
                self.rest[trapped] = False
        # the columns and the walls
        for (cx0, cx1) in SC.COLS:
            inside = act & (self.p[:, 0] > cx0 - 0.6) & (self.p[:, 0] < cx1 + 0.6) & \
                (self.p[:, 1] > SC.COL_Y[0] - 0.6) & (self.p[:, 1] < SC.COL_Y[1] + 0.6) & (self.p[:, 2] < SC.BEAM[4])
            if inside.any():
                k = np.nonzero(inside)[0]
                left = self.p[k, 0] < 0.5 * (cx0 + cx1)
                self.p[k, 0] = np.where(left, cx0 - 0.6, cx1 + 0.6)
                self.v[k, 0] = -self.v[k, 0] * 0.3
        self.p[:, 1] = np.clip(self.p[:, 1], -60.0, 21.0)
        self.p[:, 0] = np.clip(self.p[:, 0], -43.0, 43.0)


# ---------------------------------------------------------------------------------------------
# point particles: juice, goo, sparkles, purple portal bits, oil
# ---------------------------------------------------------------------------------------------
class Bits:
    def __init__(self):
        self.p = np.zeros((0, 3))
        self.v = np.zeros((0, 3))
        self.col = np.zeros((0, 3))
        self.size = np.zeros(0)
        self.life = np.zeros(0)
        self.age = np.zeros(0)
        self.glow = np.zeros(0, bool)
        self.grav = np.zeros(0)
        self.stick = np.zeros(0, bool)          # stays where it lands (juice, oil) instead of fading

    def add(self, p, v, col, size, life, glow=False, grav=1.0, stick=False):
        n = len(p)
        if n == 0:
            return
        self.p = np.concatenate([self.p, p])
        self.v = np.concatenate([self.v, v])
        self.col = np.concatenate([self.col, np.broadcast_to(np.asarray(col, float), (n, 3))])
        self.size = np.concatenate([self.size, np.broadcast_to(size, (n,))])
        self.life = np.concatenate([self.life, np.broadcast_to(life, (n,))])
        self.age = np.concatenate([self.age, np.zeros(n)])
        self.glow = np.concatenate([self.glow, np.broadcast_to(glow, (n,))])
        self.grav = np.concatenate([self.grav, np.broadcast_to(grav, (n,))])
        self.stick = np.concatenate([self.stick, np.broadcast_to(stick, (n,))])

    def remove_on(self, x0, x1, y0, y1, z0, z1):
        if len(self.p) == 0:
            return
        c = self.p
        keep = ~((c[:, 0] > x0) & (c[:, 0] < x1) & (c[:, 1] > y0) & (c[:, 1] < y1) & (c[:, 2] > z0) & (c[:, 2] < z1))
        for a in ('p', 'v', 'col', 'size', 'life', 'age', 'glow', 'grav', 'stick'):
            setattr(self, a, getattr(self, a)[keep])

    def step(self, dt, block=None):
        if len(self.p) == 0:
            return
        self.age += dt
        keep = self.age < self.life
        for a in ('p', 'v', 'col', 'size', 'life', 'age', 'glow', 'grav', 'stick'):
            setattr(self, a, getattr(self, a)[keep])
        if len(self.p) == 0:
            return
        moving = np.linalg.norm(self.v, axis=1) > 0.0
        self.v[moving, 2] -= GRAV * self.grav[moving] * dt
        self.v *= (1.0 - 0.6 * dt)
        self.p += self.v * dt
        sup = support_height(self.p[:, 0], self.p[:, 1], block)
        low = self.p[:, 2] - self.size * 0.5 < sup
        if low.any():
            self.p[low, 2] = sup[low] + self.size[low] * 0.5
            self.v[low] = np.where(self.stick[low, None], 0.0, self.v[low] * np.array([0.5, 0.5, -0.2]))

    def instances(self):
        n = len(self.p)
        out = np.zeros(n, IT.VOXEL_DTYPE)
        if n == 0:
            return out
        u = self.age / self.life
        fade = np.where(self.stick, 1.0, np.clip(1.0 - u, 0.0, 1.0) ** 0.6)
        out['pos'] = self.p
        out['quat'] = (0, 0, 0, 1)
        out['scale'] = self.size * fade
        c = np.clip(self.col, 0, 255).astype(np.uint8)
        for k in ('cx', 'cy', 'cz', 'inner'):
            out[k][:, :3] = c
        out['cx'][:, 3] = 0b111111
        out['cy'][:, 3] = 0b111111
        out['cz'][:, 3] = np.where(self.glow, 255, 0)
        return out


class Sparks:
    def __init__(self):
        self.p = np.zeros((0, 3))
        self.v = np.zeros((0, 3))
        self.life = np.zeros(0)
        self.age = np.zeros(0)

    def add(self, p, v, life):
        self.p = np.concatenate([self.p, p])
        self.v = np.concatenate([self.v, v])
        self.life = np.concatenate([self.life, life])
        self.age = np.concatenate([self.age, np.zeros(len(p))])

    def step(self, dt, block=None):
        if len(self.p) == 0:
            return
        self.age += dt
        keep = self.age < self.life
        self.p, self.v, self.life, self.age = self.p[keep], self.v[keep], self.life[keep], self.age[keep]
        self.v[:, 2] -= GRAV * 0.8 * dt
        self.p += self.v * dt
        low = self.p[:, 2] < support_height(self.p[:, 0], self.p[:, 1], block)
        self.v[low] *= np.array([0.5, 0.5, -0.35])

    def streaks(self):
        """(N, 8): head3, tail3, width, alpha."""
        n = len(self.p)
        if n == 0:
            return np.zeros((0, 8), np.float32)
        out = np.zeros((n, 8), np.float32)
        out[:, 0:3] = self.p
        out[:, 3:6] = self.p - self.v * 0.035
        out[:, 6] = 0.07
        out[:, 7] = np.clip(1.0 - self.age / self.life, 0.0, 1.0)
        return out


# ---------------------------------------------------------------------------------------------
# the show
# ---------------------------------------------------------------------------------------------
class Press:
    def __init__(self, plan, seed=3):
        self.plan = plan
        self.rng = np.random.default_rng(seed)
        self.items_def = PA.make_items()
        _, mats = PA.studio_textures()
        self.tests = plan['tests']
        self.t = 0.0
        self.frame = 0
        self.frags = Frags()
        self.bits = Bits()
        self.sparks = Sparks()
        self.vfx = VFX(seed + 7)
        self.dust = Dust(seed + 5)
        # the ram plate and the rod: voxels posed by a translation; they break at the end
        self.ram_vox = IT.make_box('ram', (SC.RAM_W, SC.RAM_W, SC.RAM_H), (-SC.RAM_W / 2, -SC.RAM_W / 2, REST),
                                   (mats['steel'], mats['hazard'], mats['steel']), inner=(70, 72, 78))
        rod_len = SC.ROD_TOP - REST - SC.RAM_H + 16.0
        self.rod_vox = IT.make_box('rod', (SC.ROD_W, SC.ROD_W, rod_len),
                                   (-SC.ROD_W / 2, -SC.ROD_W / 2, REST + SC.RAM_H),
                                   (mats['chrome'], mats['chrome'], mats['chrome']), inner=(150, 152, 158))
        order = PA.crack_order(9)
        i, j, k = self.ram_vox.ijk.T
        sides = np.where((i == 0) | (i == i.max()), order[k % 16, j % 16], order[k % 16, i % 16])
        self.ram_vox.crack = np.where(k == 0, order[j % 16, i % 16], sides)
        self.ram_z = REST
        self.ram_vz = 0.0
        self.ram_shake = np.zeros(3)
        self.ram_broken = False
        self.ram_crack = 0.0
        self.ram_vis = [self.ram_vox.visibility(), self.rod_vox.visibility()]
        self.cur = -1                    # index of the test on the platen
        self.item = None                 # the block under test (intact)
        self.results = {}
        self.events = {}
        self.pressure = 0.0
        self.integrity = 1.0
        self.shake = 0.0
        self.flash = 0.0
        self.lights = []                 # transient point lights (x, y, z, intensity, radius, r, g, b, t_end, t0)

    # ------------------------------------------------------------------------------------------
    def _round(self, x):
        f = int(np.floor(x))
        return f + int(self.rng.random() < x - f)

    def _ev(self, name, n=1, **kw):
        e = self.events.setdefault(name, {'n': 0})
        e['n'] += n
        e.update(kw)

    def clear_platen(self):
        """Sweep what's left of the last block off the platen (done at a cut, so nobody sees it go)."""
        self.frags.remove_on(-9.0, 9.0, -9.0, 7.0, SC.BASE[5] - 1.0, Z0 + 12.0)
        self.bits.remove_on(-9.0, 9.0, -9.0, 7.0, SC.BASE[5] - 1.0, Z0 + 12.0)

    def _place(self, k):
        ts = self.tests[k]
        kind = ts['kind']
        self.clear_platen()
        hollow = 2 if kind == 'chest' else 0
        vox = IT.make_block(kind, self.items_def, (0.0, 0.0), Z0, seed=k + 1, hollow=hollow)
        m = MATS[kind]
        rng = self.rng
        chunks = IT.voronoi_chunks(vox, m.get('chunks', 1), rng) if not m.get('unbreakable') else np.zeros(vox.n, int)
        crumb = rng.random(vox.n) < m.get('crumb', 0.0)
        self.item = dict(kind=kind, vox=vox, mat=m, chunks=chunks, crumb=crumb, s=1.0, bulge=m.get('bulge', 0.0),
                         t0=ts['t0'], white=0.0, swell=1.0, hop=0.0, crack=0.0, vis=vox.visibility(), state='whole',
                         pop=0.0)
        self.cur = k
        self.integrity = 1.0
        self.pressure = 0.0
        self._ev('place', kind=kind, index=k)

    # ------------------------------------------------------------------------------------------
    # the ram and the block, scripted per material
    # ------------------------------------------------------------------------------------------
    def _script(self, dt):
        """Where the ram is, how squashed the block is, and what happens when, for the test on the platen."""
        it = self.item
        if it is None:
            return
        ts = self.tests[self.cur]
        tau = self.t - ts['t0']
        a = ts.get('wait', 0.35)
        kind = it['kind']
        m = it['mat']
        d0, d1 = a, a + ts.get('descend', 0.32)
        z_prev = self.ram_z
        it['pop'] = min(1.0, tau / 0.12)
        if kind == 'slime':
            z, s = self._slime(tau, d0, d1, ts)
        elif kind == 'tnt':
            z, s = self._tnt(tau, d0, d1, ts)
        elif kind == 'bedrock':
            z, s = self._bedrock(tau, d0, d1, ts, dt)
        else:
            z, s = self._standard(tau, d0, d1, ts, m, dt)
        if z is not None:
            self.ram_z = z
        self.ram_vz = (self.ram_z - z_prev) / max(dt, 1e-6)
        if it['state'] == 'whole' and s is not None:
            it['s'] = s

    def _descend(self, tau, d0, d1, z_to=ZC):
        if tau < d0:
            return REST
        u = np.clip((tau - d0) / (d1 - d0), 0, 1)
        return REST - (REST - z_to) * u * u

    def _after(self, tau, tb, retract_at=0.35):
        """Once the block has given: the ram slams down through it, holds, and goes back up."""
        low = Z0 + 0.45
        if tau < tb + 0.12:
            u = (tau - tb) / 0.12
            zb = self._z_at_break
            return zb + (low - zb) * (1 - (1 - u) ** 2)
        if tau < tb + retract_at:
            return low
        u = np.clip((tau - tb - retract_at) / 0.55, 0, 1)
        return low + (REST - low) * float(_ease(u))

    def _standard(self, tau, d0, d1, ts, m, dt):
        it = self.item
        push = ts.get('push', m['push'])
        tb = d1 + push
        if tau < d1:
            return self._descend(tau, d0, d1), 1.0
        if it['state'] == 'whole' and tau < tb:
            u = (tau - d1) / push
            if m.get('crack'):
                # stiff: barely moves, cracks stage by stage, the ram trembles
                s = 1.0 - (1.0 - m['s_break']) * u
                it['crack'] = u
                self.ram_shake = self.rng.normal(0, 0.025 + 0.05 * u, 3)
                self.pressure = m['tons'] * (0.1 + 0.9 * u ** 1.5)
                if m.get('sparks'):
                    self._contact_sparks(self._round(dt * (30.0 + 150.0 * u)))
                if m.get('purple'):
                    self._purple(self._round(dt * 45.0))
                if m.get('sparkle'):
                    self._sparkle(self._round(dt * 25.0), (180, 250, 250))
            else:
                s = 1.0 - (1.0 - m['s_break']) * u ** 1.3
                self.pressure = m['tons'] * u
            self.integrity = 1.0 - u
            return Z0 + H * s, s
        if it['state'] == 'whole':
            self._z_at_break = Z0 + H * m['s_break']
            self._break()
            self.ram_shake = np.zeros(3)
        return self._after(tau, tb), None

    def _slime(self, tau, d0, d1, ts):
        it = self.item
        m = it['mat']
        t_sq = d1 + 0.55                 # squashed flat
        t_bounce = t_sq + 0.35           # ... and it throws the ram back up
        t_slam = t_bounce + 0.75         # second go, full speed
        t_hit = t_slam + 0.12
        if tau < d1:
            return self._descend(tau, d0, d1), 1.0
        if tau < t_sq:
            u = (tau - d1) / (t_sq - d1)
            s = 1.0 - (1.0 - 0.3) * float(_ease(u))
            self.integrity = 1.0 - 0.6 * u
            self.pressure = m['tons'] * 0.6 * u
            return Z0 + H * s, s
        if tau < t_bounce:
            self.ram_shake = self.rng.normal(0, 0.03, 3)
            it['bulge'] = 0.5 + 0.05 * np.sin(tau * 40.0)
            return Z0 + H * 0.3, 0.3
        if tau < t_slam:
            if 'bounced' not in it:
                it['bounced'] = True
                self._ev('bounce')
                self.ram_shake = np.zeros(3)
                self.integrity = 1.0
                self.pressure = 0.0
            v = tau - t_bounce
            # jelly: overshoots and wobbles back to shape, with a little hop
            s = 1.0 + 0.4 * np.exp(-v / 0.22) * np.sin(2 * np.pi * 3.2 * v + np.pi)
            s = 1.0 - 0.7 * np.exp(-v / 0.05) + (s - 1.0) * (1.0 - np.exp(-v / 0.05))
            it['hop'] = max(0.0, 2.2 * np.sin(np.pi * min(v / 0.36, 1.0))) if v < 0.36 else 0.0
            it['bulge'] = 0.5 * np.exp(-v / 0.2)
            u = np.clip(v / 0.18, 0, 1)
            zr = Z0 + H * 0.3 + (REST + 2.0 - Z0 - H * 0.3) * (1 - (1 - u) ** 3)
            zr -= 2.0 * np.clip((v - 0.18) / 0.4, 0, 1)
            return zr, s
        if tau < t_hit:
            u = (tau - t_slam) / (t_hit - t_slam)
            zr = REST - (REST - Z0 - H * 0.28) * u * u
            s = min(1.0, (zr - Z0) / H)
            self.integrity = 1.0 - u
            self.pressure = m['tons'] * u
            it['bulge'] = 0.6 * u
            return zr, s
        if it['state'] == 'whole':
            self._z_at_break = Z0 + H * 0.28
            self._break()
        return self._after(tau, t_hit, 0.4), None

    def _tnt(self, tau, d0, d1, ts):
        it = self.item
        m = it['mat']
        t_sq = d1 + m['push']
        t_boom = t_sq + m['prime']
        if tau < d1:
            return self._descend(tau, d0, d1), 1.0
        if tau < t_sq:
            u = (tau - d1) / (t_sq - d1)
            s = 1.0 - (1.0 - m['s_break']) * u
            self.integrity = 1.0 - 0.5 * u
            self.pressure = m['tons'] * 0.7 * u
            return Z0 + H * s, s
        if tau < t_boom:
            # primed: flashes white and swells like in the game, faster and faster
            v = (tau - t_sq) / m['prime']
            if 'primed' not in it:
                it['primed'] = True
                self._ev('prime')
            period = 0.26 - 0.16 * v
            it['white'] = 0.8 if ((tau - t_sq) / period) % 1.0 < 0.5 else 0.0
            it['swell'] = 1.0 + 0.12 * v ** 2
            self.integrity = 0.5 * (1.0 - v)
            self.pressure = m['tons'] * (0.7 + 0.3 * v)
            return Z0 + H * m['s_break'], m['s_break']
        if it['state'] == 'whole':
            self._z_at_break = Z0 + H * m['s_break']
            it['white'] = 0.0
            self._break(explode=True)
        # the blast throws the ram up, then it comes back to rest
        v = tau - t_boom
        zb = Z0 + H * m['s_break']
        up = REST + 3.0
        if v < 0.15:
            return zb + (up - zb) * (1 - (1 - v / 0.15) ** 2), None
        return up + (REST - up) * float(_ease((v - 0.15) / 0.6)), None

    def _bedrock(self, tau, d0, d1, ts, dt):
        it = self.item
        m = it['mat']
        strain = m['strain']
        t_end = d1 + strain
        if tau < d1:
            return self._descend(tau, d0, d1), 1.0
        if self.ram_broken:
            return None, 1.0
        v = tau - d1
        u = v / strain
        self.pressure = m['tons'] * (0.02 + 0.98 * u ** 2.2)
        self.integrity = 1.0
        self.press_health = 1.0 - u
        self.ram_shake = self.rng.normal(0, 0.03 + 0.14 * u ** 2, 3)
        self.shake = max(self.shake, 0.3 + 1.2 * u ** 2)
        self._contact_sparks(self._round(dt * (40.0 + 420.0 * u ** 2)))
        if u > 0.55:
            self.ram_crack = (u - 0.55) / 0.45
            if 'hoses' not in it:
                it['hoses'] = True
                self._ev('hoses')
            self._oil(self._round(dt * (160.0 + 260.0 * u)))
        if tau >= t_end:
            self._break_press()
            return None, 1.0
        return ZC, 1.0

    # ------------------------------------------------------------------------------------------
    # breaking
    # ------------------------------------------------------------------------------------------
    def _squashed_positions(self, vox, s, bulge):
        p = vox.pos.copy()
        c = np.array([0.0, 0.0, Z0])
        d = p - c
        h = np.clip(d[:, 2] / H, 0, 1)
        bul = 1.0 + bulge * 4.0 * h * (1.0 - h)
        sxy = 1.0 / np.sqrt(max(s, 0.2))
        out = c + np.stack([d[:, 0] * sxy * bul, d[:, 1] * sxy * bul, d[:, 2] * s], -1)
        return out

    def _break(self, explode=False):
        it = self.item
        m = it['mat']
        kind = it['kind']
        vox = it['vox']
        rng = self.rng
        it['state'] = 'broken'
        self.integrity = 0.0
        self.pressure = m['tons']
        self.results[self.cur] = 'crushed'
        wpos = self._squashed_positions(vox, it['s'], it['bulge'])
        group = it['chunks'].copy()
        crumbs = np.nonzero(it['crumb'])[0]
        group[crumbs] = 10000 + np.arange(len(crumbs))
        lo_speed, hi_speed = m['speed']
        up0, up1 = m['up']
        centre = np.array([0.0, 0.0, Z0 + H * it['s'] * 0.5])

        def vel(c):
            n = len(c)
            d = c - centre
            d[:, 2] *= 0.3
            d = _unit(d + rng.normal(0, 0.25, (n, 3)))
            sp = rng.uniform(lo_speed, hi_speed, (n, 1))
            v = d * sp
            v[:, 2] = np.abs(v[:, 2]) * 0.4 + rng.uniform(up0, up1, n)
            return v

        idx = np.arange(vox.n)
        if kind == 'chest':
            lid = vox.ijk[:, 2] >= 11
            group[lid & ~it['crumb']] = 50000
            self.frags.add(vox, idx, group, wpos, lambda c: self._chest_vel(c, vel), 9.0, rng)
            self._loot(m['loot'])
        else:
            self.frags.add(vox, idx, group, wpos, vel, 8.0 if kind != 'obsidian' else 4.0, rng)
        self._ev('break', kind=kind)
        self.shake = max(self.shake, 1.0 if kind in ('grass', 'melon', 'glass', 'chest') else 1.6)
        if m.get('dust'):
            pts = centre + rng.normal(0, 1.5, (14, 3)) * np.array([1.0, 1.0, 0.3])
            self.dust.step(pts, 0.0)
        if m.get('juice'):
            n = m['juice']
            d = _unit(rng.normal(0, 1, (n, 3)) * np.array([1.0, 1.0, 0.5]) + np.array([0.0, 0.0, 0.3]))
            p = centre + d * rng.uniform(0.3, 1.8, (n, 1))
            cols = np.where(rng.random((n, 1)) < 0.12, np.array([28, 20, 16]), np.array([214, 36, 40]))
            self.bits.add(p, d * rng.uniform(6.0, 18.0, (n, 1)), cols, rng.uniform(0.08, 0.2, n),
                          rng.uniform(2.5, 4.0, n),
                          grav=1.0, stick=True)
        if m.get('goo'):
            n = m['goo']
            d = _unit(rng.normal(0, 1, (n, 3)) * np.array([1.0, 1.0, 0.4]) + np.array([0.0, 0.0, 0.2]))
            p = centre + d * rng.uniform(0.3, 1.5, (n, 1))
            self.bits.add(p, d * rng.uniform(8.0, 20.0, (n, 1)), (118, 204, 96), rng.uniform(0.12, 0.3, n),
                          rng.uniform(3.0, 5.0, n), grav=1.0, stick=True)
        if m.get('sparkle'):
            self._sparkle(m['sparkle'], (200, 252, 250) if kind == 'diamond' else (230, 246, 255), burst=True)
        if m.get('pop'):
            self._diamonds(m['pop'])
        if m.get('purple'):
            self._purple(m['purple'], burst=True)
        if explode:
            self.vfx.step(np.array([centre]), 0.0, [(centre, 2.0)])
            self._ev('explode')
            self.shake = max(self.shake, 3.0)
            self.flash = 0.35
            self.lights.append([0.0, -2.0, Z0 + 3.0, 14.0, 40.0, 1.0, 0.6, 0.25, self.t + 0.5, self.t])
        self.item = dict(it, state='broken')

    def _chest_vel(self, c, vel):
        v = vel(c)
        big = np.linalg.norm(c - np.array([0.0, 0.0, Z0 + 3.0]), axis=1) < 0.0
        return v + big[:, None] * 0.0

    def _loot(self, n):
        rng = self.rng
        names = ['diamond', 'diamond', 'diamond', 'diamond', 'gold', 'gold', 'gold', 'iron', 'iron', 'emerald',
                 'emerald', 'apple', 'diamond', 'gold']
        for k in range(n):
            vox = IT.make_sprite_item(names[k % len(names)], 1.6)
            # lift the sprite into the chest's middle
            off = np.array([rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), Z0 + 1.0 + rng.uniform(0, 1.0)])
            wpos = vox.pos + off
            d = _unit(np.array([rng.normal(0, 1), rng.normal(-0.3, 1), 0.0]))
            v0 = d * rng.uniform(3.0, 8.0) + np.array([0.0, 0.0, rng.uniform(9.0, 16.0)])
            self.frags.add(vox, np.arange(vox.n), np.zeros(vox.n, int), wpos, lambda c, v0=v0: np.tile(v0, (len(c), 1)),
                           10.0, rng)
        self._ev('loot', n)

    def _diamonds(self, n):
        rng = self.rng
        for k in range(n):
            vox = IT.make_sprite_item('diamond', 1.4)
            off = np.array([rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8), Z0 + 1.5])
            v0 = np.array([rng.normal(0, 3.0), rng.normal(-1.0, 2.0), rng.uniform(8.0, 13.0)])
            self.frags.add(vox, np.arange(vox.n), np.zeros(vox.n, int), vox.pos + off,
                           lambda c, v0=v0: np.tile(v0, (len(c), 1)), 8.0, rng)

    def _sparkle(self, n, col, burst=False):
        if n <= 0:
            return
        rng = self.rng
        c = np.array([0.0, 0.0, Z0 + H * 0.5])
        d = _unit(rng.normal(0, 1, (n, 3)))
        p = c + d * rng.uniform(1.8, 2.6, (n, 1)) * np.array([1.0, 1.0, 0.9])
        v = d * (rng.uniform(4.0, 12.0, (n, 1)) if burst else rng.uniform(0.5, 2.0, (n, 1)))
        self.bits.add(p, v, col, rng.uniform(0.08, 0.16, n), rng.uniform(0.4, 0.9, n), glow=True, grav=0.15)

    def _purple(self, n, burst=False):
        if n <= 0:
            return
        rng = self.rng
        c = np.array([0.0, 0.0, Z0 + H * 0.5])
        d = _unit(rng.normal(0, 1, (n, 3)))
        p = c + d * rng.uniform(1.5, 2.4, (n, 1))
        v = d * (rng.uniform(3.0, 9.0, (n, 1)) if burst else 0.8) + np.array([0.0, 0.0, 1.2])
        cols = np.where(rng.random((n, 1)) < 0.5, np.array([176, 90, 255]), np.array([120, 50, 200]))
        self.bits.add(p, v, cols, rng.uniform(0.1, 0.2, n), rng.uniform(0.8, 1.6, n), glow=True, grav=-0.05)

    def _contact_sparks(self, n):
        if n <= 0:
            return
        rng = self.rng
        side = rng.integers(0, 4, n)
        t = rng.uniform(-2.0, 2.0, n)
        x = np.where(side == 0, 2.05, np.where(side == 1, -2.05, t))
        y = np.where(side == 2, 2.05, np.where(side == 3, -2.05, t))
        p = np.stack([x, y, np.full(n, self.ram_z - 0.05)], -1)
        out = np.stack([np.sign(x) * (np.abs(x) > 2.0), np.sign(y) * (np.abs(y) > 2.0), np.zeros(n)], -1)
        v = out * rng.uniform(6.0, 16.0, (n, 1)) + rng.normal(0, 3.0, (n, 3)) + np.array([0.0, 0.0, 3.0])
        self.sparks.add(p, v, rng.uniform(0.25, 0.6, n))
        self._ev('sparks', n)

    def _oil(self, n):
        if n <= 0:
            return
        rng = self.rng
        for (hx, hy, hz0, hz1) in SC.HOSES:
            p = np.tile([hx, hy - 0.4, hz1 - 1.0], (n, 1)) + rng.normal(0, 0.1, (n, 3))
            v = np.stack([np.sign(hx) * rng.uniform(1.0, 6.0, n) - np.sign(hx) * 8.0 * rng.random(n),
                          rng.uniform(-9.0, -3.0, n), rng.uniform(-2.0, 5.0, n)], -1)
            self.bits.add(p, v, (46, 30, 14), rng.uniform(0.12, 0.25, n), rng.uniform(2.0, 3.0, n), grav=1.0,
                          stick=True)

    def _break_press(self):
        """The ram plate and the rod give way: they burst into chunks; the bedrock is untouched."""
        rng = self.rng
        self.ram_broken = True
        dz = self.ram_z - REST
        for vox, n_ch, sp in ((self.ram_vox, 34, (9.0, 20.0)), (self.rod_vox, 10, (7.0, 15.0))):
            wpos = vox.pos + np.array([0.0, 0.0, dz]) + self.ram_shake
            idx = np.nonzero(wpos[:, 2] < SC.CYL[4])[0]         # the rod's top end stays in the cylinder
            wpos = wpos[idx]
            group = IT.voronoi_chunks(vox, n_ch, rng, sel=idx)
            crumbs = rng.random(len(idx)) < 0.15
            group[crumbs] = 10000 + np.arange(int(crumbs.sum()))
            c0 = np.array([0.0, 0.0, self.ram_z + 1.0])

            def vel(c, sp=sp, c0=c0):
                n = len(c)
                h = c[:, :2] - c0[:2]
                hn = np.linalg.norm(h, axis=1, keepdims=True)
                rnd = rng.normal(0, 1, (n, 2))
                rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
                dh = np.where(hn > 0.6, h / np.maximum(hn, 1e-6), rnd)
                v = np.zeros((n, 3))
                v[:, :2] = dh * rng.uniform(*sp, (n, 1))
                v[:, 1] -= rng.uniform(0.0, 5.0, n)
                v[:, 2] = rng.uniform(2.0, 11.0, n)
                return v

            self.frags.add(vox, idx, group, wpos, vel, 7.0, rng)
        c = np.array([0.0, 0.0, self.ram_z + 1.0])
        self.vfx.step(np.array([c]), 0.0, [(c, 2.4)])
        self._contact_sparks(120)
        self.sparks.v[-120:] *= 2.2
        self._oil(80)
        self.shake = 4.0
        self.flash = 0.6
        self.lights.append([0.0, -3.0, self.ram_z, 18.0, 50.0, 1.0, 0.7, 0.35, self.t + 0.6, self.t])
        self.results[self.cur] = 'survived'
        self._ev('press_break')

    # ------------------------------------------------------------------------------------------
    def step_frame(self, scale=1.0):
        self.events = {}
        n_sub = max(1, int(np.ceil(SUB * scale - 1e-9)))
        dt = scale / FPS / n_sub
        for _ in range(n_sub):
            k = self.cur + 1
            if k < len(self.tests) and self.t >= self.tests[k]['t0']:
                self._place(k)
            self._script(dt)
            it = self.item
            block = Z0 + H * it['s'] if (it is not None and it['state'] == 'whole' and it['pop'] >= 1.0) else None
            self.frags.step(dt, (self.ram_z, self.ram_vz, not self.ram_broken), self.events, block)
            self.bits.step(dt, block)
            self.sparks.step(dt, block)
            self.t += dt
        self.vfx.step(np.zeros((0, 3)), scale / FPS)
        self.dust.step(np.zeros((0, 3)), scale / FPS)
        self.shake *= 0.85 ** scale
        self.flash *= 0.7 ** scale
        self.lights = [l for l in self.lights if l[8] > self.t]
        self.frame += 1
        return self.events

    # ------------------------------------------------------------------------------------------
    # render data
    # ------------------------------------------------------------------------------------------
    def instances(self):
        """Voxels (the block under test first, then the ram and rod, then fragments and bits), the posed ranges,
        fx for the renderer."""
        parts, giants = [], []
        n0 = 0
        it = self.item
        if it is not None and it['state'] == 'whole':
            vox = it['vox']
            if it['mat'].get('crack'):
                stage = np.floor(it['crack'] * 10.0) / 10.0
                if it.get('_stage') != stage:
                    vox.set_crack(stage + 1e-6)
                    it['_stage'] = stage
            inst = vox.instances(vis=it['vis'])
            parts.append(inst)
            pop = it['pop']
            wid = (0.4 + 0.6 * pop + 0.08 * np.sin(np.pi * pop)) / np.sqrt(max(it['s'], 0.2))
            sq = (it['s'] * (0.5 + 0.5 * pop), wid, it['bulge'], H, np.array([0.0, 0.0, Z0]))
            pose = {'q': np.array([0.0, 0.0, 0.0, 1.0]), 't': np.array([0.0, 0.0, it['hop']]),
                    'pivot': np.array([0.0, 0.0, Z0]), 'axis': np.array([0.0, 0.0, 1.0]), 'stretch': 1.0,
                    'scentre': np.array([0.0, 0.0, Z0 + 2.0]), 'heat': 0.0, 'squash': sq}
            giants.append((n0, n0 + len(inst), 0.0, float(it['white']), float(it['swell']),
                           np.array([0.0, 0.0, Z0 + 2.0]), pose))
            n0 += len(inst)
        if not self.ram_broken:
            dz = self.ram_z - REST
            for vox, vis in zip((self.ram_vox, self.rod_vox), self.ram_vis):
                if self.ram_crack > 0.0 and vox is self.ram_vox:
                    vox.set_crack(self.ram_crack * 0.9, dark=0.2)
                inst = vox.instances(vis=vis)
                parts.append(inst)
                pose = {'q': np.array([0.0, 0.0, 0.0, 1.0]), 't': np.array([0.0, 0.0, dz]) + self.ram_shake,
                        'pivot': np.zeros(3), 'axis': np.array([0.0, 0.0, 1.0]), 'stretch': 1.0,
                        'scentre': np.zeros(3), 'heat': 0.0}
                giants.append((n0, n0 + len(inst), 0.0, 0.0, 1.0, np.zeros(3), pose))
                n0 += len(inst)
        parts.append(self.frags.instances())
        parts.append(self.bits.instances())
        vox = np.concatenate(parts) if parts else np.zeros(0, IT.VOXEL_DTYPE)
        lights = []
        for l in self.lights:                    # transient lights die away instead of switching off
            u = float(np.clip((l[8] - self.t) / max(1e-6, l[8] - l[9]), 0.0, 1.0))
            lights.append(l[:3] + [l[3] * u * u] + l[4:8])
        fx = {'puffs': np.concatenate([self.dust.puffs(), self.vfx.puffs()]), 'flashes': self.vfx.flashes(),
              'streaks': self.sparks.streaks(), 'lights': SC.studio_lights(lights + [list(l) + [1.0, 0.6, 0.25]
                                                                                      for l in self.vfx.lights()[:4]])}
        return vox, giants, fx
