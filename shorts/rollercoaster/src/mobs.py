"""The mobs: the ghast, the ender dragon, endermen and the end crystals.

Every part is a box in block-model units (1/16 of a block), textured from the block atlas like a block model, and
drawn as an instanced kind; a pose function turns time into one instance row per part. Model axes: x right,
y forward (the face), z up.
"""
import numpy as np

import blocks as BL
import entities as EN
import gfx
import voxel as VX

MAT = gfx.MAT_ENTITY


def qa(axis, deg):
    return EN.qaxis(axis, np.radians(deg))


def qm(*qs):
    out = np.array([0.0, 0.0, 0.0, 1.0])
    for q in qs:
        out = EN.qmul(out, q)
    return out


def rot(q, v):
    return EN.qrot(q, v)


def look_quat(fwd, up=(0.0, 0.0, 1.0)):
    f = np.asarray(fwd, float)
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-5:
        r = np.array([1.0, 0.0, 0.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return EN.basis_quat(f, r, u)


def _box(r, at, name, lo, hi, tex, per_face=None, pivot=(0, 0, 0)):
    """A kind made of one box from lo to hi (px), about pivot (px). The box is built from 0 (so each face's uvs
    start at its texture's corner) and moved back by the pivot."""
    lo, hi, pivot = np.asarray(lo, float), np.asarray(hi, float), np.asarray(pivot, float)
    ext = hi - lo
    bx = BL.cube_box(0, 0, 0, *ext, tex, per_face=per_face or {})
    r.add_kind(name, EN.boxes_mesh([bx], at, tuple(pivot - lo)), tex='blocks')


def register(r):
    at = VX.atlas()
    # ghast: a 16 px cube (face on +y) and 2x2 tentacles hanging from their tops
    _box(r, at, 'ghast_body', (-8, -8, -8), (8, 8, 8), 'ghast_side', {'py': 'ghast_face'})
    _box(r, at, 'ghast_body_shoot', (-8, -8, -8), (8, 8, 8), 'ghast_side', {'py': 'ghast_face_shoot'})
    _box(r, at, 'ghast_tentacle', (-1, -1, -12), (1, 1, 0), 'ghast_tentacle')
    # the dragon
    _box(r, at, 'dragon_body', (-12, -28, -10), (12, 28, 10), 'dragon_scale', {'nz': 'dragon_belly'})
    _box(r, at, 'dragon_seg', (-5, -5, -5), (5, 5, 5), 'dragon_scale', {'nz': 'dragon_belly'})
    _box(r, at, 'dragon_spike', (-1, -2, 0), (1, 2, 6), 'dragon_tooth')
    _box(r, at, 'dragon_skull', (-8, 0, -6), (8, 16, 8), 'dragon_scale', {'py': 'dragon_eye', 'nz': 'dragon_belly'})
    _box(r, at, 'dragon_snout', (-6, 16, -2), (6, 32, 4), 'dragon_scale', {'nz': 'dragon_mouth'})
    _box(r, at, 'dragon_horn', (-1, 0, 0), (1, 4, 6), 'dragon_tooth')
    _box(r, at, 'dragon_jaw', (-6, 0, -4), (6, 16, 0), 'dragon_scale', {'pz': 'dragon_mouth'})
    _box(r, at, 'dragon_teeth', (-5, 12, 0), (5, 15, 2), 'dragon_tooth')
    _box(r, at, 'dragon_arm', (0, -4, -4), (56, 4, 4), 'dragon_scale')
    _box(r, at, 'dragon_forearm', (0, -2, -2), (56, 2, 2), 'dragon_scale')
    r.add_kind('dragon_membrane', membrane_mesh(), [wing_image(tip=False), wing_image(tip=True)])
    # endermen
    _box(r, at, 'ender_head', (-4, -4, 0), (4, 4, 8), 'enderman_body', {'py': 'enderman_face'})
    _box(r, at, 'ender_body', (-4, -2, -12), (4, 2, 0), 'enderman_body')
    _box(r, at, 'ender_limb', (-1, -1, -30), (1, 1, 0), 'enderman_body')
    _box(r, at, 'held_grass', (-6, -6, -6), (6, 6, 6), 'grass_side', {'pz': 'grass_top', 'nz': 'dirt'})
    # end crystal: two frames and the core
    _box(r, at, 'crystal_frame', (-8, -8, -8), (8, 8, 8), 'crystal_frame')
    _box(r, at, 'crystal_core', (-4, -4, -4), (4, 4, 4), 'crystal_core')


def membrane_mesh():
    """A quad from the wing's bone (x 0..56 px, y 0) back to its trailing edge (y -56 px), uv over the whole."""
    s = 1.0 / 16.0
    P = [(0, 0, 0), (56 * s, 0, 0), (56 * s, -56 * s, 0), (0, -56 * s, 0)]
    UV = [(0, 0), (1, 0), (1, 1), (0, 1)]
    out = []
    for k in (0, 1, 2, 0, 2, 3):
        out.append((*P[k], 0, 0, 1, *UV[k], -1.0))
    return np.array(out, np.float32)


def wing_image(tip=False, n=64, seed=12):
    """A wing membrane (n x n RGBA): dark leathery skin, finger bones fanning from the root to the trailing edge,
    which is scalloped between them (the arcs are see-through)."""
    rng = np.random.default_rng(seed + tip)
    img = np.zeros((n, n, 4), np.uint8)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)          # x along the bone (u), y towards the trailing edge (v)
    skin = np.array([34, 30, 42], float) * (0.85 + 0.3 * rng.random((n, n)))[..., None]
    img[..., :3] = np.clip(skin, 0, 255)
    img[..., 3] = 255
    tips = np.array([0.18, 0.46, 0.74, 1.0]) * (n - 1) if not tip else np.array([0.3, 0.62, 0.95]) * (n - 1)
    root = np.array([0.0, 0.0]) if not tip else np.array([0.0, 0.0])
    edge = np.full(n, float(n))
    prev = 0.0
    for tx in tips:
        seg = (xx[0] >= prev) & (xx[0] <= tx)
        mid = 0.5 * (prev + tx)
        half = 0.5 * (tx - prev)
        depth = (n - 1) - 10.0 * np.sqrt(np.clip(1 - ((xx[0] - mid) / max(half, 1)) ** 2, 0, 1))
        edge = np.where(seg, depth, edge)
        prev = tx
    img[..., 3] = np.where(yy <= edge[None, :], 255, 0).astype(np.uint8)
    for tx in tips:                                        # the finger bones
        for k in range(n):
            v = k
            u = root[0] + (tx - root[0]) * (k / (n - 1))
            iu = int(round(u))
            for w in (-1, 0, 1):
                if 0 <= iu + w < n:
                    img[v, iu + w, :3] = (70, 64, 80) if w == 0 else (52, 48, 60)
                    img[v, iu + w, 3] = 255
    img[:3, :, :3] = (62, 58, 72)                          # along the arm
    img[:3, :, 3] = 255
    return img


def row(kind, pos, q, scale, emit=0.0, tint=(1, 1, 1), mat=MAT):
    sc = (scale, scale, scale) if np.isscalar(scale) else tuple(scale)
    return (kind, [*pos, *q, *sc, 0, *tint, emit, mat])


# ---------------------------------------------------------------------------------------------
# ghast
# ---------------------------------------------------------------------------------------------
GHAST_S = 4.0            # 16 px body -> 4 blocks


def ghast_rows(pos, face_dir, t, shooting=False):
    """The ghast at pos (its body's centre), turned to face face_dir, tentacles swaying."""
    q = look_quat(np.array([face_dir[0], face_dir[1], 0.0]) + np.array([0, 0, face_dir[2] * 0.4]))
    bob = np.array([0.0, 0.0, 0.35 * np.sin(t * 1.3)])
    c = np.asarray(pos, float) + bob
    out = [row('ghast_body_shoot' if shooting else 'ghast_body', c, q, GHAST_S, emit=0.30)]
    S = GHAST_S / 16.0
    k = 0
    for ix in (-5, 0, 5):
        for iy in (-5, 0, 5):
            base = c + rot(q, np.array([ix, iy, -8.0]) * S)
            ph = t * 2.1 + k * 0.9
            sway = qm(q, qa((1, 0, 0), 14 * np.sin(ph)), qa((0, 1, 0), 10 * np.sin(ph * 0.8 + 1.1)))
            length = 0.8 + 0.3 * ((k * 5) % 4) / 3.0
            out.append(row('ghast_tentacle', base, sway, (GHAST_S, GHAST_S, GHAST_S * length), emit=0.22))
            k += 1
    return out


# ---------------------------------------------------------------------------------------------
# the ender dragon
# ---------------------------------------------------------------------------------------------
DRAGON_S = 2.6           # px -> blocks * 16 (so 1 px = 0.1625 blocks)


class Dragon:
    """Pose from a flight path: path(t) -> position of the body's centre; the body faces along the velocity
    (or `face` when perched), banks into turns; wings flap, the neck and tail follow."""

    def __init__(self, path, perch=None):
        self.path = path              # function t -> (pos, flap_amount, roar)
        self.perch = perch            # None or (t0, heading vector) from when it sits

    def frame(self, t, dt=0.05):
        p0 = np.asarray(self.path(t - dt)[0])
        p1 = np.asarray(self.path(t + dt)[0])
        v = (p1 - p0) / (2 * dt)
        if self.perch is not None and t >= self.perch[0]:
            f = np.asarray(self.perch[1], float)
        else:
            f = v if np.linalg.norm(v) > 1e-3 else np.array([0.0, 1.0, 0.0])
        f = f / np.linalg.norm(f)
        # bank from the turn rate
        p2 = np.asarray(self.path(t + 3 * dt)[0])
        a = (p2 - 2 * np.asarray(self.path(t + dt)[0]) + np.asarray(self.path(t - dt)[0])) / (4 * dt * dt)
        side = np.cross(f, [0, 0, 1.0])
        bank = float(np.clip(np.dot(a, side) * 0.9, -40, 40)) if (self.perch is None or t < self.perch[0]) else 0.0
        pitch = np.degrees(np.arcsin(np.clip(f[2], -1, 1)))
        fh = np.array([f[0], f[1], 0.0])
        fh /= max(np.linalg.norm(fh), 1e-6)
        q = qm(look_quat(fh), qa((1, 0, 0), float(np.clip(pitch, -30, 30))), qa((0, 1, 0), -bank))
        return q, v

    def rows(self, t):
        pos, flap, roar = self.path(t)
        pos = np.asarray(pos, float)
        q, v = self.frame(t)
        S = DRAGON_S / 16.0
        out = []
        emit_eye = 0.0

        def at(local):
            return pos + rot(q, np.asarray(local, float) * S)
        out.append(row('dragon_body', pos, q, DRAGON_S))
        for yy in (-18, -4, 10, 22):
            out.append(row('dragon_spike', at((0, yy, 10)), q, DRAGON_S))
        # neck: five segments arching up from the chest; the head at its end
        perched = self.perch is not None and t >= self.perch[0]
        arch = 16.0 + (14.0 if perched else 0.0) + 10.0 * roar
        p = np.array([0.0, 28.0, 4.0])
        qn = q
        for k in range(5):
            bend = arch * (0.5 - k / 5.0) + 4.0 * np.sin(t * 1.4 + k * 0.6)
            qn = qm(qn, qa((1, 0, 0), bend * 0.35))
            p_w = at(p) if k == 0 else p_w + rot(qn, np.array([0, 10.0, 0]) * S)
            out.append(row('dragon_seg', p_w + rot(qn, np.array([0, 5.0, 0]) * S), qn, DRAGON_S))
        neck_end = p_w + rot(qn, np.array([0, 10.0, 0]) * S)
        # the head: looks a little down when perched (at whoever is coming), the jaw drops when it roars
        qh = qm(qn, qa((1, 0, 0), -12.0 - (18.0 if perched else 0.0) + 16.0 * roar))
        out.append(row('dragon_skull', neck_end, qh, DRAGON_S, emit=emit_eye))
        out.append(row('dragon_snout', neck_end, qh, DRAGON_S))
        for sx in (-4, 4):
            out.append(row('dragon_horn', neck_end + rot(qh, np.array([sx, 2.0, 8.0]) * S),
                           qm(qh, qa((1, 0, 0), -35)), DRAGON_S))
        qj = qm(qh, qa((1, 0, 0), -38.0 * roar))
        jaw0 = neck_end + rot(qh, np.array([0, 16.0, -6.0]) * S)
        out.append(row('dragon_jaw', jaw0, qj, DRAGON_S))
        out.append(row('dragon_teeth', jaw0, qj, DRAGON_S))
        out.append(row('dragon_teeth', neck_end + rot(qh, np.array([0, 17.0, -4.0]) * S),
                       qm(qh, qa((1, 0, 0), 180)), (DRAGON_S, DRAGON_S, DRAGON_S)))
        # tail: twelve segments trailing along where the body has been
        prev = at((0, -28.0, 0))
        for k in range(12):
            tt = t - 0.045 * (k + 1)
            pk = np.asarray(self.path(tt)[0])
            qk, _ = self.frame(tt)
            tail_dir = rot(qk, np.array([0.0, -1.0, 0.0]))
            sway = 0.25 * np.sin(t * 2.2 - k * 0.5)
            side = rot(qk, np.array([1.0, 0.0, 0.0]))
            d = tail_dir + side * sway
            d /= np.linalg.norm(d)
            cur = prev + d * 10.0 * S
            qt = look_quat(-d)
            sc = DRAGON_S * (1.0 - 0.035 * k)
            out.append(row('dragon_seg', (prev + cur) / 2, qt, sc))
            if k % 3 == 1:
                out.append(row('dragon_spike', (prev + cur) / 2 + rot(qt, np.array([0, 0, 5.0]) * sc / 16.0),
                               qt, sc))
            prev = cur
        # wings: an arm and a forearm on each side, each with its membrane; flapping, the tips lagging
        ph = t * 2 * np.pi * (0.62 if not perched else 0.35)
        up = (34.0 if not perched else 12.0) * flap * np.sin(ph) + (6.0 if not perched else 46.0)
        tip = (26.0 if not perched else 10.0) * flap * np.sin(ph - 0.9) + (-4.0 if not perched else -60.0)
        for side in (1, -1):
            sh = at((12.0 * side, 12.0, 6.0))
            qarm = qm(q, qa((0, 1, 0), -up * side))
            scl = (DRAGON_S * side, DRAGON_S, DRAGON_S)
            out.append(row('dragon_arm', sh, qarm, scl))
            out.append(('dragon_membrane', [*sh, *qarm, *scl, 0, 1, 1, 1, 0.0, MAT]))
            elbow = sh + rot(qarm, np.array([56.0 * side, 0, 0]) * S)
            qfa = qm(qarm, qa((0, 1, 0), -tip * side), qa((0, 0, 1), 8.0 * side))
            out.append(row('dragon_forearm', elbow, qfa, scl))
            out.append(('dragon_membrane', [*elbow, *qfa, *scl, 1, 1, 1, 1, 0.0, MAT]))
        return out, dict(head=neck_end + rot(qh, np.array([0, 24.0, 0]) * S), body=pos)


# ---------------------------------------------------------------------------------------------
# endermen and crystals
# ---------------------------------------------------------------------------------------------
def enderman_rows(pos, yaw_deg, look_at=None, holding=False, t=0.0):
    """An enderman standing at pos (feet), facing yaw (0 = +y); its head turns to look_at."""
    S = 1.0
    qb = qa((0, 0, 1), -yaw_deg)
    p = np.asarray(pos, float)
    out = []
    hip = p + np.array([0, 0, 30.0 / 16])
    for sx in (-2, 2):
        out.append(row('ender_limb', hip + rot(qb, np.array([sx / 16.0, 0, 0])), qb, S))
    out.append(row('ender_body', hip + np.array([0, 0, 12.0 / 16]), qb, S))
    neck = hip + np.array([0, 0, 12.0 / 16])
    if look_at is not None:
        d = np.asarray(look_at, float) - neck
        qh = look_quat(d / np.linalg.norm(d))
    else:
        qh = qb
    out.append(row('ender_head', neck + np.array([0, 0, 0.02]), qh, S))
    arm_pitch = -30.0 if holding else 4.0 * np.sin(t * 0.9)
    for sx in (-5, 5):
        qa_ = qm(qb, qa((1, 0, 0), arm_pitch))
        out.append(row('ender_limb', neck + rot(qb, np.array([sx / 16.0, 0, -0.1])), qa_, S))
    if holding:
        hand = neck + rot(qb, np.array([0, 1.3, -0.9]))
        out.append(row('held_grass', hand, qb, 0.9))
    return out


def crystal_rows(pos, t, k):
    """An end crystal floating over its pillar: two frames turning on tilted axes round a glowing core."""
    c = np.asarray(pos, float) + np.array([0, 0, 1.2 + 0.35 * np.sin(t * 2.0 + k)])
    out = []
    a = t * 110.0 + k * 40
    q1 = qm(qa((0, 0, 1), a), qa((1, 1, 0), 55))
    q2 = qm(qa((0, 0, 1), -a * 1.3), qa((1, -1, 0), 55))
    out.append(row('crystal_frame', c, q1, 2.0, emit=0.5))
    out.append(row('crystal_frame', c, q2, 1.5, emit=0.5))
    out.append(row('crystal_core', c, qm(qa((0, 0, 1), a * 2.1), qa((1, 0, 1), 40)), 1.3, emit=0.9))
    return out, c
