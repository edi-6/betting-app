"""Characters and moving props.

Box models in the game's style, built from the texture layouts in skins.py: the player (the protagonist and
NOAH_404 share the model; the skin picks who it is), villagers, cows, sheep, pigs and chickens. Each part is its own
instanced mesh (a box in pixel units around the part's pivot); an Actor turns a pose (position, body yaw, head yaw
and pitch, walk phase, arm poses) into one instance per part.

Also: the first-person arm and held items (items are the 16x16 icons extruded one pixel thick), and props built
from block models that move (doors, chest lids, the trapdoor, the ceiling block that breaks, the falling sign).
"""
import numpy as np

import blocks as BL
import gfx
import skins as SK
import textures as TX
import voxel as VX

PX = 1.0 / 16.0


# ---------------------------------------------------------------------------------------------
# quaternions (x, y, z, w)
# ---------------------------------------------------------------------------------------------
def qaxis(axis, ang):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    s = np.sin(ang / 2)
    return np.array([a[0] * s, a[1] * s, a[2] * s, np.cos(ang / 2)])


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def qrot(q, v):
    q = np.asarray(q, float)
    v = np.asarray(v, float)
    u = q[:3]
    return v + 2.0 * np.cross(u, np.cross(u, v) + q[3] * v)


QI = np.array([0.0, 0.0, 0.0, 1.0])


def qx(a):
    return qaxis((1, 0, 0), a)


def qy(a):
    return qaxis((0, 1, 0), a)


def qz(a):
    return qaxis((0, 0, 1), a)


# ---------------------------------------------------------------------------------------------
# box meshes with the game's box UVs
# ---------------------------------------------------------------------------------------------
def box_mesh(tex_box, lo, hi, inflate=0.0, tex_size=64):
    """Mesh (36, 9) pos3 nrm3 uv2 layer1 of a box from lo to hi (pixels, relative to the part's pivot), textured from
    the layout box (u, v, w, h, d) (w along x, h along z, d along y). The front is +y."""
    u, v, w, h, d = tex_box
    x0, y0, z0 = np.asarray(lo, float) - inflate
    x1, y1, z1 = np.asarray(hi, float) + inflate
    R = SK.region(u, v, w, h, d)
    T = float(tex_size)

    def rect(name):
        ru, rv, rw, rh = R[name]
        return ru / T, rv / T, (ru + rw) / T, (rv + rh) / T

    out = []
    faces = {'py': 'front', 'px': 'right', 'nx': 'left', 'ny': 'back', 'pz': 'top', 'nz': 'bottom'}
    for face, reg in faces.items():
        P, _ = VX.face_corners(face, np.float32(x0), np.float32(y0), np.float32(z0), np.float32(x1), np.float32(y1),
                               np.float32(z1))
        P = P.reshape(4, 3)
        u0, v0, u1, v1 = rect(reg)
        if face == 'pz':
            UV = [(u1, v0), (u0, v0), (u0, v1), (u1, v1)]
        elif face == 'nz':
            UV = [(u0, v1), (u1, v1), (u1, v0), (u0, v0)]
        else:
            UV = [(u0, v1), (u1, v1), (u1, v0), (u0, v0)]
        n = VX.FACE_DIRS[face]
        for k in (0, 1, 2, 0, 2, 3):
            out.append((*P[k], *n, *UV[k], -1.0))
    return np.array(out, np.float32)


# part: (layout key, lo, hi, pivot, parent, inflate, rest rotation (axis, degrees) or None)
PLAYER = {
    'body': ('body', (-4, -2, -12), (4, 2, 0), (0, 0, 24), None, 0.0),
    'jacket': ('jacket', (-4, -2, -12), (4, 2, 0), (0, 0, 0), 'body', 0.25),
    'head': ('head', (-4, -4, 0), (4, 4, 8), (0, 0, 24), None, 0.0),
    'hat': ('hat', (-4, -4, 0), (4, 4, 8), (0, 0, 0), 'head', 0.5),
    'arm_r': ('arm_r', (-2, -2, -10), (2, 2, 2), (6, 0, 22), None, 0.0),
    'sleeve_r': ('sleeve_r', (-2, -2, -10), (2, 2, 2), (0, 0, 0), 'arm_r', 0.25),
    'arm_l': ('arm_l', (-2, -2, -10), (2, 2, 2), (-6, 0, 22), None, 0.0),
    'sleeve_l': ('sleeve_l', (-2, -2, -10), (2, 2, 2), (0, 0, 0), 'arm_l', 0.25),
    'leg_r': ('leg_r', (-2, -2, -12), (2, 2, 0), (2, 0, 12), None, 0.0),
    'pants_r': ('pants_r', (-2, -2, -12), (2, 2, 0), (0, 0, 0), 'leg_r', 0.25),
    'leg_l': ('leg_l', (-2, -2, -12), (2, 2, 0), (-2, 0, 12), None, 0.0),
    'pants_l': ('pants_l', (-2, -2, -12), (2, 2, 0), (0, 0, 0), 'leg_l', 0.25),
}
VILLAGER = {
    'body': ('body', (-4, -3, -12), (4, 3, 0), (0, 0, 24), None, 0.0),
    'robe': ('robe', (-4, -3, -18), (4, 3, 0), (0, 0, 0), 'body', 0.5),
    'head': ('head', (-4, -4, 0), (4, 4, 10), (0, 0, 24), None, 0.0),
    'nose': ('nose', (-1, 4, 1), (1, 6, 5), (0, 0, 0), 'head', 0.0),
    'arms': ('arms', (-4, -2, -8), (4, 2, -4), (0, 0.5, 21.5), None, 0.0),
    'arm_side_r': ('arm_side', (4, -2, -8), (8, 2, 0), (0, 0, 0), 'arms', 0.0),
    'arm_side_l': ('arm_side', (-8, -2, -8), (-4, 2, 0), (0, 0, 0), 'arms', 0.0),
    'leg_r': ('leg', (-2, -2, -12), (2, 2, 0), (2, 0, 12), None, 0.0),
    'leg_l': ('leg', (-2, -2, -12), (2, 2, 0), (-2, 0, 12), None, 0.0),
}
# quadrupeds: the body box is modelled upright (w, h, d) and turned to lie along +y by its rest rotation
COW = {
    'body': ('body', (-6, -5, -9), (6, 5, 9), (0, 0, 17), None, 0.0, ('x', -90)),
    'head': ('head', (-4, 0, -4), (4, 6, 4), (0, 9, 20), None, 0.0),
    'horn_r': ('horn', (4, 1, 3), (5, 2, 6), (0, 0, 0), 'head', 0.0),
    'horn_l': ('horn', (-5, 1, 3), (-4, 2, 6), (0, 0, 0), 'head', 0.0),
    'leg_fr': ('leg', (-2, -2, -12), (2, 2, 0), (4, 6, 12), None, 0.0),
    'leg_fl': ('leg', (-2, -2, -12), (2, 2, 0), (-4, 6, 12), None, 0.0),
    'leg_br': ('leg', (-2, -2, -12), (2, 2, 0), (4, -6, 12), None, 0.0),
    'leg_bl': ('leg', (-2, -2, -12), (2, 2, 0), (-4, -6, 12), None, 0.0),
}
SHEEP = {
    'body': ('body', (-4, -3, -8), (4, 3, 8), (0, 0, 16), None, 0.0, ('x', -90)),
    'head': ('head', (-3, 0, -3), (3, 8, 3), (0, 6, 18), None, 0.0),
    'cap': ('cap', (-3, 1, -2), (3, 7, 4), (0, 0, 0), 'head', 0.6),
    'leg_fr': ('leg', (-2, -2, -12), (2, 2, 0), (3, 5, 12), None, 0.0),
    'leg_fl': ('leg', (-2, -2, -12), (2, 2, 0), (-3, 5, 12), None, 0.0),
    'leg_br': ('leg', (-2, -2, -12), (2, 2, 0), (3, -6, 12), None, 0.0),
    'leg_bl': ('leg', (-2, -2, -12), (2, 2, 0), (-3, -6, 12), None, 0.0),
}
PIG = {
    'body': ('body', (-5, -4, -8), (5, 4, 8), (0, 0, 10), None, 0.0, ('x', -90)),
    'head': ('head', (-4, 0, -4), (4, 8, 4), (0, 6, 11), None, 0.0),
    'snout': ('snout', (-2, 8, -3), (2, 9, 0), (0, 0, 0), 'head', 0.0),
    'leg_fr': ('leg', (-2, -2, -6), (2, 2, 0), (3, 5, 6), None, 0.0),
    'leg_fl': ('leg', (-2, -2, -6), (2, 2, 0), (-3, 5, 6), None, 0.0),
    'leg_br': ('leg', (-2, -2, -6), (2, 2, 0), (3, -5, 6), None, 0.0),
    'leg_bl': ('leg', (-2, -2, -6), (2, 2, 0), (-3, -5, 6), None, 0.0),
}
CHICKEN = {
    'body': ('body', (-3, -3, -4), (3, 3, 4), (0, 0, 8), None, 0.0, ('x', -90)),
    'head': ('head', (-2, -1.5, 0), (2, 1.5, 6), (0, 4, 8), None, 0.0),
    'beak': ('beak', (-2, 1.5, 2), (2, 3.5, 4), (0, 0, 0), 'head', 0.0),
    'wattle': ('wattle', (-1, 1.5, 0), (1, 3.5, 2), (0, 0, 0), 'head', 0.0),
    'wing_r': ('wing', (0, -3, -4), (1, 3, 0), (3, 0, 11), None, 0.0),
    'wing_l': ('wing', (-1, -3, -4), (0, 3, 0), (-3, 0, 11), None, 0.0),
    'leg_r': ('leg', (-1.5, -1.5, -5), (1.5, 1.5, 0), (1.5, 0, 5), None, 0.0),
    'leg_l': ('leg', (-1.5, -1.5, -5), (1.5, 1.5, 0), (-1.5, 0, 5), None, 0.0),
}
MODELS = {'player': PLAYER, 'villager': VILLAGER, 'cow': COW, 'sheep': SHEEP, 'pig': PIG, 'chicken': CHICKEN}
MODEL_LAYOUT = {'player': 'player', 'villager': 'villager', 'cow': 'cow', 'sheep': 'sheep', 'pig': 'pig',
                'chicken': 'chicken'}
SCALE = {'player': 0.9375, 'villager': 0.9375, 'cow': 1.0, 'sheep': 1.0, 'pig': 1.0, 'chicken': 1.0}
SKIN_LAYER = {n: i for i, n in enumerate(SK.SKIN_NAMES)}


def rest_quat(spec):
    if len(spec) > 6 and spec[6]:
        axis, deg = spec[6]
        return qaxis({'x': (1, 0, 0), 'y': (0, 1, 0), 'z': (0, 0, 1)}[axis], np.radians(deg))
    return QI


# ---------------------------------------------------------------------------------------------
# items: icons extruded one texel thick
# ---------------------------------------------------------------------------------------------
def item_mesh(icon):
    """(N, 9) mesh of an icon extruded along y (1/16 thick), 1 unit = 16 texels, centred on the origin, the icon
    standing in the x-z plane (row 0 at the top), facing -y."""
    a = icon[..., 3] > 127
    n = icon.shape[0]
    out = []
    t = 0.5
    for r in range(n):
        for c in range(n):
            if not a[r, c]:
                continue
            x0, x1 = c - n / 2, c + 1 - n / 2
            z0, z1 = (n - 1 - r) - n / 2, (n - r) - n / 2
            uv0 = ((c + 0.2) / n, (r + 0.2) / n)
            uv1 = ((c + 0.8) / n, (r + 0.8) / n)
            for face in ('ny', 'py', 'px', 'nx', 'pz', 'nz'):
                if face == 'px' and c + 1 < n and a[r, c + 1]:
                    continue
                if face == 'nx' and c > 0 and a[r, c - 1]:
                    continue
                if face == 'pz' and r > 0 and a[r - 1, c]:
                    continue
                if face == 'nz' and r + 1 < n and a[r + 1, c]:
                    continue
                P, _ = VX.face_corners(face, np.float32(x0), np.float32(-t), np.float32(z0), np.float32(x1),
                                       np.float32(t), np.float32(z1))
                P = P.reshape(4, 3)
                UV = [(uv0[0], uv1[1]), (uv1[0], uv1[1]), (uv1[0], uv0[1]), (uv0[0], uv0[1])]
                nrm = VX.FACE_DIRS[face]
                for k in (0, 1, 2, 0, 2, 3):
                    out.append((*(P[k] / n), *nrm, *UV[k], -1.0))
    return np.array(out, np.float32)


# ---------------------------------------------------------------------------------------------
# block-model props (doors, chest lid, trapdoor, sign, a plain cube)
# ---------------------------------------------------------------------------------------------
def boxes_mesh(boxes, at, pivot=(8, 8, 0)):
    """Mesh (N, 9) of block-model boxes (1/16 units) in block units around `pivot` (1/16 units)."""
    out = []
    for bx in boxes:
        for face, (tex, uv) in bx['faces'].items():
            q = VX.model_face(bx, face, uv, 0)
            P, UV, n = q
            P = P - np.array(pivot, np.float32) / 16.0
            lay = at[tex]
            for k in (0, 1, 2, 0, 2, 3):
                out.append((*P[k], *n, *UV[k], float(lay)))
    return np.array(out, np.float32)


def register(r, at=None):
    """Upload every character part, item and prop mesh as instanced kinds."""
    at = at or VX.atlas()
    S = SK.make_skins()
    layers = [S[n] for n in SK.SKIN_NAMES]
    first = True
    for model, parts in MODELS.items():
        lay = SK.LAYOUT[MODEL_LAYOUT[model]]
        for pname, spec in parts.items():
            key, lo, hi, pivot, parent, infl = spec[:6]
            mesh = box_mesh(lay[key], lo, hi, infl)
            name = f'{model}_{pname}'
            if first:
                r.add_kind(name, mesh, layers)
                first = False
                r._skin_kind = name
            else:
                r.add_kind(name, mesh, tex=r._skin_kind)
    icons = TX.item_icons()
    names = sorted(icons)
    r.add_kind('item_icons_tex', item_mesh(icons[names[0]]), [icons[k] for k in names])
    for k in names:
        r.add_kind('item_' + k, item_mesh(icons[k]), tex='item_icons_tex')
    r._item_names = names
    # block props
    cube = BL.box(0, 0, 0, 16, 16, 16, BL.auto_faces('spruce_planks', 0, 0, 0, 16, 16, 16))
    r.add_kind('prop_cube', boxes_mesh([cube], at, (8, 8, 8)) * np.array([1, 1, 1, 1, 1, 1, 1, 1, 0], np.float32) +
               np.array([0, 0, 0, 0, 0, 0, 0, 0, -1], np.float32), tex='blocks')
    for wood in ('oak', 'spruce'):
        for part in ('bottom', 'top'):
            bx = BL.m_door(wood)(BL.HALF_TOP if part == 'top' else 0, {})
            r.add_kind(f'prop_{wood}_door_{part}', boxes_mesh(bx, at, (0, 1.5, 0)), tex='blocks')
    ch = BL.m_chest(0, {})
    body = BL.cube_box(1, 1, 0, 15, 15, 10, 'chest_side', per_face={'ny': 'chest_front', 'pz': 'chest_top',
                                                                    'nz': 'chest_top'})
    lid = BL.cube_box(1, 1, 10, 15, 15, 14, 'chest_side', per_face={'ny': 'chest_front', 'pz': 'chest_top',
                                                                    'nz': 'chest_top'})
    latch = BL.cube_box(7, 0, 8, 9, 1, 12, 'chest_front')
    r.add_kind('prop_chest_body', boxes_mesh([body], at, (8, 8, 0)), tex='blocks')
    r.add_kind('prop_chest_lid', boxes_mesh([lid, latch], at, (8, 15, 10)), tex='blocks')     # hinge at the back
    del ch
    r.add_kind('prop_trapdoor', boxes_mesh(BL.m_trapdoor(0, {}), at, (8, 16, 1.5)), tex='blocks')
    r.add_kind('prop_sign', boxes_mesh(BL.m_sign(0, {}), at, (8, 8, 0)), tex='blocks')
    r.add_kind('prop_lectern_book', boxes_mesh([BL.cube_box(3, 3, 0, 13, 13, 1, 'white_wool',
                                                            per_face={'pz': 'white_wool'}),
                                                BL.cube_box(2, 2, -0.5, 14, 14, 0, 'brown_wool')], at, (8, 8, 0)),
               tex='blocks')
    # the monitor on the desk: a black case and a screen quad with its own (per frame) texture
    case = BL.cube_box(0, 6, 0, 16, 9, 11, 'black_concrete')
    stand = BL.cube_box(6, 8, -2, 10, 10, 0, 'black_concrete')
    r.add_kind('prop_monitor', boxes_mesh([case, stand], at, (8, 8, 0)), tex='blocks')
    return S


# ---------------------------------------------------------------------------------------------
# actors
# ---------------------------------------------------------------------------------------------
class Actor:
    """A posed character. Angles in radians; yaw 0 = facing +y, positive = counter-clockwise (towards -x)."""

    def __init__(self, model, skin, pos=(0, 0, 0), yaw=0.0):
        self.model = model
        self.skin = skin
        self.pos = np.array(pos, float)
        self.yaw = yaw
        self.head_yaw = 0.0          # relative to the body
        self.head_pitch = 0.0        # positive = looking down
        self.walk = 0.0              # walk cycle phase (radians)
        self.walk_amp = 0.0          # 0 standing .. 1 walking
        self.arm_r = None            # (pitch, roll) override for the right arm (radians), e.g. reaching
        self.arm_l = None
        self.sit = False
        self.tint = (1.0, 1.0, 1.0)
        self.emit = 0.0
        self.hidden_parts = ()
        self.skin_override = {}      # part -> skin (the figure's skin changes as it turns)
        self.idle_t = 0.0            # breathing / idle sway time
        self.mat = gfx.MAT_ENTITY

    def part_pose(self, pname):
        """Local rotation of a part (in the body frame)."""
        m = self.model
        s = np.sin(self.walk)
        amp = self.walk_amp
        idle = 0.03 * np.sin(self.idle_t * 1.3)
        if m == 'player':
            if pname == 'head':
                return qmul(qz(self.head_yaw), qx(-self.head_pitch))
            if pname == 'arm_r':
                if self.arm_r is not None:
                    return qmul(qx(self.arm_r[0]), qy(self.arm_r[1]))
                if self.sit:
                    return qx(0.9)
                return qmul(qx(-s * 0.8 * amp), qy(idle))
            if pname == 'arm_l':
                if self.arm_l is not None:
                    return qmul(qx(self.arm_l[0]), qy(self.arm_l[1]))
                if self.sit:
                    return qx(0.9)
                return qmul(qx(s * 0.8 * amp), qy(-idle))
            if pname == 'leg_r':
                return qx(1.5708) if self.sit else qx(s * 0.9 * amp)
            if pname == 'leg_l':
                return qx(1.5708) if self.sit else qx(-s * 0.9 * amp)
            return QI
        if m == 'villager':
            if pname == 'head':
                return qmul(qz(self.head_yaw), qx(-self.head_pitch))
            if pname == 'arms':
                return qx(0.75)
            if pname == 'leg_r':
                return qx(s * 0.7 * amp)
            if pname == 'leg_l':
                return qx(-s * 0.7 * amp)
            return QI
        # quadrupeds and the chicken
        if pname == 'head':
            return qmul(qz(self.head_yaw), qx(-self.head_pitch))
        if pname in ('leg_fr', 'leg_bl', 'leg_r'):
            return qx(s * 0.7 * amp)
        if pname in ('leg_fl', 'leg_br', 'leg_l'):
            return qx(-s * 0.7 * amp)
        if pname in ('wing_r', 'wing_l'):
            return qy((1 if pname == 'wing_r' else -1) * 0.15 * amp * abs(s))
        return QI

    def instances(self):
        parts = MODELS[self.model]
        sc = SCALE[self.model] * PX
        root_q = qz(self.yaw)
        bob = 0.0
        if self.model == 'player' and self.walk_amp > 0:
            bob = 0.035 * abs(np.cos(self.walk)) * self.walk_amp
        root_p = self.pos + np.array([0.0, 0.0, bob])
        if self.sit:
            root_p = root_p + np.array([0.0, 0.0, -0.62])
        out = []
        world = {}
        for pname, spec in parts.items():
            key, lo, hi, pivot, parent, infl = spec[:6]
            rq = rest_quat(spec)
            if parent is None:
                lq = self.part_pose(pname)
                q = qmul(root_q, lq)
                p = root_p + qrot(root_q, np.array(pivot, float) * sc)
            else:
                pp, pq = world[parent]
                q = pq
                p = pp + qrot(pq, np.array(pivot, float) * sc)
            world[pname] = (p, q)
            if pname in self.hidden_parts or (parent in self.hidden_parts if parent else False):
                continue
            skin = self.skin_override.get(pname, self.skin_override.get(parent, self.skin))
            qq = qmul(q, rq)
            out.append((f'{self.model}_{pname}', [*p, *qq, sc, sc, sc, SKIN_LAYER[skin], *self.tint, self.emit,
                                                  self.mat]))
        return out

    def head_top(self):
        """World position just above the head (for name tags)."""
        h = {'player': 2.05, 'villager': 2.2, 'cow': 1.6, 'sheep': 1.4, 'pig': 1.1, 'chicken': 0.9}[self.model]
        return self.pos + np.array([0.0, 0.0, h])

    def eye(self):
        return self.pos + np.array([0.0, 0.0, 1.62])


def collect(actor_list, extra=None):
    """dict kind -> (M, 16) float32 for the renderer."""
    d = {}
    for a in actor_list:
        for kind, row in a.instances():
            d.setdefault(kind, []).append(row)
    for kind, row in (extra or []):
        d.setdefault(kind, []).append(row)
    return {k: np.array(v, np.float32) for k, v in d.items()}


def inst(kind, pos, quat=QI, scale=1.0, layer=0, tint=(1, 1, 1), emit=0.0, mat=gfx.MAT_ENTITY):
    sc = (scale, scale, scale) if np.isscalar(scale) else scale
    return (kind, [*pos, *quat, *sc, layer, *tint, emit, mat])


# ---------------------------------------------------------------------------------------------
# first person
# ---------------------------------------------------------------------------------------------
def cam_basis(eye, target, up=(0, 0, 1)):
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return f, r, u


def basis_quat(f, r, u):
    """Rotation taking the model axes (x right, y forward, z up) to (r, f, u)."""
    R = np.stack([r, f, u], 1)
    return gfx.quat_from_mat(R)


def arm_quat(d, up_hint):
    """Rotation for an arm part whose long axis (the model's -z, shoulder -> hand) points along d, with the arm's
    front (+y) turned towards up_hint."""
    d = np.asarray(d, float)
    d = d / np.linalg.norm(d)
    uz = -d
    fy = np.asarray(up_hint, float) - np.dot(up_hint, uz) * uz
    fy /= np.linalg.norm(fy)
    rx = np.cross(fy, uz)
    return basis_quat(fy, rx, uz)


def first_person(eye, target, skin='you', item=None, swing=0.0, bob=(0.0, 0.0), reach=0.0, hold_map=False,
                 item_names=None, up=(0, 0, 1), map_layer=0):
    """Instances for the player's own right arm (and held item, or the map in both hands) in front of the camera.
    swing: 0..1 (the punch/use swing); bob: (side, down) offsets from walking; reach: 0..1 arm stretched out."""
    f, r, u = cam_basis(eye, target, up)
    out = []
    sc = PX * 0.9375
    s = np.sin(np.pi * np.clip(swing, 0, 1))
    if hold_map:
        # the map held low in both hands, tilted back, the arms coming in from the bottom corners
        mc = eye + f * 0.42 - u * (0.09 + bob[1]) + r * bob[0]
        tilt = 0.35
        mu = u * np.cos(tilt) + f * np.sin(tilt)            # the map's up, leaning away at the top
        mf = f * np.cos(tilt) - u * np.sin(tilt)            # its normal points back at the eye (quad faces -y)
        q = basis_quat(mf, r, mu)
        out.append(('held_map', [*mc, *q, 0.5, 1.0, 0.5, map_layer, 1, 1, 1, 0, gfx.MAT_HAND]))
        for side in (1, -1):
            hand = mc + r * (0.23 * side) - mu * 0.22 - mf * 0.02
            shoulder = eye + r * (0.34 * side) - u * 0.62 + f * 0.05
            q = arm_quat(hand - shoulder, u)
            out.append(('player_arm_r' if side > 0 else 'player_arm_l',
                        [*shoulder, *q, sc, sc, sc, SKIN_LAYER[skin], 1, 1, 1, 0, gfx.MAT_HAND]))
            out.append(('player_sleeve_r' if side > 0 else 'player_sleeve_l',
                        [*shoulder, *q, sc, sc, sc, SKIN_LAYER[skin], 1, 1, 1, 0, gfx.MAT_HAND]))
        return out
    # the right arm: from a shoulder off the bottom right of the frame, reaching forward and a little inwards
    shoulder = eye + r * (0.52 + bob[0]) - u * (0.70 + bob[1]) + f * (0.05 + 0.1 * reach)
    d = f * (0.95 + 0.25 * s) - r * (0.30 + 0.25 * s) + u * (0.42 + 0.35 * s)
    q = arm_quat(d, u * 0.3 + r * 0.7)
    if item is None:
        out.append(('player_arm_r', [*shoulder, *q, sc, sc, sc, SKIN_LAYER[skin], 1, 1, 1, 0, gfx.MAT_HAND]))
        out.append(('player_sleeve_r', [*shoulder, *q, sc, sc, sc, SKIN_LAYER[skin], 1, 1, 1, 0, gfx.MAT_HAND]))
    else:
        # the item held in the hand (the arm itself is hidden when holding something, like the game)
        names = item_names or []
        hand = eye + f * (0.62 + 0.15 * s) + r * (0.38 + bob[0]) - u * (0.40 + bob[1] - 0.1 * s)
        qi = qmul(base_q, qmul(qz(-0.5 - 0.4 * s), qmul(qx(0.2 + 0.5 * s), qy(0.25))))
        k = names.index(item) if item in names else 0
        out.append(('item_' + item, [*hand, *qi, 0.42, 0.42, 0.42, k, 1, 1, 1, 0, gfx.MAT_HAND]))
    return out
