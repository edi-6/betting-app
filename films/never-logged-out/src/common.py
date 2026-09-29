"""Pieces the sequences share: the moving props of the village (doors, the chest lid, the ceiling block, the trapdoor,
the carpet, the falling sign), the villagers and their routines, the animals, name tags, and scene helpers."""
import numpy as np

import anim as A
import entities as EN
import gfx
import looks
import ui
import voxel as VX
import worldgen as WG

AT = None


def atlas():
    global AT
    if AT is None:
        AT = VX.atlas()
    return AT


def facing_q(f):
    return EN.qz(np.radians(90.0 * f))


def local_to_world(pos, facing, lx, ly, lz):
    """A point given in a block's local 1/16 units (block facing south), in world coordinates."""
    p = np.array([lx, ly, lz], float) / 16.0 - np.array([0.5, 0.5, 0.0])
    a = np.radians(90.0 * facing)
    c, s = np.cos(a), np.sin(a)
    p = np.array([c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]])
    return np.array([pos[0] + 0.5, pos[1] + 0.5, pos[2]], float) + p


def row(kind, pos, q=EN.QI, scale=1.0, layer=0, tint=(1, 1, 1), emit=0.0, mat=gfx.MAT_ENTITY):
    return EN.inst(kind, pos, q, scale, layer, tint, emit, mat)


# ---------------------------------------------------------------------------------------------
# doors, chest, trapdoor, ceiling block, carpet, sign
# ---------------------------------------------------------------------------------------------
def door_props(w, name, angle_deg, wood='oak'):
    """Both halves of a door recorded in w.dyn as name_bottom / name_top, swung open by angle (0 = shut) inwards
    about its hinge (the door's local x = 0 edge)."""
    out = []
    for part in ('bottom', 'top'):
        d = w.dyn[f'{name}_{part}']
        f = d['state'] & 3
        hinge = local_to_world(d['pos'], f, 0, 1.5, 0)
        q = EN.qmul(facing_q(f), EN.qz(np.radians(angle_deg)))
        blk = d['block']
        wd = 'spruce' if blk.startswith('spruce') else 'oak'
        out.append(row(f'prop_{wd}_door_{part}', hinge, q))
    return out


def chest_props(w, name, lid_deg):
    d = w.dyn[name]
    f = d['state'] & 3
    body = local_to_world(d['pos'], f, 8, 8, 0)
    hinge = local_to_world(d['pos'], f, 8, 15, 10)
    q = facing_q(f)
    return [row('prop_chest_body', body, q), row('prop_chest_lid', hinge, EN.qmul(q, EN.qx(-np.radians(lid_deg))))]


def trapdoor_props(w, name, open_deg):
    d = w.dyn[name]
    f = d['state'] & 3
    hinge = local_to_world(d['pos'], f, 8, 16, 13 + 1.5)
    q = EN.qmul(facing_q(f), EN.qx(-np.radians(open_deg)))
    return [row('prop_trapdoor', hinge, q)]


def block_cube(pos, tex, crack=-1, scale=1.0, q=EN.QI):
    """A whole block as a prop (the ceiling block), with the breaking cracks over it."""
    at = atlas()
    c = np.array(pos, float) + 0.5
    out = [row('prop_cube', c, q, scale, at[tex])]
    if crack >= 0:
        out.append(row('prop_cube', c, q, scale * 1.004, at[f'crack_{min(int(crack), 9)}']))
    return out


def carpet(pos, tex='red_wool'):
    at = atlas()
    c = np.array(pos, float) + np.array([0.5, 0.5, 1 / 32])
    return [row('prop_cube', c, EN.QI, (1.0, 1.0, 1 / 16), at[tex])]


def sign_prop(ctx, world, key, pos, facing):
    """A standing sign at a (possibly fractional) block position, with its text."""
    p = local_to_world(pos, facing, 8, 8, 0)
    out = [row('prop_sign', p, facing_q(facing))]
    out.append(ctx.dec[world].sign_row(key, pos, facing, standing=True))
    return out


def breaking_particles(pos, t, tex, seed=3, n=14, dur=0.9):
    """Little cubes of the broken block falling (t: seconds since it broke)."""
    if t < 0 or t > dur:
        return []
    rng = np.random.default_rng(seed)
    at = atlas()
    out = []
    c = np.array(pos, float) + 0.5
    for k in range(n):
        v = rng.normal(0, 1.2, 3)
        v[2] = abs(v[2]) * 0.6
        p0 = c + rng.uniform(-0.4, 0.4, 3)
        p = p0 + v * t + np.array([0, 0, -9.0 * t * t])
        s = rng.uniform(0.07, 0.13) * (1 - 0.3 * t / dur)
        out.append(row('prop_cube', p, EN.qaxis(rng.normal(0, 1, 3), rng.uniform(0, 6)), s, at[tex]))
    return out


def world_props(ctx, world, door=0.0, lid=0.0, ceiling=None, trap=0.0, carpet=True, maphouse_door=0.0,
                copy_door=0.0, real_door=0.0):
    """Every moving block of a world in its current state. ceiling: None (whole) | crack stage 0..9 (float) |
    'gone'."""
    w = ctx.worlds[world]
    out = []
    dyn = w.dyn
    if 'house_door_bottom' in dyn:
        out += door_props(w, 'house_door', door)
    if 'house_chest' in dyn:
        out += chest_props(w, 'house_chest', lid)
    if 'house_ceiling' in dyn and ceiling != 'gone':
        out += block_cube(dyn['house_ceiling']['pos'], 'spruce_planks', crack=-1 if ceiling is None else ceiling)
    if 'house_trapdoor' in dyn:
        out += trapdoor_props(w, 'house_trapdoor', trap)
    if 'house_carpet' in dyn and carpet:
        out += carpet_prop(dyn['house_carpet']['pos'])
    if 'maphouse_door_bottom' in dyn:
        out += door_props(w, 'maphouse_door', maphouse_door)
    if 'copy_door_bottom' in dyn:
        out += door_props(w, 'copy_door', copy_door)
    if 'real_door_bottom' in dyn:
        out += door_props(w, 'real_door', real_door)
    return out


def carpet_prop(pos):
    return carpet(pos)


def ground_z(ctx, x, y, world='village'):
    w = ctx.worlds[world]
    i, j = int(np.floor(x)) - w.origin[0], int(np.floor(y)) - w.origin[1]
    return float(w.heightmap[i, j])


# ---------------------------------------------------------------------------------------------
# villagers: names, skins, routines (loops through the village, in world time so they carry on across shots)
# ---------------------------------------------------------------------------------------------
VILLAGERS = [
    ('Tom', 'villager_farmer', [(-26, -12), (-27, 2), (-10, 2), (-2, 0), (-10, -1), (-26, -3)], 1.1),
    ('Mira', 'villager_librarian', [(-4, 30), (6, 28), (8, 24), (0, 22), (-5, 25)], 0.8),
    ('Elias', 'villager_cartographer', [(12, 6), (14, 3), (9, 1), (6, 6), (9, 10)], 0.7),
    ('Rosa', 'villager_cleric', [(18, 33), (24, 30), (20, 26), (15, 30)], 0.8),
    ('Finn', 'villager_smith', [(26, 16), (30, 12), (24, 10), (20, 14)], 0.9),
    ('June', 'villager_shepherd', [(18, -30), (24, -34), (30, -32), (26, -28)], 0.8),
    ('Oskar', 'villager_farmer', [(-30, -20), (-26, -24), (-20, -16), (-26, -14)], 1.0),
    ('Ada', 'villager_librarian', [(4, 10), (10, 14), (6, 18), (0, 14)], 0.7),
]


def loop_walker(points, speed, z=0.0):
    pts = [(x + 0.5, y + 0.5, z) for (x, y) in points] + [(points[0][0] + 0.5, points[0][1] + 0.5, z)]
    path = A.Path(pts)
    return path, speed


class Village:
    """Everyone living in the village, at world time T (seconds of film time)."""

    def __init__(self, changed=False):
        self.changed = changed
        self.walks = []
        for (name, skin, pts, speed) in VILLAGERS:
            path, sp = loop_walker(pts, speed)
            self.walks.append((name if not changed else 'NOAH', skin, path, sp))

    def villagers(self, T, only=None, skip=()):
        out = []
        for i, (name, skin, path, sp) in enumerate(self.walks):
            if only is not None and name not in only:
                continue
            if name in skip:
                continue
            # walk for a while, stand for a while (villagers stop and look around)
            cyc = 14.0 + 3 * i
            phase = (T + 5.3 * i) % cyc
            walking = phase < cyc * 0.65
            base = np.floor((T + 5.3 * i) / cyc) * cyc * 0.65 * sp + min(phase, cyc * 0.65) * sp
            s = base % path.length
            a = EN.Actor('villager', skin)
            a.pos = path.at(s)
            tan = path.tangent(s)
            a.yaw = np.radians(A.yaw_of(tan))
            a.walk_amp = 1.0 if walking else 0.0
            a.walk = s / 1.2 * np.pi
            a.head_yaw = 0.0 if walking else 0.5 * np.sin(T * 0.7 + i)
            a.idle_t = T
            a.name = name
            if self.changed:
                a.walk_amp = 0.0              # in the changed world they all stand still and stare
            out.append(a)
        return out


def animals(T, changed=False):
    """The pen: two cows, three sheep, a pig; chickens by the houses. Slow grazing drift."""
    if changed:
        return []
    pc = np.array([23.5, -41.5, 0.0])
    specs = [('cow', (-3.0, 1.0), 0.0), ('cow', (2.5, 2.0), 1.3), ('sheep', (0.0, -2.0), 2.1),
             ('sheep', (3.0, -1.0), 3.3), ('sheep', (-2.5, -2.5), 0.7), ('pig', (-1.0, 3.0), 4.1)]
    out = []
    for i, (m, (dx, dy), ph) in enumerate(specs):
        a = EN.Actor(m, m)
        wx = dx + 1.2 * np.sin(T * 0.09 + ph) + 0.5 * np.sin(T * 0.23 + 2 * ph)
        wy = dy + 1.0 * np.cos(T * 0.07 + ph)
        a.pos = pc + np.array([wx, wy, 0.0])
        vx = 1.2 * 0.09 * np.cos(T * 0.09 + ph) + 0.5 * 0.23 * np.cos(T * 0.23 + 2 * ph)
        vy = -1.0 * 0.07 * np.sin(T * 0.07 + ph)
        a.yaw = np.radians(A.yaw_of((vx, vy)))
        a.walk = T * 3.0
        a.walk_amp = float(np.clip(np.hypot(vx, vy) * 3.0, 0, 0.6))
        a.head_pitch = 0.5 + 0.4 * np.sin(T * 0.5 + ph) if m in ('cow', 'sheep') else 0.0
        out.append(a)
    for i, (x, y) in enumerate(((-10.5, -16.5), (-8.5, -15.0), (12.5, -3.0))):
        a = EN.Actor('chicken', 'chicken')
        a.pos = np.array([x + 0.8 * np.sin(T * 0.4 + i), y + 0.8 * np.cos(T * 0.33 + i * 2), 0.0])
        a.yaw = T * 0.3 + i
        a.walk = T * 8
        a.walk_amp = 0.5
        a.head_pitch = 0.3 * max(0.0, np.sin(T * 2.0 + i))
        out.append(a)
    return out


def name_tags(img, cam, actors, max_dist=24.0, alpha=1.0):
    import film as FM
    named = [a for a in actors if getattr(a, 'name', None)]
    if not named:
        return
    pts = np.array([a.head_top() + np.array([0, 0, 0.25]) for a in named])
    xy, depth = FM.project(cam, pts)
    for a, (x, y), dpt in zip(named, xy, depth):
        if dpt <= 0.3 or dpt > max_dist:
            continue
        if -200 < x < ui.W + 200 and -100 < y < ui.H + 100:
            ui.name_tag(img, a.name, x, y, dpt, alpha * float(np.clip((max_dist - dpt) / 4.0, 0, 1)))


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def env(name, **kw):
    return looks.get(name, **kw)


def pov_scene(pov, t, env_name, world='village', actors=(), props=(), fp=None, item=None, swing=0.0, lights=None,
              exclude=(), hold_map=False, skin='you', map_layer=0, extra=None, envkw=None, **kw):
    cam = pov.cam(t)
    fpi = []
    if fp is None or fp:
        fpi = EN.first_person(cam['eye'], cam['target'], skin=skin, item=item, swing=swing, bob=pov.hand_bob(t),
                              hold_map=hold_map, item_names=ITEM_NAMES, map_layer=map_layer)
    e = env(env_name, **(envkw or {})) if isinstance(env_name, str) else env_name
    sc = dict(world=world, env=e, cam=cam, actors=list(actors), props=list(props),
              fp=fpi, lights=lights, exclude=exclude)
    sc.update(kw)
    if extra:
        sc.update(extra)
    return sc


ITEM_NAMES = sorted(__import__('textures').item_icons())


def held_torch_light(cam, t, strength=1.0):
    """The torch in his hand lights things around him (like a dynamic-lights mod)."""
    eye = np.asarray(cam['eye'], float)
    f = np.asarray(cam['target'], float) - eye
    f /= np.linalg.norm(f)
    p = eye + f * 0.6 + np.array([0, 0, -0.3])
    flick = 0.9 + 0.1 * np.sin(t * 13.0) * np.sin(t * 7.3 + 1.0)
    return [[*p, 11.0, 2.4 * strength * flick, 1.5 * strength * flick, 0.75 * strength * flick]]


def world_pts(ctx, variant='village'):
    return ctx.worlds[variant].points
