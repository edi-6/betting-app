"""The coaster's own props: the track (rail tiles on a bed of blocks, laid along the banked frame), the minecart,
the rider's hands, and the portal surfaces. Everything here is an instanced kind of the renderer.

Instance rows are (kind, [pos3, quat4, scale3, layer, tint3, emit, mat]) like entities.inst().
"""
import numpy as np

import blocks as BL
import entities as EN
import gfx
import voxel as VX

PX = 1.0 / 16.0
RAIL_H = 0.03           # the rail tile's height over the bed's top face


def quad_mesh(w=1.0, h=1.0, layer=-1.0):
    """A quad in the local x (right) / y (forward) plane, facing up (+z), uv 0..1 (v along y)."""
    x0, x1, y0, y1 = -w / 2, w / 2, -h / 2, h / 2
    P = [(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)]
    UV = [(0, 1), (1, 1), (1, 0), (0, 0)]
    out = []
    for k in (0, 1, 2, 0, 2, 3):
        out.append((*P[k], 0, 0, 1, *UV[k], layer))
    return np.array(out, np.float32)


def vquad_mesh(w=1.0, h=1.0, layer=-1.0):
    """A quad standing in the local x (right) / z (up) plane, facing -y, uv 0..1 (v down)."""
    x0, x1, z0, z1 = -w / 2, w / 2, -h / 2, h / 2
    P = [(x0, 0, z0), (x1, 0, z0), (x1, 0, z1), (x0, 0, z1)]
    UV = [(0, 1), (1, 1), (1, 0), (0, 0)]
    out = []
    for k in (0, 1, 2, 0, 2, 3):
        out.append((*P[k], 0, -1, 0, *UV[k], layer))
    return np.array(out, np.float32)


def cart_boxes():
    """The minecart in block-model units (1/16), its floor centre at (8, 8, 0) = the rails."""
    side, ins, bot = 'minecart_side', 'minecart_inside', 'minecart_bottom'
    out = []
    out.append(BL.cube_box(1, -1, 1, 15, 17, 3, side, per_face={'pz': ins, 'nz': bot}))
    out.append(BL.cube_box(1, 15, 3, 15, 17, 11, side, per_face={'ny': ins}))            # front wall
    out.append(BL.cube_box(1, -1, 3, 15, 1, 11, side, per_face={'py': ins}))             # back wall
    out.append(BL.cube_box(1, 1, 3, 3, 15, 11, side, per_face={'px': ins}))              # left wall
    out.append(BL.cube_box(13, 1, 3, 15, 15, 11, side, per_face={'nx': ins}))            # right wall
    return out


NETHER_PAL = [(52, 6, 124), (92, 16, 190), (132, 44, 236), (176, 96, 255), (214, 150, 255)]
SIFT_PAL = [(8, 62, 76), (20, 128, 140), (96, 196, 192), (214, 96, 150), (244, 186, 214)]


def portal_frames(n=32, size=16, seed=3, pal=None):
    """A portal's swirl, animated: n frames of (size, size, 4) and their emission. Two interleaved spirals,
    drifting: the nether portal's purples, or (pal=SIFT_PAL) the Deep Dark Portal's teal and pink."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] + 0.5
    cx = cy = size / 2
    ang = np.arctan2(yy - cy, xx - cx)
    rad = np.hypot(xx - cx, yy - cy) / size
    jit = rng.random((size, size))
    layers, emits = [], []
    pal = np.array(pal or NETHER_PAL, float)
    for f in range(n):
        ph = 2 * np.pi * f / n
        v = 0.5 + 0.5 * np.sin(ang * 2 + rad * 14.0 - ph * 2 + jit * 0.8)
        v = v * 0.75 + 0.25 * (0.5 + 0.5 * np.sin(xx * 0.9 + yy * 0.7 + ph * 3))
        k = np.clip((v * (len(pal) - 1)).round().astype(int), 0, len(pal) - 1)
        img = np.zeros((size, size, 4), np.uint8)
        img[..., :3] = pal[k]
        img[..., 3] = 255
        layers.append(img)
        emits.append(0.55 + 0.45 * v)
    return layers, emits


def end_portal_layers(n=32, size=32, seed=5):
    """The End portal's surface: black, with specks of teal, green and violet that drift (the game draws it as a
    parallax star field; the layers slide past each other here)."""
    rng = np.random.default_rng(seed)
    cols = np.array([(40, 170, 150), (70, 200, 110), (120, 80, 200), (210, 230, 220)], float)
    stars = []
    for k in range(4):
        m = rng.random((size, size)) < (0.05 - 0.008 * k)
        stars.append((m, cols[k]))
    layers, emits = [], []
    for f in range(n):
        img = np.zeros((size, size, 4), np.uint8)
        img[..., :3] = (6, 10, 14)
        img[..., 3] = 255
        em = np.zeros((size, size))
        for k, (m, c) in enumerate(stars):
            sh = int(round(f * (k + 1) * 0.5)) % size
            mm = np.roll(np.roll(m, sh, 0), sh // 2, 1)
            img[mm, :3] = c
            em[mm] = 0.9
        layers.append(img)
        emits.append(em)
    return layers, emits


def fire_frames(n=16, size=16, seed=8):
    """The game's fire: tongues of flame licking up, yellow at the root, orange to red at the tips (RGBA frames,
    emission)."""
    rng = np.random.default_rng(seed)
    layers, emits = [], []
    base = rng.random((64, size))
    for f in range(n):
        heat = np.zeros((size, size))
        for y in range(size):
            row = base[(y + f * 3) % 64]
            heat[y] = row * (y / (size - 1)) ** 0.6 * 1.35
        heat = (heat + np.roll(heat, 1, 1) + np.roll(heat, -1, 1)) / 3.0
        a = heat > 0.42
        img = np.zeros((size, size, 4), np.uint8)
        c = np.where(heat[..., None] > 0.85, (255, 236, 150), np.where(heat[..., None] > 0.62, (252, 170, 40),
                                                                         (220, 80, 20)))
        img[..., :3] = c
        img[..., 3] = np.where(a, 255, 0)
        layers.append(img)
        emits.append(np.where(a, 1.0, 0.0))
    return layers, emits


def cross_mesh(layer=-1.0):
    """Two vertical quads crossing (like the game's plants and fire), 1 block, standing on z = 0."""
    out = []
    for (dx, dy) in ((1, 1), (1, -1)):
        d = np.array([dx, dy, 0.0]) / np.sqrt(2) * 0.5
        n = np.array([-dy, dx, 0.0]) / np.sqrt(2)
        P = [(-d[0], -d[1], 0), (d[0], d[1], 0), (d[0], d[1], 1), (-d[0], -d[1], 1)]
        UV = [(0, 1), (1, 1), (1, 0), (0, 0)]
        for k in (0, 1, 2, 0, 2, 3):
            out.append((*P[k], *n, *UV[k], layer))
    return np.array(out, np.float32)


def register(r):
    """Every kind the ride draws, on top of the characters of entities.register()."""
    at = VX.atlas()
    S = EN.register(r, at)
    r.add_kind('rail_tile', quad_mesh(), tex='blocks')
    r.add_kind('cart', EN.boxes_mesh(cart_boxes(), at, (8, 8, 0)), tex='blocks')
    pl, pe = portal_frames()
    r.add_kind('nether_portal', vquad_mesh(), pl, emit=pe)
    pl, pe = portal_frames(seed=9, pal=SIFT_PAL)
    r.add_kind('sift_portal', vquad_mesh(), pl, emit=pe)
    el, ee = end_portal_layers()
    r.add_kind('end_portal', quad_mesh(), el, emit=ee)
    fl, fe = fire_frames()
    r.add_kind('fire', cross_mesh(), fl, emit=fe)
    return S


def nether_portal_rows(center, width, height, normal, t, frames=32, fps=16.0, kind='nether_portal', glow=1.0):
    """The portal's surface as one 1x1 quad per block (the texture tiles like the game's), animated. glow < 1
    dims it (a portal waking up)."""
    n = np.asarray(normal, float)
    f = int(t * fps) % frames
    # the quad faces -y in its own frame: turn it so it faces along the normal
    fwd = n / np.linalg.norm(n)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    q = EN.basis_quat(fwd, right, up)
    out = []
    c = np.asarray(center, float)
    for i in range(width):
        for k in range(height):
            p = c + right * (i - (width - 1) / 2) + up * (k - (height - 1) / 2)
            out.append((kind, [*p, *q, 1.0, 1.0, 1.0, (f + i * 3 + k * 5) % frames, glow, glow, glow, 0.9 * glow,
                               gfx.MAT_GLOW]))
    return out


def basis_q(T, R, U):
    return EN.basis_quat(T, R, U)


def track_rows(tr, s0, s1, bed_layer, rail_on=True, broken=None, tint=(1, 1, 1), bed_mat=gfx.MAT_TERRAIN):
    """Rows for the rails and the bed between arc lengths s0 and s1 (one tile per block).
    broken: None or (a, b) arc-length range where the track is gone (the gap)."""
    at = VX.atlas()
    L_rail, L_pw, L_pw_on = at['rail'], at['powered_rail'], at['powered_rail_on']
    out = []
    k0 = max(int(np.floor(s0)), 0)
    k1 = min(int(np.ceil(s1)), int(tr.length) - 1)
    for k in range(k0, k1):
        s = k + 0.5
        if broken is not None and broken[0] <= s <= broken[1]:
            continue
        if tr.in_gap(s):
            continue
        P = tr.pos(s)
        T, R, U = tr.frame(s)
        q = basis_q(T, R, U)
        if tr.is_powered(s):
            lay, em = (L_pw_on, 0.0) if rail_on else (L_pw, 0.0)
        else:
            lay, em = L_rail, 0.0
        out.append(('rail_tile', [*(P + U * RAIL_H), *q, 1.0, 1.04, 1.0, lay, *tint, em, gfx.MAT_ENTITY]))
        out.append(('prop_cube', [*(P - U * 0.5), *q, 1.0, 1.03, 1.0, bed_layer, *tint, 0.0, bed_mat]))
    return out


def cart_rows(pos, T, R, U, tint=(1, 1, 1)):
    q = basis_q(T, R, U)
    return [('cart', [*pos, *q, 1.0, 1.0, 1.0, 0, *tint, 0.0, gfx.MAT_ENTITY])]


def rider_hands(eye, T, R, U, up=0.0, shake=(0.0, 0.0), skin='steve'):
    """The rider's two arms: gripping the front of the cart (up = 0), or thrown up in front of him (up = 1).
    eye/T/R/U: the camera's position and axes."""
    out = []
    sc = PX * 0.9375
    lay = EN.SKIN_LAYER[skin]
    u = float(np.clip(up, 0, 1))
    u = u * u * (3 - 2 * u)
    for side in (1, -1):
        shoulder = eye + R * (0.34 * side) - U * 0.50 - T * 0.02
        grip = eye + T * 0.44 + R * (0.27 * side) - U * 0.60             # resting on the front wall's top edge
        raised = eye + T * 0.22 + R * (0.52 * side) + U * 0.30          # up and out, clear of the view
        hand = grip * (1 - u) + raised * u
        hand = hand + R * shake[0] * side + U * shake[1]
        d = hand - shoulder
        q = EN.arm_quat(d, U * (1 - u) + T * u + R * 0.2 * side)
        kind = 'player_arm_r' if side > 0 else 'player_arm_l'
        sl = 'player_sleeve_r' if side > 0 else 'player_sleeve_l'
        # the arm model reaches 10 px from its shoulder pivot to the hand: stretch it to reach
        L = np.linalg.norm(d) / (10 * sc)
        scale = (sc, sc, sc * L)
        out.append((kind, [*shoulder, *q, *scale, lay, 1, 1, 1, 0, gfx.MAT_HAND]))
        out.append((sl, [*shoulder, *q, *scale, lay, 1, 1, 1, 0, gfx.MAT_HAND]))
    return out


def rows_to_instances(rows):
    d = {}
    for kind, row in rows:
        d.setdefault(kind, []).append(row)
    return {k: np.array(v, np.float32) for k, v in d.items()}
