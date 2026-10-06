"""MLG water bucket from the build limit: standing on one block at Y=320 over the Sift valley, a water bucket in hand;
a step off, a fall of 220 blocks through the clouds, spinning slowly, the Y counter racing down; the bucket placed at
the last instant on the spire's top, the splash, the water scooped back up, and the view sweeping up over the valley.

python mlg.py OUT_DIR [--from SEC --to SEC] [--res 1440] [--ss 2]   (a segment of frames -> OUT_DIR/video.mp4)
python mlg.py OUT_DIR --audio                                        (the soundtrack -> OUT_DIR/audio.wav)
"""
import argparse
import os

import numpy as np
from PIL import Image

import audio as AU
import director as DR
import entities as EN
import flight as FL
import fx
import gfx
import looks
import pixelfont as PF
import props as PR
import textures as TX
import thumbnail as TH
import voxel as VX
from main import Writer

FPS = 60
TOP = np.array([0.5, 0.5, 100.0])                 # the middle of the spire's top: the target
Z0 = 320.0 + 1.62                                 # the eye, standing on the block at the build limit
T_STEP, VT, TAU = 0.62, 62.0, 1.15                # the step off, terminal speed, how fast it gets there
EYE_WATER = 100.62                                # the eye with the feet in the water on the spire's top
END = 10.0


def fall_d(a):
    return VT * (a - TAU * (1.0 - np.exp(-a / TAU))) if a > 0 else 0.0


def _solve_impact():
    lo, hi = 0.0, 20.0
    for _ in range(60):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if fall_d(m) < Z0 - EYE_WATER else (lo, m)
    return T_STEP + (lo + hi) / 2


T_IMPACT = _solve_impact()
T_PLACE = T_IMPACT - 0.13
T_PICK = T_IMPACT + 0.78
YAW0, SPIN = 90.0, 21.0                           # looking north; the slow spin of the fall (deg/s)


def smooth(a, b, x):
    return FL.smooth(a, b, x)


def state(t):
    """(eye, yaw, pitch, fov, speed, shake) at time t."""
    a = t - T_STEP
    if t < T_STEP:
        eye = np.array([0.5, 0.15 + 0.25 * smooth(0.25, T_STEP, t), Z0 + 0.03 * np.sin(t * 5.0)])
        v = 0.0
    elif t < T_IMPACT:
        z = Z0 - fall_d(a)
        drift = smooth(0.0, 1.2, a)
        eye = np.array([0.5, 0.40 + 0.10 * drift, z])
        v = VT * (1.0 - np.exp(-a / TAU))
    else:
        eye = np.array([0.5, 0.5, 101.62 - (101.62 - EYE_WATER) * (1.0 - smooth(T_IMPACT, T_IMPACT + 0.5, t))])
        v = 0.0
    yaw_fall = YAW0 + SPIN * max(0.0, min(t, T_IMPACT) - T_STEP)
    up = smooth(T_PICK + 0.05, T_PICK + 2.6, t)
    yaw = yaw_fall + (400.0 - yaw_fall) * up + 4.0 * max(0.0, t - T_PICK)
    pitch = -80.0 + (-7.0) * smooth(T_STEP - 0.2, T_STEP + 0.6, t) + 65.0 * up
    vf = VT * (1.0 - np.exp(-max(0.0, min(t, T_IMPACT) - T_STEP) / TAU))
    fov = 72.0 + 30.0 * smooth(8.0, 55.0, vf) * (1.0 - smooth(T_IMPACT, T_IMPACT + 0.35, t)) + 10.0 * up
    shake = 0.006 + 0.012 * smooth(20.0, 60.0, v) + 0.06 * np.exp(-max(0.0, t - T_IMPACT) / 0.18) * (t >= T_IMPACT)
    return eye, yaw, pitch, fov, v, shake


def view(t):
    eye, yaw, pitch, fov, v, shake = state(t)
    y, p = np.radians(yaw), np.radians(pitch)
    F = np.array([np.cos(p) * np.cos(y), np.cos(p) * np.sin(y), np.sin(p)])
    R = np.array([np.sin(y), -np.cos(y), 0.0])
    U = np.cross(R, F)
    w0 = 2 * np.pi
    jx = np.sin(t * w0 * 3.7) * np.sin(t * w0 * 1.3) * shake
    jy = np.sin(t * w0 * 2.9 + 1.0) * np.sin(t * w0 * 0.9) * shake
    F2 = F + R * jx + U * jy
    F2 /= np.linalg.norm(F2)
    U2 = np.cross(R, F2)
    U2 /= np.linalg.norm(U2)
    cam = dict(eye=eye, target=eye + F2, fov=float(fov), up=U2)
    return cam, eye, (F2, np.cross(F2, U2), U2), v


def item_rows(t, axes, eye, fov, names):
    F, R, U = axes
    k = np.tan(np.radians(39.0)) / np.tan(np.radians(fov / 2))
    s = 0.0
    for tu in (T_PLACE, T_PICK):
        if 0.0 <= t - tu < 0.25:
            s = np.sin(np.pi * (t - tu) / 0.25)
    item = 'bucket' if T_PLACE + 0.08 <= t < T_PICK + 0.08 else 'water_bucket'
    pos = eye + 0.5 * (F * 0.60 * k * (1.0 + 0.12 * s) + R * (0.215 - 0.06 * s) - U * (0.33 - 0.10 * s))
    q = EN.basis_quat(F, R, U)
    q = EN.qmul(q, EN.qmul(EN.qz(-0.5), EN.qmul(EN.qx(0.25 - 0.6 * s), EN.qy(0.15))))
    lay = names.index(item) if item in names else 0
    return [('item_' + item, [*pos, *q, 0.125, 0.125, 0.125, lay, 1, 1, 1, 0, gfx.MAT_HAND])]


def _h(i, j, s):
    return (np.sin(i * 127.1 + j * 311.7 + s * 74.7) * 43758.5453) % 1.0


def vnoise(x, y, s):
    """Smooth value noise on a unit grid."""
    i, j = np.floor(x), np.floor(y)
    u, v = x - i, y - j
    u, v = u * u * (3 - 2 * u), v * v * (3 - 2 * v)
    a, b, c, d = _h(i, j, s), _h(i + 1, j, s), _h(i, j + 1, s), _h(i + 1, j + 1, s)
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v


_LAND, _CLOUDS = [], []


def land_rows(at, meta):
    """Beyond the valley's edges, more of the dimension as it looks from up here: 16-block tiles (one texel to a
    block) of pink meadow with patches of teal sculk, rolling a few blocks, and mesas of siftslate; 48-block tiles
    further out, in the haze. The valley itself is left as it is."""
    if _LAND:
        return _LAND
    o, sz = np.asarray(meta['origin'], float), np.asarray(meta['size'], float)
    x0, y0, x1, y1 = o[0], o[1], o[0] + sz[0], o[1] + sz[1]

    def tile(tx, ty, n):
        cx, cy = tx + n / 2, ty + n / 2
        if tx < x1 and tx + n > x0 and ty < y1 and ty + n > y0:
            return
        h = 4.0 + 8.0 * vnoise(cx / 90, cy / 90, 1) + 3.0 * vnoise(cx / 31, cy / 31, 2)
        m = vnoise(cx / 150, cy / 150, 3)
        lay = 'sift_grass_top'
        if m > 0.70:
            h += 12.0 + 60.0 * (m - 0.70)
            lay = 'siftslate_top'
        elif vnoise(cx / 45, cy / 45, 4) > 0.66:
            lay = 'healthy_sculk'
        top = 41.0 + np.round(h)
        _LAND.append(('prop_cube', [cx, cy, (top + 20.0) / 2, *EN.QI, n, n, top - 20.0, at[lay], 1, 1, 1, 0.0,
                                    gfx.MAT_TERRAIN]))
    B = 48.0
    for I in range(-19, int(np.ceil(sz[0] / B)) + 19):
        for J in range(-19, int(np.ceil(sz[1] / B)) + 19):
            bx, by = x0 + I * B, y0 + J * B
            d = np.hypot(max(x0 - bx - B, 0.0, bx - x1), max(y0 - by - B, 0.0, by - y1))
            if d < 260.0:
                for a in range(3):
                    for b in range(3):
                        tile(bx + 16.0 * a, by + 16.0 * b, 16.0)
            else:
                tile(bx, by, B)
    return _LAND


def cloud_rows(at, t):
    """Minecraft's clouds: 12-block cells, 4 blocks thick, at Y=192, drifting east. Clear over the fall line and
    thinner over the valley, so the target shows through them from the top."""
    C = 12.0
    if not _CLOUDS:
        n = 36
        for j in range(-n, n):
            run = None
            cy = 0.5 + (j + 0.5) * C
            for i in range(-n, n + 1):
                cx = 0.5 + (i + 0.5) * C
                d = np.hypot(cx - 0.5, cy - 0.5)
                v = 0.6 * vnoise(cx / 60, cy / 60, 7) + 0.4 * vnoise(cx / 22, cy / 22, 8)
                on = i < n and d > 16.0 and v > 0.5 + 0.14 * np.exp(-(d / 110.0) ** 2)
                if on and run is None:
                    run = i
                elif not on and run is not None:
                    _CLOUDS.append((0.5 + run * C, cy, (i - run) * C))
                    run = None
    dx = 0.8 * t
    return [('prop_cube', [xs + L / 2 + dx, cy, 194.0, *EN.QI, L, C, 4.0, at['white'], 1.0, 0.99, 0.97, 0.08,
                           gfx.MAT_ENTITY]) for xs, cy, L in _CLOUDS]


_TOAST = {}


def toast(img, t, k):
    """The advancement toast, sliding in from the right after the clutch: "Advancement Made! MLG Water Bucket"."""
    t_in, t_out = T_PICK + 0.25, END - 0.5
    if t < t_in:
        return
    u = max(1, int(round(4.5 * k)))
    if u not in _TOAST:
        W, H = 160 * u, 32 * u
        col = np.zeros((H, W, 3), np.float32)
        al = np.zeros((H, W, 1), np.float32)
        al[u:H - u] = 0.94
        al[:, u:W - u] = 0.94
        col[u:H - u, u:W - u] = (92, 92, 92)
        col[2 * u:H - 2 * u, 2 * u:W - 2 * u] = (33, 33, 33)
        ic = np.repeat(np.repeat(TX.bucket_icon(True), u, 0), u, 1)
        for spr, x, y in ((ic, 8 * u, 8 * u),
                          (PF.render('Advancement Made!', px=u, color=(255, 255, 0), outline=0, shadow=True), 30 * u,
                           7 * u),
                          (PF.render('MLG Water Bucket', px=u, color=(255, 255, 255), outline=0, shadow=True), 30 * u,
                           18 * u)):
            h, w = spr.shape[:2]
            a = spr[..., 3:4] / 255.0
            col[y:y + h, x:x + w] = col[y:y + h, x:x + w] * (1 - a) + spr[..., :3] * a
        _TOAST[u] = (col, al)
    col, al = _TOAST[u]
    H, W = col.shape[:2]
    slide = 1.0 - smooth(t_in, t_in + 0.3, t) + smooth(t_out, t_out + 0.3, t)
    x0 = int(round(img.shape[1] - W - 6 * u + slide * (W + 6 * u)))
    y0 = int(round(200 * k))
    xa, xb = max(x0, 0), min(x0 + W, img.shape[1])
    if xb > xa:
        c, a = col[:, xa - x0:xb - x0], al[:, xa - x0:xb - x0]
        img[y0:y0 + H, xa:xb] = img[y0:y0 + H, xa:xb] * (1 - a) + c * a


def splash(t):
    b = t - T_IMPACT
    if not (0.0 <= b < 1.2):
        return np.zeros((0, 8), np.float32), np.zeros((0, 8), np.float32)
    k = np.arange(160)
    h1, h2, h3 = (np.sin(k * 3.17) * 43758.5453) % 1, (np.sin(k * 7.41) * 43758.5453) % 1, \
        (np.sin(k * 5.03) * 43758.5453) % 1
    ang = h1 * 2 * np.pi
    sp = 2.0 + 5.0 * h2
    x = TOP[0] + np.cos(ang) * sp * b
    y = TOP[1] + np.sin(ang) * sp * b
    z = TOP[2] + 0.9 + (3.0 + 4.0 * h3) * b - 9.0 * b * b
    life = np.clip(1.0 - b / 1.2, 0, 1)
    glow = np.column_stack([x, y, z, np.full(160, 0.07), np.full(160, 1.6), np.full(160, 2.2), np.full(160, 2.8),
                            life * (z > TOP[2])]).astype(np.float32)
    soft = np.column_stack([x[:40], y[:40], np.maximum(z[:40], TOP[2] + 0.5), 0.6 + 1.6 * b * np.ones(40),
                            np.full(40, 0.85), np.full(40, 0.92), np.full(40, 1.0), np.full(40, 0.35 * life)]).astype(np.float32)
    return soft, glow


def render_frame(r, fl, meta, f, at):
    t = f / FPS
    cam, eye, axes, v = view(t)
    env = looks.get(DR.LOOK)
    names = getattr(r, '_item_names', [])
    rows = item_rows(t, axes, eye, cam['fov'], names)
    # the block at the build limit; the water on the spire's top; the plain round the world, far below its edges
    rows.append(('prop_cube', [0.5, -0.5, 319.5, *EN.QI, 1, 1, 1, at['siftslate_bricks'], 1, 1, 1, 0, gfx.MAT_ENTITY]))
    if T_PLACE <= t < T_PICK and eye[2] > TOP[2] + 0.9:
        hw = 0.88 * min(1.0, (t - T_PLACE) / 0.06)
        rows.append(('prop_cube', [TOP[0], TOP[1], TOP[2] + 0.02 + hw / 2, *EN.QI, 0.998, 0.998, hw, at['water'],
                                   0.75, 0.9, 1.1, 0.0, gfx.MAT_ENTITY]))
    rows.append(('prop_cube', [70.0, 60.0, 40.6, *EN.QI, 2400.0, 2400.0, 0.6, at['sift_grass_top'], 0.6, 0.5, 0.5,
                               0.0, gfx.MAT_TERRAIN]))
    rows += land_rows(at, meta) + cloud_rows(at, t)
    for (p, yaw, ph) in fx._ST.get('blubs', []):
        if np.linalg.norm(p - eye) < 110.0:
            rows += fx.mobs.blub_rows(p, yaw, t, ph)
    sp_soft, sp_glow = splash(t)
    f_soft, f_glow = fx.falls_spray(meta, t, eye)
    parts = dict(soft=np.concatenate([sp_soft, f_soft]),
                 glow=np.concatenate([sp_glow, f_glow, fx.motes(eye, t, 7.0, 56.0, 44.5, 58.0, (0.9, 2.2, 2.6), 0.08,
                                                                  0.6, 4.1, rise=4.0)]))
    r.use_world('valley')
    r.time = t
    dt = 0.5 / FPS * (1.0 - 0.85 * np.exp(-((t - T_IMPACT) / 0.05) ** 2))
    mb = dict(cam0=view(t - dt / 2)[0], cam1=view(t + dt / 2)[0], maxpx=64.0 * r.W / DR.W, taps=20)
    r.render(cam, env, instances=PR.rows_to_instances(rows), lights=None, particles=parts, streaks=None,
             clip_z=(-1e9, 1e9), mblur=mb, plant_dist=320.0)
    img = r.finish(env).astype(np.float32)
    # under the water for a moment
    if T_PLACE <= t < T_PICK and eye[2] < TOP[2] + 0.9:
        a = 0.55 * min(1.0, (TOP[2] + 0.9 - eye[2]) / 0.25)
        img = img * (1 - a) + np.array([40.0, 90.0, 200.0]) * a
    # the debug screen's height
    yv = max(100, int(np.floor(eye[2] - 1.62 + 1e-6)))
    k = r.W / DR.W
    spr = PF.render(f'Y: {yv}', px=max(2, int(round(8 * k))), color=(255, 255, 255), outline=0, shadow=True)
    x0, y0 = int(44 * k), int(150 * k)
    h, w = spr.shape[:2]
    img[y0 - 8:y0 + h + 8, x0 - 10:x0 + w + 10] *= 0.55
    al = spr[..., 3:4] / 255.0
    img[y0:y0 + h, x0:x0 + w] = img[y0:y0 + h, x0:x0 + w] * (1 - al) + spr[..., :3] * al
    toast(img, t, k)
    return np.clip(img + 0.5, 0, 255).astype(np.uint8)


def soundtrack(path):
    rng = np.random.default_rng(9)
    dur = END
    mix = AU.Mix(dur + 3.0)
    n = int(dur * AU.SR)
    tt = np.arange(n) / AU.SR
    v = np.array([state(x)[4] for x in tt[::480]])
    v = np.interp(tt, tt[::480], v)
    sp = np.clip(v / VT, 0, 1)
    # the wind: a breeze up there, a roar as the fall picks up, cut dead at the splash
    cut = np.where(tt < T_IMPACT, 1.0, np.exp(-(tt - T_IMPACT) / 0.05))
    env = (0.12 + 0.95 * sp ** 1.5) * cut
    w1 = AU.shaped_noise(dur + 0.5, lambda t: 260 + 1500 * np.interp(t, tt, sp), lambda t: 1.4 + 0 * t,
                         lambda t: np.interp(t, tt, env), rng)[:n]
    w2 = AU.shaped_noise(dur + 0.5, lambda t: 2400 + 2600 * np.interp(t, tt, sp), lambda t: 0.7 + 0 * t,
                         lambda t: np.interp(t, tt, env * sp * 0.6), rng)[:n]
    mix.add2(0.55 * AU.norm(w1 + w2), 0.55 * AU.norm(np.roll(w1, 900) + np.roll(w2, 600)), 0.0)
    after = np.clip((tt - T_PICK) / 0.5, 0, 1)
    mix.add(AU.shaped_noise(dur + 0.5, lambda t: 500 + 0 * t, lambda t: 1.2 + 0 * t,
                            lambda t: np.interp(t, tt, 0.10 * after), rng)[:n], 0.0, 0.5)
    # the step, the dread: a drone and a riser, a heartbeat quickening
    mix.add(AU.grass_step(rng) if hasattr(AU, 'grass_step') else AU.kick(1) * 0.2, T_STEP - 0.05, 0.4)
    L, R = AU.choir((45, 52, 57), T_IMPACT - T_STEP, rng, attack=2.0, release=0.1, vowel='o')
    mix.add2(L * 0.35, R * 0.35, T_STEP)
    mix.add(AU.riser(rng, T_IMPACT - T_STEP - 0.15, 120, 9000), T_STEP, 0.32)
    tb, gap = T_STEP + 0.4, 0.75
    while tb < T_IMPACT - 0.2:
        mix.add(AU.kick(0) * 0.6, tb, 0.42)
        mix.add(AU.kick(1) * 0.4, tb + 0.16, 0.3)
        gap = max(0.32, gap * 0.9)
        tb += gap
    # the bucket: the pour, the splash, the hit; then the scoop
    pour = AU.shaped_noise(0.25, lambda t: 900 + 2500 * t, lambda t: 1.0 + 0 * t, lambda t: np.clip(t / 0.05, 0, 1), rng)
    mix.add(AU.norm(pour), T_PLACE, 0.35)
    ns = int(0.9 * AU.SR)
    spl = AU.bp(rng.standard_normal(ns), 200, 6000, 2) * AU.expenv(ns, 0.18, 0.002)
    mix.add(AU.norm(spl), T_IMPACT, 0.9)
    for k in range(10):
        mix.add(AU.gloop(rng) if hasattr(AU, 'gloop') else AU.pluck(48, 0.1, rng), T_IMPACT + 0.05 + 0.06 * k, 0.18)
    mix.add(AU.impact(rng, 1.2), T_IMPACT, 0.85)
    L, R = AU.supersaw([AU.midi_hz(m) for m in (45, 57, 61, 64, 69, 73)], 2.8, rng, attack=0.01, release=1.4,
                       cutoff=5200)
    mix.add2(L * 0.55, R * 0.55, T_IMPACT)
    L, R = AU.choir((57, 61, 64, 69), 3.2, rng, attack=0.1, release=1.5, vowel='a')
    mix.add2(L * 0.4, R * 0.4, T_IMPACT)
    scoop = AU.shaped_noise(0.3, lambda t: 3000 - 2200 * t / 0.3, lambda t: 1.0 + 0 * t,
                            lambda t: np.clip(t / 0.04, 0, 1) * np.exp(-t / 0.12), rng)
    mix.add(AU.norm(scoop), T_PICK, 0.35)
    for k, m in enumerate((69, 73, 76, 81)):
        mix.add(AU.bell(m + 12, rng, dur=2.2), T_PICK + 0.4 + 0.18 * k, 0.16, -0.5 + 0.33 * k)
    sw = AU.shaped_noise(0.4, lambda t: 1200 + 7000 * t, lambda t: 1.0 + 0 * t,
                         lambda t: np.clip(t / 0.08, 0, 1) * np.exp(-t / 0.12), rng)
    mix.add(AU.norm(sw), T_PICK + 0.25, 0.14, 0.4)
    L, R = mix.L[:n], mix.R[:n]
    L, R = AU.reverb(L, R, seed=4, decay=1.6, wet=0.16)
    g = 10 ** ((-14.0 - AU.integrated_lufs(L, R)) / 20.0)
    L, R = AU.limiter(L * g, R * g, ceiling=10 ** (-1.2 / 20))
    AU._write(path, L, R)
    print(f'[audio] {AU.integrated_lufs(L, R):.1f} LUFS', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--from', dest='t0', type=float, default=0.0)
    ap.add_argument('--to', dest='t1', type=float, default=END)
    ap.add_argument('--res', type=int, default=1440)
    ap.add_argument('--ss', type=float, default=2.0)
    ap.add_argument('--stills', default='')
    ap.add_argument('--audio', action='store_true')
    ap.add_argument('--cover', type=float, default=None, help='a cover from the frame at this time -> OUT/thumbnail.jpg')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.audio:
        soundtrack(os.path.join(a.out, 'audio.wav'))
        return
    r, fl, meta = DR.setup(scale=a.res / DR.W, ss=a.ss)
    at = VX.atlas()
    print(f'impact at {T_IMPACT:.2f}s, place {T_PLACE:.2f}s, pick {T_PICK:.2f}s', flush=True)
    if a.cover is not None:
        img = render_frame(r, fl, meta, int(round(a.cover * FPS)), at).astype(np.float32)
        H, W = img.shape[:2]
        k = W / DR.W
        y0 = int(330 * k)
        band = np.zeros(H)
        band[y0 - int(60 * k):y0 + int(330 * k)] = 1.0
        band = np.convolve(band, np.ones(int(120 * k)) / int(120 * k), 'same')
        img *= (1.0 - 0.45 * band)[:, None, None]
        for txt, col, grad, fill in (('MLG FROM THE', (255, 255, 255), None, 0.80),
                                     ('BUILD LIMIT', None, ((150, 230, 255), (40, 130, 255)), 0.92)):
            w1 = PF.render(txt, px=1, outline=1).shape[1]
            spr = PF.render(txt, px=max(1, int(fill * W / w1)), color=col or (255, 255, 255), grad=grad, outline=1)
            x = (W - spr.shape[1]) / 2
            TH.blit(img, spr, x, y0)
            y0 += spr.shape[0] + int(22 * k)
        sp = TH.splash('CLUTCH?!', max(2, int(round(7 * k))))
        TH.blit(img, sp, min(x + spr.shape[1] - sp.shape[1] * 0.55, W - sp.shape[1] - 28 * k), y0 - int(28 * k))
        Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(os.path.join(a.out, 'thumbnail.jpg'), quality=94)
        return
    if a.stills:
        for ts in a.stills.split(','):
            img = render_frame(r, fl, meta, int(round(float(ts) * FPS)), at)
            Image.fromarray(img).save(os.path.join(a.out, f't{float(ts):05.2f}.png'))
        return
    seg = os.path.join(a.out, 'seg')
    os.makedirs(seg, exist_ok=True)
    names = []
    for s0 in range(int(a.t0), int(np.ceil(a.t1))):
        path = os.path.join(seg, f's{s0:02d}.mp4')
        names.append(path)
        if os.path.exists(path):
            continue
        wr = Writer(path, r.W, r.H, FPS)
        for f in range(s0 * FPS, min((s0 + 1) * FPS, int(round(a.t1 * FPS)))):
            wr.write(render_frame(r, fl, meta, f, at))
        wr.close()
        print(f'  segment {s0} done', flush=True)
    with open(os.path.join(seg, 'list.txt'), 'w') as fh:
        fh.writelines(f"file '{os.path.abspath(n)}'\n" for n in names)
    import subprocess
    from main import ffmpeg_exe
    subprocess.run([ffmpeg_exe(), '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', os.path.join(seg, 'list.txt'),
                    '-c', 'copy', os.path.join(a.out, 'video.mp4')], check=True)
    print('rendered', os.path.join(a.out, 'video.mp4'), flush=True)


if __name__ == '__main__':
    main()
