"""Render "A Black Hole Spawns in Minecraft": a flash-forward cold open, then the black hole growing SIZE 1 -> 10 ->
100 -> 1,000 while it rips up the arena and eats the four giants, its collapse and the explosion, with the BLOCKS
EATEN HUD; then the sound design and the YouTube encode.

Usage:
  python main.py --out OUT_DIR [--preview] [--ss 1.5] [--shots a,b] [--every N] [--stills]
                 [--no-audio] [--encode-only]
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
from PIL import Image

import arrows as AR
import blackhole as BH
import scene
import timeline as T
from hud import HUD
from mathutil import quat_rotate

W, H = 1080, 1920
FPS = T.FPS
NEAR_CENTER = (-8.0, -8.0, 12.0)
ORDER = HUD.ORDER                      # zombie, creeper, steve, warden (the portraits' order)
SONIC_COL = np.array([0.12, 0.80, 1.0]) * 6.5
SHOCK_COL = np.array([0.95, 0.80, 1.0]) * 5.0


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return 'ffmpeg'


def encode_youtube(video_in, wav_in, out, workdir, bitrate='12M'):
    """Two-pass H.264 encode following YouTube's recommended upload settings (High profile, closed GOP of
    half the frame rate, 2 B-frames, BT.709 tags) muxed with 384 kbps AAC at 48 kHz, moov atom up front."""
    common = ['-c:v', 'libx264', '-preset', 'slower', '-profile:v', 'high', '-level', '4.1',
              '-b:v', bitrate, '-maxrate', '20M', '-bufsize', '24M', '-pix_fmt', 'yuv420p',
              '-x264-params', 'keyint=15:min-keyint=15:scenecut=0:open-gop=0:bframes=2',
              '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv']
    log = os.path.join(workdir, 'x264pass')
    subprocess.run([ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', video_in] + common +
                   ['-pass', '1', '-passlogfile', log, '-an', '-f', 'null', os.devnull], check=True)
    subprocess.run([ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', video_in, '-i', wav_in, '-map', '0:v:0', '-map',
                    '1:a:0'] + common +
                   ['-pass', '2', '-passlogfile', log, '-c:a', 'aac', '-b:a', '384k', '-ar', '48000', '-ac', '2',
                    '-movflags', '+faststart', '-shortest', '-metadata', 'title=A Black Hole Spawns in Minecraft',
                    out], check=True)


def compact_events(ev):
    return {k: [int(cnt), (psum / pn).tolist() if pn else None] for k, (cnt, psum, pn) in ev.items()}


def skip_to(sim, t):
    """Advance unseen to simulation time t in whole frames (slightly slowed so it ends right on t)."""
    rem = t - sim.t
    if rem <= 1e-6:
        return
    n = int(np.ceil(rem * FPS - 1e-9))
    s = rem * FPS / n
    for _ in range(n):
        sim.step_frame(s)


def smooth01(u):
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3 - 2 * u)


# ---------------------------------------------------------------------------------------------
# pass 1: run the whole story on the edit's schedule without rendering; the HUD's story comes from it
# ---------------------------------------------------------------------------------------------
def analyse(sched):
    w = BH.World(T.plan())
    n = len(sched)
    rec = {'t': np.zeros(n), 'eaten': np.zeros(n), 'ev': [], 'gone': []}
    t0 = time.time()
    for i, (si, f, scale, skip) in enumerate(sched):
        if skip is not None:
            skip_to(w, skip)
        ev = w.step_frame(scale)
        rec['t'][i] = w.t
        rec['eaten'][i] = w.eaten
        rec['ev'].append(compact_events(ev))
        rec['gone'].append([name for name, _ in w.eaten_bodies])
        if i % 60 == 0:
            print(f'  [analyse] frame {i}/{n}  sim t={w.t:.2f}  eaten {w.eaten:.0f}  ({time.time() - t0:.0f}s)',
                  flush=True)
    return rec


def story(rec, sched):
    """When the HUD does what: SIZE banners, EATEN stamps, the counter's displayed value and pulse, flashes."""
    n = len(sched)
    t = rec['t']
    hole = BH.Hole(T.plan()['hole'])
    banners = []                                                # (frame, label)
    for i in range(n):
        if 'stage' in rec['ev'][i]:
            banners.append((i, T.SIZE_LABELS[hole.stage(t[i])]))
    reborn = next((i for i in range(n) if t[i] >= T.REBORN), None)
    if reborn is not None:
        banners.append((reborn + T.sec(0.15), T.SIZE_LABELS[0]))
    eaten = {}
    for i in range(n):
        for name in rec['gone'][i]:
            if name in ORDER and ORDER.index(name) not in eaten:
                eaten[ORDER.index(name)] = i
    stamps = []
    for gi, fe in sorted(eaten.items(), key=lambda x: x[1]):
        f0 = fe + T.sec(0.45 if ORDER[gi] == 'creeper' else 0.2)
        nxt = min([b for b, _ in banners if b > f0] + [n])
        stamps.append((f0, gi, min(1.5, (nxt - f0) / FPS - 0.05)))
    disp = np.zeros(n)
    pulse = np.zeros(n)
    for i in range(n):
        prev = disp[i - 1] if i else 0.0
        disp[i] = prev + (rec['eaten'][i] - prev) * 0.35
        jump = rec['eaten'][i] - (rec['eaten'][i - 1] if i else 0.0)
        pulse[i] = max((pulse[i - 1] if i else 0.0) * 0.8, float(np.clip(jump / 400.0, 0.0, 1.0)))
    spawn = next((i for i in range(n) if 'spawn' in rec['ev'][i]), 0)
    boom = next((i for i in range(n) if 'explode' in rec['ev'][i]), n)
    return {'banners': banners, 'eaten': eaten, 'stamps': stamps, 'disp': disp, 'pulse': pulse, 'spawn': spawn,
            'boom': boom}


def hud_state(i, st, t_sim, R):
    s = {'eaten': {gi: (i - fe) / FPS for gi, fe in st['eaten'].items() if i >= fe}}
    if i >= st['spawn']:
        s['counter'] = (int(round(st['disp'][i])), float(st['pulse'][i]), smooth01((i - st['spawn']) / (0.35 * FPS)))
    for fb, label in st['banners']:
        tb = (i - fb) / FPS
        if 0 <= tb < 1.6:
            s['size'] = (label, tb)
    for f0, gi, dur in st['stamps']:
        te = (i - f0) / FPS
        if 0 <= te < dur:
            s['stamp'] = (gi, te + max(0.0, 1.5 - dur) * smooth01((te - (dur - 0.3)) / 0.3))
    flash = 0.0
    if 0 <= i - st['spawn'] < 8:
        flash = max(flash, 0.45 * (1.0 - (i - st['spawn']) / 8.0))
    if 0 <= i - st['boom'] < 14:
        flash = max(flash, (1.0 - (i - st['boom']) / 14.0) ** 1.8)
    s['flash'] = flash
    c0, c1 = T.COLLAPSE
    if c0 <= t_sim < T.BOOM:
        s['dark'] = 0.3 * smooth01((t_sim - c0) / (c1 - c0))
    s['vignette'] = 0.18 + 0.4 * min(1.0, R / 9.0)
    return s


# ---------------------------------------------------------------------------------------------
# the black hole's look over time
# ---------------------------------------------------------------------------------------------
def mood(t):
    """0: a sunny day; 1: the sky darkened by the SIZE 1,000 black hole; darker still while it collapses, then
    the light comes back after the explosion."""
    hole = BH.Hole(T.plan()['hole'])
    if t < hole.spawn:
        return 0.0
    if t < hole.collapse[0]:
        return float(np.clip(hole.radius(t) / 9.0, 0, 1)) ** 0.6
    if t < hole.boom:
        return 1.0 + 0.35 * smooth01((t - hole.collapse[0]) / (hole.boom - hole.collapse[0]))
    return 1.35 * np.exp(-(t - hole.boom) / 0.8)


def bh_params(w, t):
    h = w.hole
    R = h.radius(t)
    C = h.centre(t)
    m = mood(t)
    sky_mul = tuple(np.clip(1.0 + (np.array([0.52, 0.40, 0.48]) - 1.0) * m, 0.12, 1.0))
    light_mul = tuple(np.clip(1.0 + (np.array([0.74, 0.64, 0.64]) - 1.0) * m, 0.2, 1.0))
    if R <= 0.0:
        if t >= T.REBORN:
            R = 0.32 * float(BH._ease_back((t - T.REBORN) / 0.3, 1.4))
            C = np.array(T.REBORN_C, float)
        else:
            return dict(c=C, r=0.0, sky_mul=sky_mul, light_mul=light_mul, hot=1.5) if m > 0.005 else None
    s = float(np.clip(R / 9.0, 0, 1)) ** 0.6
    u = 0.0
    if h.collapse[0] <= t < h.collapse[1]:
        u = (t - h.collapse[0]) / (h.collapse[1] - h.collapse[0])
    bright = (0.85 + 0.6 * min(1.0, R / 4.0)) * (1.0 + 2.5 * u * u)
    rin = R * 1.12
    rout = R * (2.1 - 0.6 * u) + min(1.2, 1.5 * R)
    return dict(c=C, r=R, lens=1.6 - 0.3 * s, lens_fade=(1.5, 2.5), sway=min(2.5, 0.35 * R), time=t,
                sky_mul=sky_mul, light_mul=light_mul, hot=1.5,
                disk=dict(n=h.normal, rin=rin, rout=rout, rot=2.0 * t, bright=bright, beam=(0.8, 0.0)))


def pulse_rings(w, t, eye):
    """A glowing ring bursts out of the hole when it appears and every time it grows a size (and when the tiny new
    one pops up at the end); drawn facing the camera."""
    h = w.hole
    events = [(ts, max(rs, 0.3), prev) for (ts, rs), prev in zip(h.stages, [0.0] + [s[1] for s in h.stages[:-1]])]
    events.append((T.REBORN, 0.32, 0.0))
    out = []
    for ts, rs, prev in events:
        age = t - ts
        if 0.0 <= age < 0.55:
            C = h.centre(ts) if ts < T.REBORN else np.array(T.REBORN_C, float)
            axis = np.asarray(eye, float) - C
            axis /= np.linalg.norm(axis) + 1e-9
            u = age / 0.55
            rad = max(prev, rs * 0.5) + (rs * 3.2 + 2.0) * (1.0 - (1.0 - u) ** 2)
            out.append([*C, *axis, rad, (1.0 - u) ** 1.5 * 0.9])
    return np.array(out, np.float32).reshape(-1, 8) if out else None


def singularity_flash(w, t):
    """Between the collapse and the explosion: a single point of light, swelling."""
    h = w.hole
    if not (h.collapse[1] - 0.05 <= t < h.boom + 0.02):
        return None
    u = float(np.clip((t - h.collapse[1] + 0.05) / (h.boom - h.collapse[1] + 0.05), 0.0, 1.0))
    C = h.centre(t)
    return np.array([[*C, 1.5 + 5.0 * u ** 2, 2.0 + 10.0 * u ** 2]], np.float32)


def shock_rings(w, t):
    """The explosion's shock waves: a flat ring racing out over the ground and one in the disk plane."""
    h = w.hole
    age = t - h.boom
    if age < 0 or age > 1.6:
        return None
    C = h.centre(h.boom)
    out = []
    for k, (axis, speed, delay) in enumerate((((0.0, 0.0, 1.0), 75.0, 0.0), (tuple(h.normal), 60.0, 0.06),
                                              ((0.0, 0.0, 1.0), 45.0, 0.18))):
        a = age - delay
        if a <= 0:
            continue
        rad = 2.0 + speed * a * (1.0 - 0.25 * min(1.0, a))
        alpha = (1.0 - a / 1.4) ** 1.5 if a < 1.4 else 0.0
        if alpha > 0.01:
            out.append([*C, *axis, rad, alpha * (1.0 if k == 0 else 0.7)])
    return np.array(out, np.float32).reshape(-1, 8) if out else None


# ---------------------------------------------------------------------------------------------
# cameras, fx
# ---------------------------------------------------------------------------------------------
class Shaker:
    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.ph = rng.uniform(0, 2 * np.pi, (3, 3))
        self.fr = np.array([7.3, 11.1, 17.9])

    def offset(self, t, amp):
        o = np.zeros(3)
        for a in range(3):
            o[a] = sum(np.sin(2 * np.pi * self.fr[k] * t + self.ph[a, k]) / (k + 1) for k in range(3))
        return o * amp


def shake_kick(ev, eye):
    def att(name):
        e = ev.get(name)
        if not e or e[1] is None:
            return 1.0
        return float(np.clip(40.0 / (np.linalg.norm(np.asarray(e[1]) - eye) + 5.0), 0.3, 1.6))

    k = 0.0
    for name, amt in (('spawn', 0.6), ('stage', 2.2), ('lift', 1.0), ('sever', 1.4), ('eaten', 1.2),
                      ('boom', 2.0), ('collapse', 0.6)):
        if name in ev:
            k += amt * att(name)
    if 'creeper_boom' in ev:
        k += 6.0 * att('creeper_boom')
    if 'tnt_boom' in ev:
        k += np.log1p(ev['tnt_boom'][0]) * 0.8 * att('tnt_boom')
    if 'explode' in ev:
        k += 10.0
    if 'peel' in ev:
        k += np.log1p(ev['peel'][0]) * 0.05
    return k


def fade_props(props, eye, d0=1.3, d1=3.4):
    """Screen-door fade for blocks, anvils and TNT passing close to the lens."""
    if props is None or not len(props):
        return props
    dist = np.linalg.norm(props[:, :3].astype(np.float64) - eye, axis=1)
    u = np.clip((dist - d0) / (d1 - d0), 0.0, 1.0)
    fade = u * u * (3 - 2 * u) * props[:, 9]
    out = props[fade > 0].copy()
    out[:, 9] = fade[fade > 0]
    return out


def fade_arrows(arrows, eye, d0=0.8, d1=2.0):
    if arrows is None or not len(arrows):
        return arrows
    p = arrows[:, :3].astype(np.float64)
    d = quat_rotate(arrows[:, 3:7].astype(np.float64), np.tile([1.0, 0.0, 0.0], (len(arrows), 1)))
    a = p - d * AR.TIP
    ab = d * (2 * AR.TIP)
    tt = np.clip(np.sum((np.asarray(eye) - a) * ab, 1) / np.sum(ab * ab, 1), 0.0, 1.0)
    dist = np.linalg.norm(a + ab * tt[:, None] - eye, axis=1)
    u = np.clip((dist - d0) / (d1 - d0), 0.0, 1.0)
    fade = u * u * (3 - 2 * u)
    out = arrows[fade > 0].copy()
    out[:, 9] = fade[fade > 0]
    return out


def boom_charge(t):
    """Extra glow while the Warden charges and fires its sonic boom."""
    tb = T.plan()['giants']['warden']['boom']
    dt = t - tb
    if -0.7 < dt < 0.0:
        return ((dt + 0.7) / 0.7) ** 2 * 3.0
    if 0.0 <= dt < 0.5:
        return 3.0 * (1.0 - dt / 0.5)
    return 0.0


class Tracker:
    """Camera target: the shot's keyed target, partly pulled towards a flying giant (smoothed)."""

    def __init__(self):
        self.p = None

    def target(self, w, tgt, track):
        if not track:
            self.p = None
            return tgt
        name, wt = track
        body = next((b for b in w.bodies if b.name == name and b.state in ('lean', 'fly') and b.count > 0), None)
        goal = tgt if body is None else tgt * (1 - wt) + body.com() * wt
        self.p = goal if self.p is None else self.p + (goal - self.p) * 0.25
        return self.p


def render_frame(r, w, cam, preview):
    vox, ranges, props, arrows, fx = w.instances()
    eye = np.asarray(cam['eye'], float)
    nb = ranges[-1][1] if ranges else 0
    if len(vox) > nb:
        tail = vox[nb:]
        keep = np.linalg.norm(tail['pos'].astype(np.float64) - eye, axis=1) > 1.6
        vox = np.concatenate([vox[:nb], tail[keep]])
    props = {k: fade_props(v, eye) for k, v in props.items()}
    arrows = fade_arrows(arrows, eye)
    gl = [(a, b, 0.0, float(bd.white), float(bd.swell), bd.g.center, bd.pose()) for a, b, bd in ranges]
    if w.ground.dirty:
        r.set_ground_region(*w.ground.mesh())
    rings = fx.get('rings')
    shock = shock_rings(w, w.t)
    pulse = pulse_rings(w, w.t, eye)
    r.ring_col = SONIC_COL
    if shock is not None:
        fx['rings'] = shock
        r.ring_col = SHOCK_COL
    elif pulse is not None:
        fx['rings'] = pulse
        r.ring_col = SHOCK_COL
    elif rings is not None and len(rings):
        fx['rings'] = rings
    sing = singularity_flash(w, w.t)
    if sing is not None:
        fl = fx.get('flashes')
        fx['flashes'] = sing if fl is None or not len(fl) else np.concatenate([fl, sing])
    r.render(cam, voxels=vox, props=props, arrows=arrows, fx=fx, near_center=NEAR_CENTER, giants=gl,
             glow=5.0 + 2.0 * boom_charge(w.t), sculk=(0.0, 0.0), bh=bh_params(w, w.t))
    img = r.finish()
    if preview:
        img = np.array(Image.fromarray(img).resize((W, H), Image.BILINEAR))
    return img


def bodies_info(w, eye):
    out = []
    for b in w.bodies:
        if b.state == 'gone':
            continue
        c = b.com()
        out.append({'name': b.name, 'state': b.state, 'dist': float(np.linalg.norm(c - eye)),
                    'stretch': float(b.stretch), 'white': float(b.white), 'frac': b.count / max(1, b.total)})
    return out


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--preview', action='store_true')
    ap.add_argument('--ss', type=float, default=1.5, help='supersampling factor (1.5 = delivered quality)')
    ap.add_argument('--shots', default=None, help='render only these shots (comma separated names, or "cold")')
    ap.add_argument('--every', type=int, default=1, help='render every Nth frame only (stills)')
    ap.add_argument('--stills', action='store_true', help='write JPEG stills instead of a video')
    ap.add_argument('--crf', type=int, default=15)
    ap.add_argument('--no-audio', action='store_true')
    ap.add_argument('--fast', action='store_true', help='camera work: skip the analysis pass (no HUD story)')
    ap.add_argument('--encode-only', action='store_true',
                    help='rebuild audio from OUT/cues.json and redo the final encode of OUT/video_noaudio.mp4')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.encode_only:
        import audio
        with open(os.path.join(args.out, 'cues.json')) as fh:
            d = json.load(fh)
        wav = os.path.join(args.out, 'audio.wav')
        audio.build(d['frames'], d['fps'], wav)
        final = os.path.join(args.out, 'final.mp4')
        encode_youtube(os.path.join(args.out, 'video_noaudio.mp4'), wav, final, args.out)
        print('final:', final)
        return

    shots = T.shots()
    sched = T.schedule()
    only = set(args.shots.split(',')) if args.shots else None

    # ---- pass 1
    t0 = time.time()
    if args.fast:
        n = len(sched)
        rec = {'t': np.zeros(n), 'eaten': np.zeros(n), 'ev': [{} for _ in range(n)], 'gone': [[] for _ in range(n)]}
    else:
        rec = analyse(sched)
    st = story(rec, sched)
    print(f'[analyse] {len(sched)} frames in {time.time() - t0:.0f}s; banners {st["banners"]}; eaten {st["eaten"]}; '
          f'stamps {st["stamps"]}; final count {rec["eaten"][-1]:.0f}', flush=True)
    with open(os.path.join(args.out, 'story.json'), 'w') as fh:
        json.dump({'t': rec['t'].tolist(), 'eaten': rec['eaten'].tolist(), 'banners': st['banners'],
                   'stamps': st['stamps'], 'eaten_frames': st['eaten']}, fh)

    # ---- renderer / encoder
    if args.preview:
        r, _ = scene.make_renderer(width=W // 2, height=H // 2, ss=1.0, shadow_res=2048)
    else:
        r, _ = scene.make_renderer(width=W, height=H, ss=args.ss, shadow_res=4096)
    r.sculk_r = 0.0
    hud = HUD(W, H)
    video_path = os.path.join(args.out, 'video_noaudio.mp4')
    enc = None
    if not args.stills:
        cmd = [ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
               '-r', str(FPS), '-i', '-', '-vf', 'scale=out_color_matrix=bt709:out_range=tv',
               '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
               '-c:v', 'libx264', '-preset', 'slow', '-crf', str(args.crf),
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', video_path]
        enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    cues = []
    n_out = [0]
    t_start = time.time()

    def emit(img, cue, tag):
        if enc is not None:
            enc.stdin.write(np.ascontiguousarray(img).tobytes())
        else:
            Image.fromarray(img).save(os.path.join(args.out, f'{tag}.jpg'), quality=90)
        cues.append(cue)
        n_out[0] += 1
        if n_out[0] % 15 == 0:
            el = time.time() - t_start
            print(f'  frame {n_out[0]}  {el:.0f}s  ({el / n_out[0]:.2f}s/frame)', flush=True)

    def wanted(name, f):
        return (only is None or name in only) and f % args.every == 0

    def cue_of(seg, name, f, cut, ev, eye, tgt, scale, w, extra):
        C = w.hole.centre(w.t)
        c = {'seg': seg, 'shot': name, 'shot_frame': f, 'cut': cut, 'events': ev, 'cam': eye.tolist(),
             'tgt': np.asarray(tgt).tolist(), 'scale': float(scale), 't_sim': float(w.t),
             'R': float(w.hole.radius(w.t)), 'stage': int(w.hole.stage(w.t)), 'mood': float(mood(w.t)),
             'hole_dist': float(np.linalg.norm(C - eye)), 'bodies': bodies_info(w, eye),
             'n_blocks': len(w.blocks), 'n_debris': len(w.debris)}
        c.update(extra)
        return c

    # ---- the cold open: Steve being spaghettified, shown first
    for ci, clip in enumerate(T.cold_open()):
        w = BH.World(T.plan())
        skip_to(w, clip['sim'])
        r.set_ground_region(*w.ground.mesh())
        shaker = Shaker(11)
        tracker = Tracker()
        acc = 0.0
        nf = T.sec(clip['dur'])
        i_hud = int(np.argmin(np.abs(rec['t'] - clip['sim'])))
        for k in range(nf):
            ev = compact_events(w.step_frame(clip['scale']))
            eye, tgt, fov = T.cam_path(clip['keys'], k / FPS)
            tgt = tracker.target(w, tgt, clip.get('track'))
            acc = acc * 0.86 + shake_kick(ev, eye)
            off = shaker.offset(k / FPS, 0.14 * np.tanh(acc / 3.0) + 0.012 * w.hole.radius(w.t))
            if not wanted('cold', k):
                continue
            cam = {'eye': eye + off, 'target': tgt + off * 0.6, 'fov': fov, 'roll': float(off[0] * 0.8)}
            img = render_frame(r, w, cam, args.preview)
            hs = {'counter': (int(round(st['disp'][i_hud])), 0.0, 1.0),
                  'eaten': {gi: 9.0 for gi, fe in st['eaten'].items() if fe <= i_hud},
                  'vignette': 0.18 + 0.4 * min(1.0, w.hole.radius(w.t) / 9.0)}
            if k >= nf - 3:
                hs['flash'] = [0.25, 0.55, 0.85][k - (nf - 3)]
            img = hud.draw(img, hs)
            emit(img, cue_of('cold', f'cold{ci}', k, k == 0, ev, eye, tgt, clip['scale'], w, {'hud': {}}),
                 f'cold{ci}_{k:03d}')
    print(f'[render] cold open done ({time.time() - t_start:.0f}s)', flush=True)

    # ---- the main edit
    w = BH.World(T.plan())
    r.set_ground_region(*w.ground.mesh())
    shaker = Shaker(3)
    tracker = Tracker()
    acc = 0.0
    prev_si = None
    for i, (si, f, scale, skip) in enumerate(sched):
        sh = shots[si]
        if skip is not None:
            skip_to(w, skip)
        ev = w.step_frame(scale)
        cev = compact_events(ev) if args.fast else rec['ev'][i]
        eye, tgt, fov = T.cam_path(sh['keys'], f / FPS)
        tgt = tracker.target(w, tgt, sh.get('track'))
        acc = acc * 0.86 + shake_kick(cev, eye)
        amp = sh.get('shake', 0.14) * np.tanh(acc / 3.0) + 0.012 * w.hole.radius(w.t)
        off = shaker.offset(i / FPS, amp)
        hs = hud_state(i, st, w.t, w.hole.radius(w.t))
        if i < 3:
            hs['flash'] = max(hs.get('flash', 0.0), [0.7, 0.35, 0.12][i])
        extra = {'hud': {}, 'counter': float(st['disp'][i]), 'flash': float(hs.get('flash', 0.0))}
        for fb, label in st['banners']:
            if fb == i:
                extra['hud']['banner'] = label
        for f0, gi, _ in st['stamps']:
            if f0 == i:
                extra['hud']['stamp'] = ORDER[gi]
        if wanted(sh['name'], f):
            cam = {'eye': eye + off, 'target': tgt + off * 0.6, 'fov': fov, 'roll': float(off[0] * 0.8)}
            img = render_frame(r, w, cam, args.preview)
            img = hud.draw(img, hs)
            emit(img, cue_of('main', sh['name'], f, si != prev_si, cev, eye, tgt, scale, w, extra),
                 f'{i:04d}_{sh["name"]}_{f:03d}')
        prev_si = si
    print(f'[render] main done ({time.time() - t_start:.0f}s)', flush=True)

    if enc is not None:
        enc.stdin.close()
        enc.wait()
    with open(os.path.join(args.out, 'cues.json'), 'w') as fh:
        json.dump({'fps': FPS, 'frames': cues}, fh)
    print(f'video frames: {n_out[0]}, render time {time.time() - t_start:.0f}s', flush=True)

    if not args.no_audio and enc is not None and only is None and args.every == 1:
        import audio
        wav = os.path.join(args.out, 'audio.wav')
        audio.build(cues, FPS, wav)
        final = os.path.join(args.out, 'final.mp4')
        encode_youtube(video_path, wav, final, args.out)
        print('final:', final)


if __name__ == '__main__':
    main()
