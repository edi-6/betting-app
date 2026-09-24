"""Render "Last Giant Standing": a flash-forward cold open, the line-up, three rounds (1,000 arrows, 1,000 anvils,
10,000 TNT) and the winner, with the scoreboard HUD; then the sound design and the YouTube encode.

Usage:
  python main.py --out OUT_DIR [--preview] [--ss 1.5] [--shots R1_boom,W_hero] [--every N] [--stills]
                 [--no-audio] [--encode-only]
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
from PIL import Image

import anvil as AN
import arena as A
import arrows as AR
import scene
import timeline as T
from hud import HUD
from mathutil import quat_rotate

W, H = 1080, 1920
FPS = T.FPS
UP = np.array([0.0, 0.0, 1.0])
NEAR_CENTER = (-4.0, 0.0, 14.0)
SCULK = (23.0, 0.0)
TITLES = {1: '1,000 ARROWS', 2: '1,000 ANVILS', 3: '10,000 TNT'}
LETHAL = {T.ZOMBIE: 0.55, T.CREEPER: 0.30, T.STEVE: 0.93}     # the Warden's is set so he ends on half a heart
HURT_GAP = 14


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
    subprocess.run([ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', video_in, '-i', wav_in, '-map', '0:v:0', '-map', '1:a:0']
                   + common + ['-pass', '2', '-passlogfile', log, '-c:a', 'aac', '-b:a', '384k', '-ar', '48000', '-ac', '2',
                               '-movflags', '+faststart', '-shortest', '-metadata', 'title=Last Giant Standing', out],
                   check=True)


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


# ---------------------------------------------------------------------------------------------
# pass 1: run the whole battle on the edit's schedule without rendering; the heart story comes from it
# ---------------------------------------------------------------------------------------------
def analyse(sched):
    sim = A.Arena(T.plan())
    totals = np.array([g.total for g in sim.giants], float)
    n = len(sched)
    rec = {'t': np.zeros(n), 'd': np.zeros((n, 4)), 'ev': [], 'creeper_done': np.zeros(n, bool)}
    t0 = time.time()
    for i, (si, f, scale, skip) in enumerate(sched):
        if skip is not None:
            skip_to(sim, skip)
        ev = sim.step_frame(scale)
        rec['t'][i] = sim.t
        rec['d'][i] = sim.destroyed / totals
        rec['ev'].append(compact_events(ev))
        rec['creeper_done'][i] = bool(sim.creeper['done'])
        if i % 60 == 0:
            print(f'  [analyse] frame {i}/{n}  sim t={sim.t:.2f}  destroyed {np.round(100 * rec["d"][i], 1)}  '
                  f'({time.time() - t0:.0f}s)', flush=True)
    return rec


def story(rec, sched, shots):
    """Hearts of every giant per frame, the frames their elimination stamps land and the winner card."""
    d = rec['d']
    n = len(d)
    L = np.array([LETHAL[0], LETHAL[1], LETHAL[2], max(d[-1, 3], 1e-3) / 0.96])
    hp = np.clip(np.ceil(20.0 * (1.0 - d / L[None, :]) - 1e-9), 0, 20).astype(int)
    hp[rec['creeper_done'], T.CREEPER] = 0
    hp = np.minimum.accumulate(hp, axis=0)
    hp[:, T.WARDEN] = np.maximum(hp[:, T.WARDEN], 1)
    elim = {}
    for g, delay in ((T.CREEPER, 0.3), (T.ZOMBIE, 0.0), (T.STEVE, 0.25)):
        z = np.nonzero(hp[:, g] == 0)[0]
        if len(z) == 0:
            continue
        f0 = int(z[0])
        if g == T.ZOMBIE:
            # the stamp lands with him: when his toppled body hits the ground
            land = [i for i in range(f0, n) if rec['ev'][i].get('chunk_land', [0])[0] > 20000]
            f0 = land[0] if land else f0 + T.sec(1.0)
        elim[g] = min(n - 1, f0 + T.sec(delay))
    first = {}
    for i, (si, f, _, _) in enumerate(sched):
        first.setdefault(si, i)
    names = [s['name'] for s in shots]
    win_f = first[names.index('W_hero')] + T.sec(shots[names.index('W_hero')].get('winner', 0.6))
    intro_f = {s['intro']: first[k] for k, s in enumerate(shots) if 'intro' in s}
    return {'hp': hp, 'elim': elim, 'win': win_f, 'intro': intro_f, 'first': first, 'L': L}


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
        return float(np.clip(35.0 / (np.linalg.norm(np.asarray(e[1]) - eye) + 5.0), 0.25, 1.6))

    k = 0.0
    if 'creeper_boom' in ev:
        k += 7.0 * att('creeper_boom')
    if 'explode' in ev:
        k += np.log1p(ev['explode'][0]) * 0.9 * att('explode')
    if 'boom' in ev:
        k += 3.0
    if 'land_flesh' in ev:
        k += np.log1p(ev['land_flesh'][0]) * 0.35 * att('land_flesh')
    for nm in ('land_anvil', 'land_ground'):
        if nm in ev:
            k += np.log1p(ev[nm][0]) * 0.12 * att(nm)
    if 'chunk_land' in ev and ev['chunk_land'][0] > 3000:
        k += 3.0 * att('chunk_land')
    if 'impact' in ev:
        k += np.log1p(ev['impact'][0]) * 0.12
    return k


def fade_props(props, eye, lift, d0=1.3, d1=3.2):
    """Screen-door fade for anvils / TNT passing close to the lens."""
    if props is None or not len(props):
        return props
    c = props[:, :3].astype(np.float64) + np.array([0.0, 0.0, lift])
    dist = np.linalg.norm(c - eye, axis=1)
    u = np.clip((dist - d0) / (d1 - d0), 0.0, 1.0)
    fade = u * u * (3 - 2 * u) * props[:, 9]
    out = props[fade > 0].copy()
    out[:, 9] = fade[fade > 0]
    return out


def fade_arrows(arrows, eye, d0=0.8, d1=2.0):
    """Screen-door fade for arrows passing close to the lens."""
    if arrows is None or not len(arrows):
        return arrows
    p = arrows[:, :3].astype(np.float64)
    d = quat_rotate(arrows[:, 3:7].astype(np.float64), np.tile([1.0, 0.0, 0.0], (len(arrows), 1)))
    a = p - d * AR.TIP
    ab = d * (2 * AR.TIP)
    t = np.clip(np.sum((np.asarray(eye) - a) * ab, 1) / np.sum(ab * ab, 1), 0.0, 1.0)
    dist = np.linalg.norm(a + ab * t[:, None] - eye, axis=1)
    u = np.clip((dist - d0) / (d1 - d0), 0.0, 1.0)
    fade = u * u * (3 - 2 * u)
    out = arrows[fade > 0].copy()
    out[:, 9] = fade[fade > 0]
    return out


def nearest(points, eye, radius=30.0):
    if points is None or not len(points):
        return 0, 999.0, None
    d = np.linalg.norm(points - eye, axis=1)
    k = int(np.argmin(d))
    return int((d < radius).sum()), float(d[k]), points[k].tolist()


def swarm_info(sim, eye):
    """Arrows / anvils / TNT in flight near the camera (for fly-bys, whistles and fuse hiss)."""
    ar = sim.a_p[sim.a_state == A.A_FLY] if sim.a_n else None
    fl = sim.n_state == A.N_FALL
    an = np.c_[sim.n_xy[fl], sim.n_z[fl] + AN.HEIGHT * 0.5] if sim.n_n else None
    tf = (sim.t_state == A.T_FALL) | (sim.t_state == A.T_BLOWN)
    tn = sim.t_p[tf] if sim.t_n else None
    return {'arrows': nearest(ar, eye), 'anvils': nearest(an, eye), 'tnt': nearest(tn, eye, 20.0)}


class Heart:
    """The Warden's heart: his glow pulses with it; it races with the danger and calms once he has won."""

    def __init__(self, phase=0.35):
        self.phase = phase
        self.last_beat = -10.0

    def step(self, dt_sim, t_sim, danger):
        beat = False
        bpm = 56.0 + 70.0 * danger
        self.phase += bpm / 60.0 * dt_sim
        if self.phase >= 1.0:
            self.phase -= 1.0
            self.last_beat = t_sim
            beat = True
        a = t_sim - self.last_beat
        pulse = np.exp(-a / 0.13) + 0.55 * np.exp(-max(a - 0.2, 0.0) / 0.1) * (a > 0.2)
        return beat, 0.55 + 0.6 * min(pulse, 1.2)


def boom_charge(plan, t_sim):
    """Extra glow while the Warden charges and fires a sonic boom."""
    g = 0.0
    for b in plan.get('booms', ()):
        dt = t_sim - b[0]
        if -0.7 < dt < 0.0:
            g = max(g, ((dt + 0.7) / 0.7) ** 2 * 3.0)
        elif 0.0 <= dt < 0.5:
            g = max(g, 3.0 * (1.0 - dt / 0.5))
    return g


DANGER = {'L': 0.15, 'R1': 0.35, 'R2': 0.5, 'R3_sky': 0.75, 'R3': 1.0}


def danger_of(shot, t_loc):
    name = shot['name']
    if name == 'W_hero':
        return 0.95 - 0.6 * min(1.0, t_loc / 2.0)
    if name in DANGER:
        return DANGER[name]
    return DANGER.get(name.split('_')[0], 0.3)


def render_frame(r, sim, cam, hurt, glow, preview):
    vox, ranges, props, arrows, fx = sim.instances()
    eye = np.asarray(cam['eye'], float)
    props = {k: fade_props(v, eye, AN.HEIGHT * 0.5 if k == 'anvil' else 0.0) for k, v in props.items()}
    arrows = fade_arrows(arrows, eye)
    gl = []
    for gi, (a, b) in enumerate(ranges):
        white, swell = sim.creeper_fx() if gi == T.CREEPER else (0.0, 1.0)
        gl.append((a, b, float(hurt[gi]), float(white), float(swell), sim.giants[gi].center))
    if sim.ground.dirty:
        r.set_ground_region(*sim.ground.mesh())
    r.render(cam, voxels=vox, props=props, arrows=arrows, fx=fx, near_center=NEAR_CENTER, giants=gl, glow=glow,
             sculk=SCULK)
    img = r.finish()
    if preview:
        img = np.array(Image.fromarray(img).resize((W, H), Image.BILINEAR))
    return img


# ---------------------------------------------------------------------------------------------
# HUD state for a frame of the main edit
# ---------------------------------------------------------------------------------------------
def hud_state(i, sched, shots, st, hurt_t):
    si, f, _, _ = sched[i]
    sh = shots[si]
    hp = st['hp'][i]
    rows = []
    for g in range(4):
        f_in = st['intro'].get(g, 0)
        vis = float(np.clip((i - f_in + 1) / (0.25 * FPS), 0.0, 1.0))
        out = float(np.clip((i - st['elim'][g]) / (0.45 * FPS), 0.0, 1.0)) if g in st['elim'] else 0.0
        win = float(np.clip((i - st['win']) / (0.5 * FPS), 0.0, 1.0)) if g == T.WARDEN else 0.0
        dim = float(np.clip((i - st['win']) / (0.5 * FPS), 0.0, 1.0)) if g != T.WARDEN else 0.0
        hurt = float(np.clip(1.0 - (i - hurt_t[g]) / (0.4 * FPS), 0.0, 1.0)) if i >= hurt_t[g] else 0.0
        rows.append({'hp': int(hp[g]), 'vis': vis, 'out': out, 'win': win, 'dim': dim, 'hurt': hurt})
    s = {'board': rows, 'seed': i}
    t_loc = f / FPS
    if 'intro' in sh:
        s['intro'] = (sh['intro'], t_loc)
    for k2, s2 in enumerate(shots):
        if 'banner' in s2:
            tb = (i - st['first'][k2]) / FPS
            if 0 <= tb < 1.55:
                s['banner'] = (s2['banner'], TITLES[s2['banner']], tb)
    for g, fe in st['elim'].items():
        te = (i - fe) / FPS
        # a stamp makes way for the next round's banner or the winner card
        nxt = min([st['first'][k2] for k2, s2 in enumerate(shots) if 'banner' in s2 and st['first'][k2] > fe]
                  + [st['win']])
        end = min(1.65, (nxt - fe) / FPS - 0.05)
        if 0 <= te < end:
            s['elim'] = (g, te + max(0.0, 1.65 - end) * smooth01((te - (end - 0.3)) / 0.3))
    if i >= st['win']:
        s['winner'] = (T.WARDEN, (i - st['win']) / FPS)
    return s


def smooth01(u):
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3 - 2 * u)


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
    names = [s['name'] for s in shots]
    sched = T.schedule()
    plan = T.plan()
    only = set(args.shots.split(',')) if args.shots else None

    # ---- pass 1
    t0 = time.time()
    rec = analyse(sched)
    st = story(rec, sched, shots)
    print(f'[analyse] {len(sched)} frames in {time.time() - t0:.0f}s; lethal {np.round(st["L"], 3)}; '
          f'eliminations {st["elim"]}; winner card at {st["win"]}; final hp {st["hp"][-1].tolist()}', flush=True)
    with open(os.path.join(args.out, 'story.json'), 'w') as fh:
        json.dump({'hp': st['hp'].tolist(), 'elim': st['elim'], 'win': st['win'], 't': rec['t'].tolist(),
                   'd': rec['d'].tolist()}, fh)

    # ---- renderer / encoder
    if args.preview:
        r, _ = scene.make_renderer(width=W // 2, height=H // 2, ss=1.0, shadow_res=2048)
    else:
        r, _ = scene.make_renderer(width=W, height=H, ss=args.ss, shadow_res=4096)
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

    def cue_of(seg, name, f, cut, ev, eye, tgt, scale, sim, extra):
        c = {'seg': seg, 'shot': name, 'shot_frame': f, 'cut': cut, 'events': ev, 'cam': eye.tolist(),
             'tgt': np.asarray(tgt).tolist(), 'scale': float(scale), 't_sim': float(sim.t)}
        c.update(swarm_info(sim, eye))
        c.update(extra)
        return c

    # ---- the cold open: the same moments of round 3, shown first
    clips = T.cold_open()
    idx_of = {}
    for i, (si, f, _, _) in enumerate(sched):
        idx_of[(names[si], f)] = i
    sim = A.Arena(T.plan())
    r.set_ground_region(*sim.ground.mesh())
    heart = Heart(phase=0.6)
    shaker = Shaker(11)
    acc = 0.0
    j = 0
    for ci, clip in enumerate(clips):
        f0 = T.sec(clip['t0'])
        nf = T.sec(clip['dur'])
        i0 = idx_of[(clip['shot'], f0)]
        while j < i0:
            si, f, scale, skip = sched[j]
            if skip is not None:
                skip_to(sim, skip)
            sim.step_frame(scale)
            j += 1
        for k in range(nf):
            si, f, scale, skip = sched[j]
            if skip is not None:
                skip_to(sim, skip)
            ev = sim.step_frame(scale)
            j += 1
            eye, tgt, fov = T.cam_path(clip['keys'], k / FPS)
            acc = acc * 0.86 + shake_kick(compact_events(ev), eye)
            off = shaker.offset(k / FPS, 0.14 * np.tanh(acc / 3.0))
            beat, gm = heart.step(scale / FPS, sim.t, 1.0)
            glow = 5.0 * gm + boom_charge(plan, sim.t) * 2.0
            tag = f'cold{ci}_{k:03d}'
            if not wanted('cold', k):
                continue
            cam = {'eye': eye + off, 'target': tgt + off * 0.6, 'fov': fov, 'roll': float(off[0] * 0.8)}
            img = render_frame(r, sim, cam, np.zeros(4), glow, args.preview)
            flash = 0.0
            if ci == len(clips) - 1 and k >= nf - 3:
                flash = [0.25, 0.55, 0.85][k - (nf - 3)]
            img = hud.draw(img, {'flash': flash}) if flash else img
            emit(img, cue_of('cold', f'cold{ci}', k, k == 0, compact_events(ev), eye, tgt, scale, sim,
                             {'beat': bool(beat), 'glow': float(glow)}), tag)
    print(f'[render] cold open done ({time.time() - t_start:.0f}s)', flush=True)

    # ---- the main edit
    sim = A.Arena(T.plan())
    r.set_ground_region(*sim.ground.mesh())
    heart = Heart()
    shaker = Shaker(3)
    acc = 0.0
    hurt_t = np.full(4, -10 ** 6)
    last_flash = np.full(4, -10 ** 6)
    prev_si = None
    for i, (si, f, scale, skip) in enumerate(sched):
        sh = shots[si]
        if skip is not None:
            skip_to(sim, skip)
        ev = sim.step_frame(scale)
        cev = rec['ev'][i]
        t_loc = f / FPS
        eye, tgt, fov = T.cam_path(sh['keys'], t_loc)
        acc = acc * 0.86 + shake_kick(cev, eye)
        amp = sh.get('shake', 0.12) * np.tanh(acc / 3.0)
        off = shaker.offset(i / FPS, amp)
        hp = st['hp'][i]
        hp_prev = st['hp'][i - 1] if i else np.full(4, 20)
        hurt = np.zeros(4)
        for g in range(4):
            if hp[g] < hp_prev[g]:
                hurt_t[g] = i
                if i - last_flash[g] >= HURT_GAP:
                    last_flash[g] = i
            if 0 <= i - last_flash[g] < 7 and hp[g] > 0:
                hurt[g] = 0.75
        beat, gm = heart.step(scale / FPS, sim.t, danger_of(sh, t_loc))
        glow = 5.0 * gm + boom_charge(plan, sim.t) * 2.0
        hs = hud_state(i, sched, shots, st, hurt_t)
        flash = 0.0
        if i < 3:
            flash = [0.7, 0.35, 0.12][i]
        if 'creeper_boom' in cev:
            flash = max(flash, 0.0)
        extra = {'hp': hp.tolist(), 'hp_prev': hp_prev.tolist(), 'beat': bool(beat), 'glow': float(glow),
                 'hud': {}, 'danger': danger_of(sh, t_loc), 'creeper_white': float(sim.creeper_fx()[0]),
                 'hissing': bool(sim.creeper.get('hissing') and not sim.creeper['done'])}
        if f == 0 and 'banner' in sh:
            extra['hud']['banner'] = sh['banner']
        if f == 0 and 'intro' in sh:
            extra['hud']['intro'] = sh['intro']
        for g, fe in st['elim'].items():
            if fe == i:
                extra['hud']['elim'] = g
        if i == st['win']:
            extra['hud']['winner'] = T.WARDEN
        if wanted(sh['name'], f):
            cam = {'eye': eye + off, 'target': tgt + off * 0.6, 'fov': fov, 'roll': float(off[0] * 0.8)}
            img = render_frame(r, sim, cam, hurt, glow, args.preview)
            if flash:
                hs['flash'] = flash
            img = hud.draw(img, hs)
            emit(img, cue_of('main', sh['name'], f, si != prev_si, cev, eye, tgt, scale, sim, extra),
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
