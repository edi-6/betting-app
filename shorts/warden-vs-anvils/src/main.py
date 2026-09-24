"""Render "Warden vs 10,000 Anvils": hook montage + five rounds, HUD, then sound design and final mux.

Usage:
  python main.py --out OUT_DIR [--preview] [--ss 1.5] [--only r3] [--no-audio] [--encode-only]
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
from PIL import Image

import anvil as AN
import scene
import sim as S
import timeline as T
from hud import HUD

W, H = 1080, 1920
FPS = T.FPS
UP = np.array([0.0, 0.0, 1.0])


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
                               '-movflags', '+faststart', '-shortest', '-metadata', 'title=Warden vs 10,000 Anvils', out],
                   check=True)


def compact_events(ev):
    return {k: [int(cnt), (psum / pn).tolist() if pn else None] for k, (cnt, psum, pn) in ev.items()}


def make_sim(rd):
    sim = S.RoundSim(rd['formation'], **rd['sim'])
    for _ in range(T.sec(rd.get('preroll', 0.0))):
        sim.step_frame(1.0)
    return sim


# ---------------------------------------------------------------------------------------------
# pass 1: simulate a round without rendering (same time-scale schedule), measure damage/events
# ---------------------------------------------------------------------------------------------
def analyse_round(rd):
    sim = make_sim(rd)
    n = T.sec(rd['duration'])
    scales = T.frame_scales(rd.get('scale'), n)
    destroyed = np.zeros(n)
    events = []
    for f in range(n):
        ev = sim.step_frame(scales[f])
        destroyed[f] = sim.destroyed
        events.append(compact_events(ev))
    return {'destroyed': destroyed, 'events': events, 'total': sim.g.total, 'scales': scales}


def hp_curve(rd, an):
    d = an['destroyed']
    n = len(d)
    stamp_f = min(n - 1, T.sec(rd['stamp_at']))
    if rd.get('death_fraction'):
        ref = rd['death_fraction'] * an['total']
    else:
        ref = max(d[stamp_f], 1.0)
    loss = rd['hp_loss'] * np.clip(d / ref, 0, 1)
    steps = np.ceil(loss - 1e-6).astype(int)
    steps = np.minimum(steps, rd['hp_loss'])
    if rd.get('death_fraction'):
        # make sure he is dead before the check mark lands, whatever the sim did
        f_dead = np.nonzero(steps >= rd['hp_loss'])[0]
        deadline = max(0, stamp_f - T.sec(0.9))
        if len(f_dead) == 0 or f_dead[0] > deadline:
            first = np.nonzero(steps > 0)[0]
            f0 = first[0] if len(first) else deadline // 2
            ramp = np.arange(n)
            forced = np.ceil(rd['hp_loss'] * np.clip((ramp - f0) / max(1, deadline - f0), 0, 1)).astype(int)
            steps = np.maximum(steps, forced)
    steps[stamp_f:] = steps[stamp_f]
    return 20 - steps


def shake_curve(rd, an):
    n = len(an['destroyed'])
    s = np.zeros(n)
    acc = 0.0
    for f in range(n):
        ev = an['events'][f]
        flesh = ev.get('land_flesh', [0])[0]
        metal = ev.get('land_anvil', [0])[0] + ev.get('land_ground', [0])[0]
        big = ev.get('chunk_land', [0])[0] > 0
        boom = ev.get('boom', [0])[0] > 0
        kick = np.log1p(flesh) * 0.45 + np.log1p(metal) * 0.12 + 1.5 * big + 4.0 * boom
        acc = acc * 0.86 + kick
        s[f] = acc
    return rd['shake'] * np.tanh(s / 3.0)


# ---------------------------------------------------------------------------------------------
# cameras
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


class AnvilCam:
    """Anvil-cam: rides along with a falling anvil, a little in front of and above it, keeping both the anvil and
    the spot it will land on in frame. With `stop` it brakes smoothly and halts `stop` blocks above the landing
    spot, so the hit is seen rather than flown into; after the hit it eases round to a reveal pose."""

    def __init__(self, sim, spec):
        self.sim = sim
        self.i = spec['anvil']
        self.back = spec.get('back', 3.0)
        self.off = np.array([*spec.get('off', (3.0, -16.0)), 0.0])
        self.fov = spec.get('fov', 66.0)
        self.stop = spec.get('stop', 0.0)
        self.soft = spec.get('soft', 2.0)
        self.reveal = spec.get('reveal_eye')
        self.reveal_tgt = spec.get('reveal_tgt')
        self.reveal_fov = spec.get('reveal_fov', self.fov)
        self.reveal_time = spec.get('reveal_time', 1.5)
        cx, cy = int(sim.ix[self.i]), int(sim.iy[self.i])
        self.land = np.array([sim.xy[self.i, 0], sim.xy[self.i, 1], float(sim.support(cx, cy))])
        self.hit = None

    def _pose(self):
        s = self.sim
        p = np.array([s.xy[self.i, 0], s.xy[self.i, 1], s.z[self.i] + AN.HEIGHT * 0.5])
        eye = p + UP * self.back + self.off
        if self.stop > 0:
            x = eye[2] - self.land[2]
            xs = self.stop + self.soft * np.log1p(np.exp(np.clip((x - self.stop) / self.soft, -30, 30)))
            eye[2] = self.land[2] + xs
        a = p - eye
        b = self.land - eye
        d = a / np.linalg.norm(a) + b / np.linalg.norm(b)          # halfway between the anvil and its target
        return eye, eye + d / np.linalg.norm(d) * 10.0

    def __call__(self, t):
        s = self.sim
        if s.state[self.i] in (S.WAIT, S.FALL):
            eye, tgt = self._pose()
            return eye, tgt, self.fov, UP, True
        if self.hit is None:
            eye, tgt = self._pose()
            self.hit = (eye, tgt, t)
        eye0, tgt0, t0 = self.hit
        if self.reveal is None:
            return eye0, tgt0, self.fov, UP, False
        u = np.clip((t - t0) / self.reveal_time, 0.0, 1.0)
        u = u * u * (3 - 2 * u)
        eye = eye0 * (1 - u) + np.asarray(self.reveal) * u
        tgt = tgt0 * (1 - u) + np.asarray(self.reveal_tgt) * u
        return eye, tgt, self.fov * (1 - u) + self.reveal_fov * u, UP, False


def swarm_info(sim, eye, exclude=None):
    """Falling anvils within 30 blocks of the camera, the closest one's distance and position (whooshes)."""
    f = sim.state == S.FALL
    if exclude is not None:
        f[exclude] = False
    if not f.any():
        return 0, 999.0, None
    p = np.c_[sim.xy[f], sim.z[f] + AN.HEIGHT * 0.5]
    d = np.linalg.norm(p - eye, axis=1)
    k = int(np.argmin(d))
    return int((d < 30.0).sum()), float(d[k]), p[k].tolist()


def fade_near(props, eye, d0=1.6, d1=3.5, keep=None):
    """Screen-door fade for anvils passing close to the lens (between d1 and d0 from their centre)."""
    if not len(props):
        return props
    c = props[:, :3].astype(np.float64) + np.array([0.0, 0.0, AN.HEIGHT * 0.5])
    dist = np.linalg.norm(c - eye, axis=1)
    u = np.clip((dist - d0) / (d1 - d0), 0.0, 1.0)
    fade = u * u * (3 - 2 * u)
    if keep is not None:
        fade[np.linalg.norm(props[:, :3] - keep, axis=1) < 1e-3] = 1.0
    out = props[fade > 0].copy()
    out[:, 9] = fade[fade > 0]
    return out


# ---------------------------------------------------------------------------------------------
# the Warden's heart: his glow pulses with it, it races as the danger grows and stops when he dies
# ---------------------------------------------------------------------------------------------
class Heart:
    def __init__(self, phase=0.35):
        self.phase = phase
        self.last_beat = -10.0
        self.stopped = False
        self.dead_t = None

    def step(self, dt_sim, t_sim, danger, dead):
        """Advance by dt_sim seconds of simulation time (the heart slows down in slow motion too).
        Returns (beat happened this frame, glow multiplier). self.dim goes from 1 to 0.3 once he's dead."""
        beat = False
        if dead and self.dead_t is None:
            self.dead_t = t_sim
        bpm = 58.0 + 62.0 * danger
        if self.dead_t is not None:
            bpm = max(20.0, 58.0 - 50.0 * (t_sim - self.dead_t))       # slows ... and stops
            if t_sim - self.dead_t > 0.9:
                self.stopped = True
        if not self.stopped:
            self.phase += bpm / 60.0 * dt_sim
            if self.phase >= 1.0:
                self.phase -= 1.0
                self.last_beat = t_sim
                beat = True
        a = t_sim - self.last_beat
        pulse = np.exp(-a / 0.13) + 0.55 * np.exp(-max(a - 0.2, 0.0) / 0.1) * (a > 0.2)
        g = 0.55 + 0.6 * min(pulse, 1.2)
        self.dim = 1.0
        if self.dead_t is not None:
            k = min(1.0, (t_sim - self.dead_t) / 1.6)
            g *= 1.0 - k                                                   # the light goes out
            self.dim = 1.0 - 0.7 * k
        return beat, g


def render_frame(r, cam, sim, preview, hurt=0.0, follow=None, glow=5.0, glow_dim=1.0):
    vox, anvils, fx = sim.instances()
    if follow is not None:
        keep = np.array([sim.xy[follow.i, 0], sim.xy[follow.i, 1], sim.z[follow.i]])
        anvils = fade_near(anvils, np.asarray(cam['eye'], float), 2.5, 7.0, keep=keep)
    else:
        anvils = fade_near(anvils, np.asarray(cam['eye'], float))
    r.render(cam, voxels=vox, props=anvils, fx=fx, hurt=(hurt, sim.n_static), glow=glow, glow_dim=glow_dim)
    img = r.finish()
    if preview:
        img = np.array(Image.fromarray(img).resize((W, H), Image.BILINEAR))
    return img


HURT_GAP = 18


def hurt_amount(frames_since_hit):
    return 1.0 if 0 <= frames_since_hit < 8 else 0.0


def boom_charge(rd, t_sim):
    """Extra glow while the Warden charges and fires his sonic boom."""
    g = 0.0
    for b in rd['sim'].get('booms', ()):
        dt = t_sim - b[0]
        if -0.7 < dt < 0.0:
            g = max(g, ((dt + 0.7) / 0.7) ** 2 * 3.0)
        elif 0.0 <= dt < 0.5:
            g = max(g, 3.0 * (1.0 - dt / 0.5))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--preview', action='store_true')
    ap.add_argument('--ss', type=float, default=1.5, help='supersampling factor (1.5 = delivered quality)')
    ap.add_argument('--only', default=None, help='comma separated list of segments: hook,r1..r5')
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

    rounds = T.round_defs()
    rmap = {rd['key']: rd for rd in rounds}
    only = set(args.only.split(',')) if args.only else None

    # ---- pass 1
    t0 = time.time()
    analysis = {}
    for rd in rounds:
        if only is not None and rd['key'] not in only:
            continue
        a = analyse_round(rd)
        a['hp'] = hp_curve(rd, a)
        a['shake'] = shake_curve(rd, a)
        analysis[rd['key']] = a
        print(f"[analyse] {rd['key']}: destroyed {int(a['destroyed'][-1])}/{a['total']}  hp_end {a['hp'][-1]}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    # ---- renderer
    if args.preview:
        r, _ = scene.make_renderer(width=W // 2, height=H // 2, ss=1.0, shadow_res=2048)
    else:
        r, _ = scene.make_renderer(width=W, height=H, ss=args.ss, shadow_res=4096)
    hud = HUD(W, H)
    video_path = os.path.join(args.out, 'video_noaudio.mp4')
    cmd = [ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
           '-r', str(FPS), '-i', '-', '-vf', 'scale=out_color_matrix=bt709:out_range=tv',
           '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
           '-c:v', 'libx264', '-preset', 'slow', '-crf', str(args.crf),
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', video_path]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    cues = []
    out_frame = 0
    t_start = time.time()

    def emit(img, cue):
        nonlocal out_frame
        enc.stdin.write(np.ascontiguousarray(img).tobytes())
        cues.append(cue)
        out_frame += 1
        if out_frame % 15 == 0:
            el = time.time() - t_start
            print(f'  frame {out_frame}  {el:.0f}s  ({el / out_frame:.2f}s/frame)', flush=True)
        if out_frame % 30 == 1:
            Image.fromarray(img).save(os.path.join(args.out, f'thumb_{out_frame:04d}.jpg'), quality=88)

    # ---- hook
    if only is None or 'hook' in only:
        for hi, hs in enumerate(T.hook_shots()):
            rd = rmap[hs['round']]
            sim = S.RoundSim(rd['formation'], **rd['sim'])
            for _ in range(T.sec(hs['sim_start'])):
                sim.step_frame(1.0)
            heart = Heart(phase=0.6)
            nf = T.sec(hs['dur'])
            scales = T.frame_scales(hs.get('scale'), nf)
            follow = AnvilCam(sim, hs['anvilcam']) if 'anvilcam' in hs else None
            for f in range(nf):
                ev = sim.step_frame(scales[f])
                t = f / FPS
                if follow is not None:
                    eye, tgt, fov, up, riding = follow(t)
                else:
                    eye, tgt, fov = T.cam_path(hs['keys'], t)
                    up, riding = UP, False
                beat, gm = heart.step(scales[f] / FPS, sim.t, 0.8, False)
                glow = 5.0 * gm + boom_charge(rd, sim.t) * 2.0
                img = render_frame(r, {'eye': eye, 'target': tgt, 'fov': fov, 'up': tuple(up)}, sim, args.preview,
                                   follow=follow, glow=glow)
                img = hud.draw(img, {'hp': 20})
                n_near, dmin, cpos = swarm_info(sim, eye, None if follow is None else follow.i)
                emit(img, {'seg': 'hook', 'shot': hi, 'shot_frame': f, 'round': rd['key'],
                           'events': compact_events(ev), 'cam': eye.tolist(), 'tgt': np.asarray(tgt).tolist(),
                           'scale': float(scales[f]), 'swarm': n_near, 'swarm_dmin': dmin, 'swarm_pos': cpos,
                           'riding': bool(riding), 'cut': f == 0, 'beat': bool(beat), 'glow': float(glow)})

    # ---- rounds
    for rd in rounds:
        if only is not None and rd['key'] not in only:
            continue
        a = analysis[rd['key']]
        sim = make_sim(rd)
        shaker = Shaker(sum(map(ord, rd['key'])))
        heart = Heart()
        n = T.sec(rd['duration'])
        stamp_f = T.sec(rd['stamp_at'])
        flash_start = -100
        follows = {}
        prev_si = None
        for f in range(n):
            sim.step_frame(a['scales'][f])
            t = f / FPS
            si = max(k for k, s in enumerate(rd['shots']) if t >= s['start'] - 1e-9)
            shot = rd['shots'][si]
            riding = False
            follow = None
            up = UP
            if 'anvilcam' in shot:
                if si not in follows:
                    follows[si] = AnvilCam(sim, shot['anvilcam'])
                follow = follows[si]
                eye, tgt, fov, up, riding = follow(t - shot['start'])
            else:
                eye, tgt, fov = T.cam_path(shot['keys'], t)
            amp = a['shake'][f]
            off = shaker.offset(t, amp)
            eye = eye + off
            tgt = tgt + off * 0.6
            roll = float(off[0] * 0.8)
            hp = int(a['hp'][f])
            if f > 0 and hp < a['hp'][f - 1] and f - flash_start >= HURT_GAP:
                flash_start = f
            hurt = hurt_amount(f - flash_start)
            danger = float(np.clip(swarm_info(sim, np.array([0.0, 0.0, 20.0]))[0] / 40.0 + (20 - hp) / 25.0, 0, 1))
            beat, gm = heart.step(a['scales'][f] / FPS, sim.t, danger, hp <= 0)
            glow = 5.0 * gm + boom_charge(rd, sim.t) * 2.0
            img = render_frame(r, {'eye': eye, 'target': tgt, 'fov': fov, 'roll': roll, 'up': tuple(up)}, sim,
                               args.preview, hurt=hurt, follow=follow, glow=glow, glow_dim=heart.dim)
            st = {'hp': hp, 'flash': (f - flash_start) < 12 and ((f - flash_start) // 3) % 2 == 0,
                  'jiggle_seed': out_frame}
            lab_in, lab_out = 0.0, 1.75
            if lab_in <= t < lab_out:
                st['label'] = rd['label']
                st['label_t'] = min(1.0, (t - lab_in) / 0.3)
                st['label_alpha'] = min(1.0, max(0.0, (lab_out - t) / 0.2))
            if f >= stamp_f:
                st['stamp'] = rd['stamp']
                st['stamp_t'] = min(1.0, (f - stamp_f) / (0.3 * FPS))
            img = hud.draw(img, st)
            n_near, dmin, cpos = swarm_info(sim, eye, None if follow is None else follow.i)
            emit(img, {'seg': rd['key'], 'shot': si, 'round_frame': f, 'events': a['events'][f], 'cam': eye.tolist(),
                       'tgt': np.asarray(tgt).tolist(), 'hp': hp, 'hp_prev': int(a['hp'][f - 1]) if f else 20,
                       'label': f == T.sec(lab_in), 'stamp': rd['stamp'] if f == stamp_f else None,
                       'cut': f == 0 or si != prev_si, 'scale': float(a['scales'][f]),
                       'swarm': n_near, 'swarm_dmin': dmin, 'swarm_pos': cpos, 'riding': bool(riding),
                       'hurt': hurt, 'beat': bool(beat), 'glow': float(glow), 'dead': hp <= 0})
            prev_si = si
        print(f"[render] {rd['key']} done", flush=True)

    enc.stdin.close()
    enc.wait()
    with open(os.path.join(args.out, 'cues.json'), 'w') as fh:
        json.dump({'fps': FPS, 'frames': cues}, fh)
    print(f'video frames: {out_frame}, render time {time.time() - t_start:.0f}s', flush=True)

    if not args.no_audio:
        import audio
        wav = os.path.join(args.out, 'audio.wav')
        audio.build(cues, FPS, wav)
        final = os.path.join(args.out, 'final.mp4')
        encode_youtube(video_path, wav, final, args.out)
        print('final:', final)


if __name__ == '__main__':
    main()
