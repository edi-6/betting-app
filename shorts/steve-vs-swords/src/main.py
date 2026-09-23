"""Render the whole Short: hook montage + five rounds, HUD, then sound design and final mux.

Usage:
  python main.py --out OUT_DIR [--preview] [--ss 1.0] [--only r3] [--no-audio]
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
from PIL import Image

import scene
import sim as S
import timeline as T
from hud import HUD

W, H = 1080, 1920
FPS = T.FPS


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
                               '-movflags', '+faststart', '-shortest', '-metadata', 'title=Steve vs 10,000 Swords', out],
                   check=True)


# ---------------------------------------------------------------------------------------------
# pass 1: simulate a round without rendering, measure damage/events
# ---------------------------------------------------------------------------------------------
def analyse_round(rd):
    sim = S.RoundSim(rd['formation'], **rd['sim'])
    n = T.sec(rd['duration'])
    destroyed = np.zeros(n)
    events = []
    for f in range(n):
        ev = sim.step_frame()
        destroyed[f] = sim.destroyed
        compact = {}
        for k, (cnt, psum, pn) in ev.items():
            compact[k] = [int(cnt), (psum / pn).tolist() if pn else None]
        events.append(compact)
    total = sim.g.total
    return {'destroyed': destroyed, 'events': events, 'total': total}


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
        # make sure the giant is dead before the check mark lands, whatever the sim did
        f_dead = np.nonzero(steps >= rd['hp_loss'])[0]
        deadline = max(0, stamp_f - T.sec(0.6))
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
        imp = ev.get('impact', [0])[0]
        carve = ev.get('carve', [0])[0]
        shat = ev.get('shatter', [0])[0]
        kick = np.log1p(imp) * 0.35 + np.log1p(carve) * 0.05 + np.log1p(shat) * 0.25
        acc = acc * 0.86 + kick
        s[f] = acc
    return rd['shake'] * np.tanh(s / 3.0)


# ---------------------------------------------------------------------------------------------
# pass 2: render
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


def render_frame(r, cam, sim, preview):
    vox, sw = sim.instances()
    r.render(cam, voxels=vox, swords=sw)
    img = r.finish()
    if preview:
        img = np.array(Image.fromarray(img).resize((W, H), Image.BILINEAR))
    return img


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
        a = analyse_round(rd)
        a['hp'] = hp_curve(rd, a)
        a['shake'] = shake_curve(rd, a)
        analysis[rd['key']] = a
        print(f"[analyse] {rd['key']}: destroyed {int(a['destroyed'][-1])}/{a['total']}  "
              f"hp_end {a['hp'][-1]}  ({time.time() - t0:.0f}s)", flush=True)

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
    cues = []          # per output frame: dict of audio cues
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
            f0 = T.sec(hs['sim_start'])
            for _ in range(f0):
                sim.step_frame()
            nf = T.sec(hs['dur'])
            for f in range(nf):
                ev = sim.step_frame()
                t = f / FPS
                eye, tgt, fov = T.cam_path(hs['keys'], t)
                img = render_frame(r, {'eye': eye, 'target': tgt, 'fov': fov}, sim, args.preview)
                img = hud.draw(img, {'hp': 20})
                cue = {'seg': 'hook', 'shot': hi, 'shot_frame': f, 'round': rd['key'], 'sim_frame': f0 + f,
                       'events': analysis[rd['key']]['events'][f0 + f], 'cam': eye.tolist()}
                emit(img, cue)

    # ---- rounds
    for rd in rounds:
        if only is not None and rd['key'] not in only:
            continue
        a = analysis[rd['key']]
        sim = S.RoundSim(rd['formation'], **rd['sim'])
        shaker = Shaker(sum(map(ord, rd['key'])))       # deterministic across runs
        n = T.sec(rd['duration'])
        stamp_f = T.sec(rd['stamp_at'])
        flash_start = -100
        for f in range(n):
            sim.step_frame()
            t = f / FPS
            si = max(k for k, s in enumerate(rd['shots']) if t >= s['start'] - 1e-9)
            shot = rd['shots'][si]
            eye, tgt, fov = T.cam_path(shot['keys'], t)
            amp = a['shake'][f]
            off = shaker.offset(t, amp)
            eye = eye + off
            tgt = tgt + off * 0.6
            roll = float(off[0] * 0.8)
            img = render_frame(r, {'eye': eye, 'target': tgt, 'fov': fov, 'roll': roll}, sim, args.preview)
            hp = int(a['hp'][f])
            if f > 0 and hp < a['hp'][f - 1]:
                flash_start = f
            flash = (f - flash_start) < 12 and ((f - flash_start) // 3) % 2 == 0
            st = {'hp': hp, 'flash': flash, 'jiggle_seed': out_frame}
            lab_in, lab_out = 0.0, 1.75
            if lab_in <= t < lab_out:
                st['label'] = rd['label']
                st['label_t'] = min(1.0, (t - lab_in) / 0.3)
                st['label_alpha'] = min(1.0, max(0.0, (lab_out - t) / 0.2))
            if f >= stamp_f:
                st['stamp'] = rd['stamp']
                st['stamp_t'] = min(1.0, (f - stamp_f) / (0.3 * FPS))
            img = hud.draw(img, st)
            cue = {'seg': rd['key'], 'shot': si, 'round_frame': f, 'events': a['events'][f], 'cam': eye.tolist(),
                   'hp': hp, 'hp_prev': int(a['hp'][f - 1]) if f else 20,
                   'label': f == T.sec(lab_in), 'stamp': rd['stamp'] if f == stamp_f else None,
                   'cut': f == 0 or si != max(k for k, s in enumerate(rd['shots']) if (f - 1) / FPS >= s['start'] - 1e-9),
                   'launch': rd['formation']['launch'].min(), 'n_swords': len(rd['formation']['pos'])}
            emit(img, cue)
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
