"""Render "10,000 Dominoes in Minecraft": the run through the plains (punched off in first person), the spiral, the
three-line race, the bridge, the stop a block short, the creeper that sets it going again, and the field that falls
into a picture of Herobrine, whose eyes light up. Then the music and sound design and the YouTube encode.

Usage:
  python main.py --out OUT_DIR [--preview] [--ss 1.5] [--every N] [--stills] [--from F --to F]
                 [--no-audio] [--no-hud] [--cues-only] [--encode-only]
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
from PIL import Image

import creeper as CR
import director as DR
import dominoes as DM
import effects as FX
import hand as HD
import hud as HUDM
import layout as LY
import picture as PIC
import props as PR
import scene as SC
import timeline as TL

W, H = 1080, 1920
FPS = TL.FPS
CREEPER_Y = 16.4
CREEPER_X = (6.6, 1.1)
BLAST_Z = 0.8
KNOCK = 4.2                        # the blast knocks over what stands this close
FLING = 2.8                        # ... and throws what lies this close
STRIKE = (-22.8, 50.0)             # where the lightning strikes (just off the field's west edge)


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return 'ffmpeg'


def encode_youtube(video_in, wav_in, out, workdir, bitrate='12M'):
    """Two-pass H.264 following YouTube's recommended upload settings (High profile, closed GOP of half the frame
    rate, 2 B-frames, BT.709 tags) muxed with 384 kbps AAC at 48 kHz, moov atom up front."""
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
                    '-movflags', '+faststart', '-shortest', '-metadata', 'title=10,000 Dominoes in Minecraft', out],
                   check=True)


def _ss(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


class Show:
    """Everything that happens, as functions of simulation time."""

    def __init__(self):
        img, eyes = PIC.picture(132, LY.NCOL)
        self.layout = LY.Layout(img)
        assert self.layout.nrow == 132
        self.chains = DM.Chains(self.layout, t_blast_after=TL.BLAST_AFTER)
        self.chains.field_lines((CREEPER_X[1], CREEPER_Y, KNOCK))
        self.dom = PR.Dominoes(self.layout, self.chains, eyes)
        self.ev = TL.events(self.chains)
        self.times = TL.schedule(self.ev)
        self.dir = DR.Director(self.layout, self.chains, self.ev)
        self.dir.creeper = lambda t: self.creeper_pose(t)[0]
        self.cams = self.dir.cameras(self.times)
        self.creeper = CR.Creeper()
        self.arm = HD.Arm()
        self.t_done = self._first_time(lambda t: self.dom.fallen(t) >= LY.TOTAL)

    def _first_time(self, cond, lo=None, hi=None):
        lo = self.ev['blast'] if lo is None else lo
        hi = self.ev['end'] + 1.0 if hi is None else hi
        for _ in range(40):
            mid = (lo + hi) / 2
            if cond(mid):
                hi = mid
            else:
                lo = mid
        return hi

    def creeper_pose(self, t):
        ev = self.ev
        w0, w1 = ev['walk'], ev['walk'] + TL.WALK
        u = np.clip((t - w0) / (w1 - w0), 0.0, 1.0)
        prog = u if u < 0.8 else 0.8 + (1 - (1 - (u - 0.8) / 0.2) ** 2) * 0.2      # slows to a stop
        x = CREEPER_X[0] + (CREEPER_X[1] - CREEPER_X[0]) * prog
        yaw_walk = -np.pi / 2
        yaw_look = np.arctan2(2.6 - x, -(12.4 - CREEPER_Y))                         # faces the camera
        v = _ss(ev['turn'], ev['turn'] + TL.TURN, t)
        yaw = yaw_walk * (1 - v) + yaw_look * v
        phase = 2 * np.pi * 1.8 * max(0.0, t - w0)
        amp = 0.55 * (1.0 - _ss(w1 - 0.35, w1, t)) * (t > w0)
        return np.array([x, CREEPER_Y, 0.0]), yaw, phase, amp

    def twist(self, t):
        """0..1 phases of the ending: eyes glowing, darkness, the lightning flash (and its bolt's time)."""
        e = self.ev['end']
        glow = _ss(e + 1.0, e + 2.0, t)
        dark = _ss(e + 1.0, e + 2.2, t)
        t_strike = e + 2.15
        flash = np.exp(-max(0.0, t - t_strike) / 0.09) * (t >= t_strike)
        flash += 0.6 * np.exp(-max(0.0, t - t_strike - 0.22) / 0.07) * (t >= t_strike + 0.22)
        return glow, dark, flash, t_strike


def bolt_voxels(t, t_strike, seed=13):
    """A lightning bolt: a jagged column of glowing voxels from the clouds to the ground, flickering out."""
    age = t - t_strike
    if age < 0 or age > 0.55:
        return np.zeros(0, CR.VOXEL_DTYPE)
    if 0.12 < age < 0.2:
        return np.zeros(0, CR.VOXEL_DTYPE)                         # the flicker between the two strokes
    rng = np.random.default_rng(seed)
    pts = []
    p = np.array([STRIKE[0] + 8.0, STRIKE[1] + 6.0, 95.0])
    target = np.array([STRIKE[0], STRIKE[1], 0.0])
    while p[2] > 0:
        step = (target - p) / max(1.0, p[2] / 2.0)
        step += rng.normal(0, 1.1, 3) * np.array([1, 1, 0.2])
        step[2] = -abs(step[2]) - 1.2
        n = int(np.ceil(np.linalg.norm(step) / 0.35))
        for k in range(n):
            pts.append(p + step * k / n)
        p = p + step
    pts = np.array(pts)
    inst = np.zeros(len(pts), CR.VOXEL_DTYPE)
    inst['pos'] = pts
    inst['quat'] = (0, 0, 0, 1)
    inst['scale'] = 0.45 * (1.0 - 0.6 * min(1.0, age / 0.55))
    for k in ('cx', 'cy', 'cz', 'inner'):
        inst[k][:, :3] = (215, 225, 255)
    inst['cx'][:, 3] = 0b111111
    inst['cy'][:, 3] = 0b111111
    inst['cz'][:, 3] = 255
    return inst


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    return x


def audio_cues(show):
    """Per frame: dominoes set off (run / field), dominoes coming to rest, where the run's front is (for the
    panning), and the story's events."""
    C, ev = show.chains, show.ev
    runs = [C.main, C.tail] + C.branches + list(C.feed.values())
    t_run = np.concatenate([ln.t0 for ln in runs])
    s_run = np.concatenate([ln.t_settle for ln in runs])
    t_run, s_run = t_run[np.isfinite(t_run)], s_run[np.isfinite(s_run)]
    t_field = (C.column.t0[None, :] + C.col_start[:, None]).ravel()
    s_field = (C.column.t_settle[None, :] + C.col_start[:, None]).ravel()
    frames = []
    times = show.times
    prev = times[0] - 1.0 / FPS
    walk0 = ev['walk']
    steps = walk0 + np.arange(0, TL.WALK - 0.2, 1.0 / (2 * 1.8))
    for i, t in enumerate(times):
        c = {'t': float(t), 'dt': float(t - prev)}
        c['run'] = int(((t_run >= prev) & (t_run < t)).sum())
        c['field'] = int(((t_field >= prev) & (t_field < t)).sum())
        c['rest'] = int(((s_run >= prev) & (s_run < t)).sum()) + int(((s_field >= prev) & (s_field < t)).sum())
        cam = show.cams[i]
        c['cam'] = [float(v) for v in cam['eye']]
        c['tgt'] = [float(v) for v in cam['target']]
        evs = []
        for name, te in (('punch', 0.0), ('split', ev['split']), ('merge', ev['merge']), ('lit', ev['lit']),
                         ('turn', ev['turn']), ('blast', ev['blast']), ('done', show.t_done),
                         ('glow', ev['end'] + 1.0), ('strike', show.twist(0.0)[3]),
                         ('stop', C.tail.t_settle[-1] - 0.02)):
            if prev <= te < t:
                evs.append(name)
        evs += ['step'] * int(((steps >= prev) & (steps < t)).sum())
        c['events'] = evs
        frames.append(c)
        prev = t
    return frames


def render_frame(r, show, fx, i, t, preview, pov_arm=True):
    cam = show.cams[i]
    D = show.dom
    fx.apply(D)
    props, P = D.instances(t)
    vox = []
    # the player's arm in the POV
    if t < 0.4 and pov_arm:
        vox.append(show.arm.instances(cam['eye'], cam['target'], TL.SWING_AT, t, bob=t * 4.0))
    # the creeper, from when it walks in until it goes off
    if show.ev['walk'] - 0.1 <= t < show.ev['blast']:
        pos, yaw, ph, amp = show.creeper_pose(t)
        swell, white = CR.fuse(show.ev['lit'], t, TL.FUSE)
        vox.append(show.creeper.instances(pos, yaw, ph, amp, swell=swell, white=white))
    bits, puffs, flashes, lights = fx.render_data()
    if len(bits):
        vox.append(bits)
    glow, dark, flash, t_strike = show.twist(t)
    bolt = bolt_voxels(t, t_strike)
    if len(bolt):
        vox.append(bolt)
        lights = lights + [[STRIKE[0], STRIKE[1], 20.0, 120.0 * flash, 90.0, 0.75, 0.82, 1.0]]
    vox = np.concatenate(vox) if vox else None
    light_arr = np.array(lights[:16], np.float32) if lights else np.zeros((0, 8), np.float32)
    r.water_time = t
    m = 1.0 - 0.72 * dark + 1.6 * flash
    bh = {'c': (0.0, 0.0, -9999.0), 'r': 0.0, 'sway': 0.0, 'sky_mul': (m * 0.9, m * 0.92, m), 'light_mul': (m, m, m)}
    # the eyes glow; so does the bolt (the same emission)
    g = 7.0 * glow + (14.0 if len(bolt) else 0.0)
    near = np.array(cam['target'], float)
    if t > show.ev['blast'] + 1.0:
        near = np.array([0.0, np.clip(near[1], 30.0, 70.0), 0.0])
    r.render(cam, voxels=vox, props=props, fx={'puffs': puffs, 'flashes': flashes, 'lights': light_arr},
             near_center=near, glow=g, sculk=(0.0, 0.0), bh=bh)
    img = r.finish()
    if preview:
        img = np.array(Image.fromarray(img).resize((W, H), Image.BILINEAR))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--preview', action='store_true')
    ap.add_argument('--ss', type=float, default=1.5, help='supersampling factor (1.5 = delivered quality)')
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--stills', action='store_true')
    ap.add_argument('--from', dest='f0', type=int, default=0)
    ap.add_argument('--to', dest='f1', type=int, default=10 ** 9)
    ap.add_argument('--crf', type=int, default=15)
    ap.add_argument('--no-audio', action='store_true')
    ap.add_argument('--no-hud', action='store_true')
    ap.add_argument('--cues-only', action='store_true')
    ap.add_argument('--encode-only', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.encode_only:
        import audio
        with open(os.path.join(args.out, 'cues.json')) as fh:
            d = json.load(fh)
        wav = os.path.join(args.out, 'audio.wav')
        audio.build(d, wav)
        encode_youtube(os.path.join(args.out, 'video_noaudio.mp4'), wav, os.path.join(args.out, 'final.mp4'),
                       args.out)
        return

    t0 = time.time()
    show = Show()
    n = len(show.times)
    print(f'[show] {n} frames ({n / FPS:.1f}s); events ' +
          ', '.join(f'{k} {v:.2f}' for k, v in show.ev.items()) + f', 10000 at {show.t_done:.2f}; '
          f'{time.time() - t0:.0f}s', flush=True)
    cues = audio_cues(show)
    meta = {'fps': FPS, 'frames': cues, 'events': show.ev, 't_done': show.t_done, 'winner': show.chains.winner}
    with open(os.path.join(args.out, 'cues.json'), 'w') as fh:
        json.dump(jsonable(meta), fh)
    if args.cues_only:
        if not args.no_audio:
            import audio
            audio.build(meta, os.path.join(args.out, 'audio.wav'))
        return

    if args.preview:
        r, _ = SC.make_renderer(show.layout, width=W // 2, height=H // 2, ss=1.0, shadow_res=2048)
    else:
        r, _ = SC.make_renderer(show.layout, width=W, height=H, ss=args.ss, shadow_res=4096)
    hud = HUDM.HUD(W, H)
    fx = FX.Effects()
    r.set_ground_region(*FX.ground_patch(None))
    video_path = os.path.join(args.out, 'video_noaudio.mp4')
    enc = None
    if not args.stills:
        cmd = [ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
               '-r', str(FPS), '-i', '-', '-vf', 'scale=out_color_matrix=bt709:out_range=tv',
               '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
               '-c:v', 'libx264', '-preset', 'slow', '-crf', str(args.crf), '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', video_path]
        enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    blasted = False
    prev = show.times[0]
    t_start = time.time()
    n_out = 0
    for i, t in enumerate(show.times):
        if not blasted and t >= show.ev['blast']:
            blasted = True
            _, P = show.dom.instances(show.ev['blast'])
            A = show.layout.all
            c = np.array([CREEPER_X[1], CREEPER_Y, BLAST_Z])
            pos = np.stack([A['x'] + show.dom.cos * P[:, 0], A['y'] + show.dom.sin * P[:, 0], P[:, 1]], -1)
            is_run = A['line'] < LY.FEED_E
            near = np.nonzero(is_run & (np.linalg.norm(pos - c, axis=1) < FLING))[0]
            fx.blast(show.ev['blast'], c, show.dom, P, near)
            r.set_ground_region(*FX.ground_patch(fx.crater))
        fx.step(t - prev, t)
        prev = t
        if not (args.f0 <= i < args.f1) or i % args.every:
            continue
        img = render_frame(r, show, fx, i, t, args.preview)
        if not args.no_hud:
            nf = show.dom.fallen(t)
            glow, dark, flash, t_strike = show.twist(t)
            done = (t - show.t_done) if t >= show.t_done else None
            glitch = float(np.clip((t - t_strike) / 0.1, 0, 1)) * (0.6 + 0.4 * np.sin(t * 40.0) ** 2) \
                if t >= t_strike else 0.0
            img = hud.draw(img, nf, done, glitch, t, alpha=1.0)
        if enc is not None:
            enc.stdin.write(np.ascontiguousarray(img).tobytes())
        else:
            Image.fromarray(img).save(os.path.join(args.out, f'{i:04d}.jpg'), quality=90)
        n_out += 1
        if n_out % 15 == 0:
            el = time.time() - t_start
            print(f'  frame {i}/{n}  {el:.0f}s  ({el / n_out:.2f}s/frame)', flush=True)
    if enc is not None:
        enc.stdin.close()
        enc.wait()
    print(f'video frames: {n_out}, render time {time.time() - t_start:.0f}s', flush=True)
    if not args.no_audio and enc is not None and args.every == 1 and args.f0 == 0:
        import audio
        wav = os.path.join(args.out, 'audio.wav')
        audio.build(meta, wav)
        final = os.path.join(args.out, 'final.mp4')
        encode_youtube(video_path, wav, final, args.out)
        print('final:', final)


if __name__ == '__main__':
    main()
