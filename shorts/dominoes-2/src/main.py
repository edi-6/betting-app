"""Render "100,000 Dominoes in Minecraft... Part 2": the run punched off in first person, ten dominoes each bigger than
the last up to a giant that slams down and starts a four-colour race, red falling short at the finish, gold winning
a photo finish, the run stopping a block short of the field, the storm, lightning that sets the field off, the field
falling towards the cliff into the words LOOK BEHIND YOU, and the turn. Then the music and sound design and the
YouTube encode.

Usage:
  python main.py --out OUT_DIR [--preview] [--ss 2] [--every N] [--stills] [--from F --to F]
                 [--no-audio] [--no-hud] [--cues-only] [--encode-only]

Frames are rendered in one-second segments (OUT_DIR/seg), so a stopped render picks up where it left off.
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
from PIL import Image

import director as DR
import dominoes as DM
import effects as FX
import hand as HD
import herobrine as HB
import hud as HUDM
import layout as LY
import picture as PIC
import props as PR
import scene as SC
import storm as ST
import timeline as TL

W, H = 1080, 1920
FPS = TL.FPS
SEG = FPS                          # frames per segment file
TITLE = '100,000 Dominoes in Minecraft... Part 2'


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return 'ffmpeg'


def encode_youtube(video_in, wav_in, out, workdir, bitrate='16M'):
    """Two-pass H.264 following YouTube's recommended upload settings (High profile, closed GOP of half the frame
    rate, 2 B-frames, BT.709 tags) muxed with 384 kbps AAC at 48 kHz, moov atom up front. 16 Mbps for 1080p60 (above
    the recommendation: 100,000 dominoes and the rain are a lot of fine detail)."""
    gop = FPS // 2
    common = ['-c:v', 'libx264', '-preset', 'slower', '-profile:v', 'high', '-level', '4.2',
              '-b:v', bitrate, '-maxrate', '26M', '-bufsize', '32M', '-pix_fmt', 'yuv420p', '-r', str(FPS),
              '-x264-params', f'keyint={gop}:min-keyint={gop}:scenecut=0:open-gop=0:bframes=2',
              '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv']
    log = os.path.join(workdir, 'x264pass')
    subprocess.run([ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', video_in] + common +
                   ['-pass', '1', '-passlogfile', log, '-an', '-f', 'null', os.devnull], check=True)
    subprocess.run([ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', video_in, '-i', wav_in, '-map', '0:v:0', '-map',
                    '1:a:0'] + common +
                   ['-pass', '2', '-passlogfile', log, '-c:a', 'aac', '-b:a', '384k', '-ar', '48000', '-ac', '2',
                    '-movflags', '+faststart', '-shortest', '-metadata', f'title={TITLE}', out], check=True)


def _ss(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


class Show:
    """Everything that happens, as functions of simulation time."""

    def __init__(self):
        t = time.time()
        self.layout = LY.Layout(picture=PIC.picture)
        self.chains = DM.load(self.layout)
        kinds = PIC.kinds(self.layout.field['x'], self.layout.field['y'])
        self.dom = PR.Dominoes(self.layout, self.chains, kinds)
        C = self.chains
        self.t_done = self._first_time(lambda tt: self.dom.fallen(tt) >= LY.TOTAL, C.t_field0, C.t_end + 1.0)
        self.ev = TL.events(C, self.t_done)
        self.times = TL.schedule(self.ev)
        self.dir = DR.Director(self.layout, C, self.ev)
        self.cams = self.dir.cameras(self.times)
        self.hero = HB.Herobrine()
        self.arm = HD.Arm()
        self.sightings = self._sightings()
        print(f'[show] built in {time.time() - t:.1f}s', flush=True)

    def _first_time(self, cond, lo, hi):
        for _ in range(40):
            mid = (lo + hi) / 2
            if cond(mid):
                hi = mid
            else:
                lo = mid
        return hi

    def _sightings(self):
        """Where Herobrine stands, and when (he's there to be spotted on a second watch): beside the giant at the
        start, in the open by the race, on the edge of the cliff in the storm (where the last shot will stand), and
        right behind you at the end. (t_from, t_to, feet, yaw)."""
        ev = self.ev

        def facing(p, at):
            d = np.array(at, float) - np.array(p[:2], float)
            return float(np.arctan2(d[0], -d[1]))
        first = self.dir.F.first
        out = []
        p = (-11.5, LY.GIANT_Y + 1.5, 0.0)
        out.append((TL.T0, ev['grow'] + 1.2, p, facing(p, (first[0], first[1]))))
        p = (-13.8, -71.0, 0.0)
        out.append((ev['land'] + 0.3, ev['red_end'] - 0.6, p, facing(p, (0.6, -95.0))))
        p = (0.0, LY.CLIFF_Y + 1.0, float(LY.CLIFF_Z))
        out.append((ev['stop'] + 0.5, ev['strike'] + 0.05, p, facing(p, (2.7, 12.3))))
        out.append((ev['turn'] - 0.2, 1e9, TL.HEROBRINE_END, 0.0))
        return out

    def herobrines(self, t):
        ev = self.ev
        vox = []
        for (a, b, pos, yaw) in self.sightings:
            if a <= t < b:
                hy = hp = 0.0
                if t >= ev['turn'] - 0.2:
                    # face to face: his head tips a little as he looks at you
                    u = _ss(ev['turn'] + TL.TURN * 0.6, ev['black'], t)
                    hp = np.radians(-4.0) - np.radians(6.0) * u
                    hy = np.radians(5.0) * u
                vox.append(self.hero.instances(pos, yaw, head_yaw=hy, head_pitch=hp))
        return vox

    def glow(self, t, bolts_now):
        """Strength of the glowing materials: his eyes (brighter as it gets dark), the bolts, the words at the end."""
        ev = self.ev
        g = 3.0 + 4.0 * ST.darkness(t, ev)
        if t >= ev['done'] + 0.1:
            g = 0.8 + 1.5 * _ss(ev['done'] + 0.1, ev['done'] + 0.9, t)
        if t >= ev['turn']:
            g = 5.0 + 4.0 * _ss(ev['turn'] + TL.TURN + 0.3, ev['black'], t)
        if bolts_now:
            g = max(g, 14.0)
        return g


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
    """Per frame: dominoes set off (run / field), dominoes coming to rest, the growth's hits (by size), the camera,
    and the story's events."""
    C, ev = show.chains, show.ev
    lines = C.lines()
    t_run = np.concatenate([ln.t0 for ln in lines.values()])
    s_run = np.concatenate([ln.t_settle for ln in lines.values()])
    t_run, s_run = t_run[np.isfinite(t_run)], s_run[np.isfinite(s_run)]
    t_field = (C.column.t0[None, :] + C.col_start[:, None]).ravel()
    s_field = (C.column.t_settle[None, :] + C.col_start[:, None]).ravel()
    grow = [(float(C.main.t0[show.layout.n_open + i]), float(LY.GROW_S[i])) for i in range(LY.NGROW)]
    frames = []
    times = show.times
    prev = times[0] - 1.0 / FPS
    named = [('punch', 0.0), ('land', ev['land']), ('race', ev['race']), ('red_end', ev['red_end'] + 0.22),
             ('merge', ev['merge']), ('stop', C.tail.t_settle[-1] - 0.02), ('rumble', ev['stop'] + 0.8),
             ('done', ev['done']), ('turn', ev['turn']), ('face', ev['turn'] + TL.TURN * 0.8),
             ('black', ev['black'])]
    named += [('strike', ts) for ts, _ in ev['strikes']]
    for i, t in enumerate(times):
        c = {'t': float(t), 'dt': float(t - prev)}
        c['run'] = int(((t_run >= prev) & (t_run < t)).sum())
        c['field'] = int(((t_field >= prev) & (t_field < t)).sum())
        c['rest'] = int(((s_run >= prev) & (s_run < t)).sum()) + int(((s_field >= prev) & (s_field < t)).sum())
        cam = show.cams[i]
        c['cam'] = [float(v) for v in cam['eye']]
        c['tgt'] = [float(v) for v in cam['target']]
        c['grow'] = [s for (tg, s) in grow if prev <= tg < t]
        c['events'] = [name for name, te in named if prev <= te < t]
        c['dark'] = ST.darkness(t, ev)
        c['rain'] = ST.rain_amount(t, ev)
        frames.append(c)
        prev = t
    return frames


def glitch_out(img, u, t):
    """The end: the picture tears (slices jump sideways, the colours split) and cuts to black."""
    if u >= 0.45:
        return np.zeros_like(img)
    rng = np.random.default_rng(int(t * 997))
    out = img.astype(np.int16)
    h = out.shape[0]
    for _ in range(int(6 + 30 * u)):
        y = int(rng.integers(0, h))
        hh = int(rng.integers(4, 60))
        out[y:y + hh] = np.roll(out[y:y + hh], int(rng.integers(-120, 120)), axis=1)
    out[..., 0] = np.roll(out[..., 0], int(12 + 40 * u), axis=1)
    out[..., 2] = np.roll(out[..., 2], -int(12 + 40 * u), axis=1)
    out = out * (1.0 - u / 0.45)
    return np.clip(out, 0, 255).astype(np.uint8)


def render_frame(r, show, fx, i, t, preview):
    ev = show.ev
    cam = show.cams[i]
    D = show.dom
    props, P = D.instances(t)
    vox = []
    if t < 0.4:
        vox.append(show.arm.instances(cam['eye'], cam['target'], TL.SWING_AT, t, bob=t * 4.0))
    vox += show.herobrines(t)
    bvox, blights = ST.bolts(t, ev)
    vox += bvox
    bits, puffs, flashes, lights = fx.render_data()
    if len(bits):
        vox.append(bits)
    lights = lights + blights
    vox = np.concatenate(vox) if vox else None
    light_arr = np.array(lights[:16], np.float32) if lights else np.zeros((0, 8), np.float32)
    dark = ST.darkness(t, ev)
    fl = ST.flash(t, ev)
    m = (1.0 - 0.8 * dark) + 1.4 * fl
    bh = {'c': (0.0, 0.0, -9999.0), 'r': 0.0, 'sway': 0.0, 'sky_mul': (m * 0.84, m * 0.9, m * 1.0),
          'light_mul': (m * 0.9, m * 0.94, m * 1.0)}
    words = t >= ev['done'] + 0.1
    for kind in ('text', 'eye'):
        r.prop_kinds[kind]['sheen'] = 9.0 if words else 0.10
    r.fog = 0.0022 * (1.0 - 0.6 * dark)
    rain = ST.rain(cam['eye'], t, ST.rain_amount(t, ev))
    tgt = np.array(cam['target'], float)
    eye = np.array(cam['eye'], float)
    # the near shadow map follows what the camera looks at (on the ground, not too far off)
    d = tgt - eye
    near = eye + d * min(1.0, 30.0 / max(np.linalg.norm(d), 1e-6))
    near[2] = 0.0
    r.water_time = t
    r.render(cam, voxels=vox, props=props, fx={'puffs': puffs, 'flashes': flashes, 'lights': light_arr,
                                               'streaks': rain},
             near_center=near, glow=show.glow(t, len(bvox) > 0), sculk=(0.0, 0.0), bh=bh)
    img = r.finish()
    if preview:
        img = np.array(Image.fromarray(img).resize((W, H), Image.BILINEAR))
    return img


def step_effects(show, fx, r, t, prev, state):
    """Set off the effects whose moment has come, and advance them to time t."""
    ev = show.ev
    if not state.get('slam') and t >= ev['land']:
        state['slam'] = True
        xs = np.linspace(-6.5, 6.5, 7)
        ys = np.linspace(LY.GIANT_Y + 3.0, LY.RACE_Y0 - 1.0, 12)
        pts = np.array([(x, y, 0.4) for x in xs for y in ys])
        fx.slam(ev['land'], pts)
    for k, (ts, x) in enumerate(ev['strikes']):
        key = f'strike{k}'
        if not state.get(key) and t >= ts:
            state[key] = True
            fx.strike(ts, x, LY.FEEDER_Y, big=1.4 if k == 0 else 0.9)
            if k == 0:
                r.set_ground_region(*FX.ground_patch(None, scorch=(x + 0.3, LY.FEEDER_Y - 0.3, 1.3)))
    fx.step(t - prev, t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--preview', action='store_true')
    ap.add_argument('--ss', type=float, default=2.0, help='supersampling factor (2 = delivered quality)')
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--stills', action='store_true')
    ap.add_argument('--from', dest='f0', type=int, default=0)
    ap.add_argument('--to', dest='f1', type=int, default=10 ** 9)
    ap.add_argument('--frames', default='', help='only these frames (comma separated), as stills')
    ap.add_argument('--crf', type=int, default=12)
    ap.add_argument('--no-audio', action='store_true')
    ap.add_argument('--no-hud', action='store_true')
    ap.add_argument('--cues-only', action='store_true')
    ap.add_argument('--encode-only', action='store_true')
    args = ap.parse_args()
    pick = {int(v) for v in args.frames.split(',')} if args.frames else None
    if pick:
        args.stills = True
    os.makedirs(args.out, exist_ok=True)
    video_path = os.path.join(args.out, 'video_noaudio.mp4')
    if args.encode_only:
        import audio
        with open(os.path.join(args.out, 'cues.json')) as fh:
            d = json.load(fh)
        wav = os.path.join(args.out, 'audio.wav')
        audio.build(d, wav)
        encode_youtube(video_path, wav, os.path.join(args.out, 'final.mp4'), args.out)
        print('final:', os.path.join(args.out, 'final.mp4'), flush=True)
        return

    show = Show()
    n = len(show.times)
    print(f'[show] {n} frames ({n / FPS:.1f}s); events ' +
          ', '.join(f'{k} {v:.2f}' for k, v in show.ev.items() if not isinstance(v, list)), flush=True)
    cues = audio_cues(show)
    meta = {'fps': FPS, 'frames': cues, 'events': show.ev, 'winner': show.chains.winner}
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
    hud = HUDM.HUD(W, H, total=LY.TOTAL)
    fx = FX.Effects()
    r.set_ground_region(*FX.ground_patch(None))
    seg_dir = os.path.join(args.out, 'seg')
    os.makedirs(seg_dir, exist_ok=True)
    whole = args.f0 == 0 and args.f1 >= n and args.every == 1 and not args.stills
    state = {}
    prev = show.times[0]
    t_start = time.time()
    n_out = 0
    wr = None
    from writer import Writer
    for i, t in enumerate(show.times):
        step_effects(show, fx, r, t, prev, state)
        prev = t
        if not (args.f0 <= i < args.f1) or i % args.every or (pick is not None and i not in pick):
            continue
        seg = os.path.join(seg_dir, f's{i // SEG:03d}.mp4')
        if not args.stills and whole and os.path.exists(seg):
            continue
        img = render_frame(r, show, fx, i, t, args.preview)
        if not args.no_hud:
            ev = show.ev
            nf = show.dom.fallen(t)
            done = (t - ev['done']) if t >= ev['done'] else None
            glitch = float(np.clip((t - ev['turn'] - TL.TURN * 0.7) / 0.1, 0, 1)) if t >= ev['turn'] else 0.0
            img = hud.draw(img, nf, done, glitch, t, alpha=1.0)
            img = hud.race(img, t, ev, show.chains.winner)
        if t >= show.ev['black']:
            img = glitch_out(img, t - show.ev['black'], t)
        if args.stills:
            Image.fromarray(img).save(os.path.join(args.out, f'{i:04d}.jpg'), quality=90)
        else:
            if wr is None:
                wr = Writer(seg if whole else video_path, W, H, FPS, crf=args.crf)
            wr.write(img)
            if whole and (i % SEG == SEG - 1 or i == n - 1):
                wr.close()
                wr = None
        n_out += 1
        if n_out % 30 == 0:
            el = time.time() - t_start
            print(f'  frame {i}/{n} (t {t:.2f})  {el:.0f}s  ({el / n_out:.2f}s/frame)', flush=True)
    if wr is not None:
        wr.close()
    print(f'video frames: {n_out}, render time {time.time() - t_start:.0f}s', flush=True)
    if whole:
        segs = sorted(os.listdir(seg_dir))
        with open(os.path.join(seg_dir, 'list.txt'), 'w') as fh:
            fh.writelines(f"file '{os.path.abspath(os.path.join(seg_dir, s))}'\n" for s in segs if s.endswith('.mp4'))
        subprocess.run([ffmpeg_exe(), '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i',
                        os.path.join(seg_dir, 'list.txt'), '-c', 'copy', video_path], check=True)
        if not args.no_audio:
            import audio
            wav = os.path.join(args.out, 'audio.wav')
            audio.build(meta, wav)
            final = os.path.join(args.out, 'final.mp4')
            encode_youtube(video_path, wav, final, args.out)
            print('final:', final, flush=True)


if __name__ == '__main__':
    main()
