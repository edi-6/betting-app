"""Render "10,000 Marbles in Minecraft... Wait for the End": a marble machine in the plains. 10,000 marbles pour out of a
hopper, bounce down through a field of end rods and pile up behind glass in the colours of a hidden picture. A
golden ball jams the hopper, a goat rams the machine and frees it, the last marble drops on its own, and the
picture is a creeper. Which hisses. Then the music and sound design and the YouTube encode.

Usage:
  python main.py --out OUT_DIR [--preview] [--ss 2] [--every N] [--stills] [--from F --to F] [--frames a,b,c]
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

import fx as FX
import goat as GT
import hopper as HP
import hud as HUDM
import machine as M
import props as PR
import scene as SC
import sim as SIM
import timeline as TL
from herobrine import Herobrine
import layout as LY

W, H = 1080, 1920
FPS = TL.FPS
SEG = FPS
TITLE = '10,000 Marbles in Minecraft... Wait for the End'


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return 'ffmpeg'


def encode_youtube(video_in, wav_in, out, workdir, bitrate='16M'):
    """Two-pass H.264 following YouTube's recommended upload settings (High profile, closed GOP of half the frame
    rate, 2 B-frames, BT.709 tags) muxed with 384 kbps AAC at 48 kHz, moov atom up front. 16 Mbps for 1080p60:
    10,000 marbles are a lot of fine detail."""
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


GLASS = {'quad': [(M.MX - M.HALF, M.MY - M.SLOT, M.MZ), (M.MX + M.HALF, M.MY - M.SLOT, M.MZ),
                  (M.MX - M.HALF, M.MY - M.SLOT, M.MZ + M.BOARD_TOP),
                  (M.MX + M.HALF, M.MY - M.SLOT, M.MZ + M.BOARD_TOP)],
         'size': (2 * M.HALF, M.BOARD_TOP)}


class Show:
    """Everything that happens, as functions of story time (sim time, carrying on past the last marble)."""

    def __init__(self):
        t = time.time()
        self.sim = SIM.load()
        ev = self.sim['events']
        self.t_jam, self.t_save = float(ev[0]), float(ev[1])
        self.t_ram = self.t_save - 0.12
        self.t_land = float(ev[5])
        self.hopper = HP.Hopper(self.sim, self.t_ram)
        self.marbles = PR.Marbles(self.sim, self.hopper)
        self.tl = TL.Timeline(self.sim, self.marbles.hero_pos, self.marbles.level)
        self.goat = GT.Goat()
        self.goat_script = GT.Script(self.t_ram, M.MY - 6.0)
        self.hero = Herobrine()
        self.t_boom = self.t_land + TL.BOOM_AT
        self.n = self.tl.n_frames
        self.frames = [self.tl.at(i) for i in range(self.n)]
        print(f'[show] built in {time.time() - t:.1f}s: {self.n} frames ({self.tl.duration:.1f}s)', flush=True)

    def goat_voxels(self, ts):
        gs = self.goat_script
        if not gs.visible(ts):
            return None
        return self.goat.instances(gs.pose(ts))

    def herobrine(self, ts):
        """On the edge of the cliff in the last shot, for a second watch."""
        if ts < self.t_boom + 1.5:
            return None
        p = (-24.0, LY.CLIFF_Y + 1.0, float(LY.CLIFF_Z))
        return self.hero.instances(p, 0.12)

    def glass(self, ts):
        if ts >= self.t_boom:
            return None
        g = dict(GLASS)
        if ts >= self.t_land + TL.FUSE_AT:
            a = (ts - self.t_land - TL.FUSE_AT) / TL.FUSE_LEN
            g['crack'] = float(np.clip(a, 0, 1)) ** 1.5
            g['crack_c'] = ((self.marbles.creeper_c[0] + M.HALF) / (2 * M.HALF), 30.0 / M.BOARD_TOP)
            g['bulge'] = 0.3 * float(np.clip(a, 0, 1)) ** 2
        return g

    def captions(self, i):
        """[(text, alpha, style)] for frame i."""
        s, u, ts = self.frames[i]
        out = []
        v = (i + 0.5) / FPS
        st = dict(zip([x.name for x in self.tl.shots], self.tl.starts))

        def win(a, b, fade=0.18):
            return float(np.clip((v - a) / fade, 0, 1) * np.clip((b - v) / fade, 0, 1))
        o = st['open']
        out.append(('GUESS THE PICTURE...', win(o + 0.15, st['rise'] + 0.6), 'white', v - o - 0.15))
        g = st['gold']
        tr = g + (self.hopper.t_roll - self.tl.shots[3].ts(0)) * 1.0
        sj = st['stuck']
        tj = g + (self.t_jam - self.tl.shots[3].ts(0))
        out.append(('UH OH...', win(tr + 0.2, tj - 0.05, 0.12), 'yellow', v - tr - 0.2))
        out.append(("IT'S STUCK!", win(tj + 0.08, sj + 1.1), 'red', v - tj - 0.08))
        gg = st['goat_a']
        out.append(('...A GOAT?', win(gg + 0.4, gg + 1.6), 'white', v - gg - 0.4))
        b = st['burst']
        out.append(('THE GOAT SAVED IT!', win(b + 0.15, b + 1.45), 'yellow', v - b - 0.15))
        last = st['last']
        out.append(('THE LAST ONE...', win(st['count'] + 0.1, last + 2.6), 'white', v - st['count'] - 0.1))
        d = st['done']
        out.append(("IT'S A CREEPER!", win(d + 0.5, st['fuse'] + 0.1), 'green', v - d - 0.5))
        f = st['fuse']
        out.append(('RUN!!!', win(f + 0.45, st['boom'] + 0.05, 0.08), 'red', v - f - 0.45))
        return [(t_, a, sty, age) for (t_, a, sty, age) in out if a > 0.0]


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


def render_frame(r, show, fx, i, preview):
    cam = show.tl.camera(i)
    ts = cam['ts']
    sh = 0.0
    if 0.0 <= ts - show.t_ram < 0.5:
        sh = np.exp(-(ts - show.t_ram) * 8.0)
    marbles = show.marbles.instances(ts, shake=sh)
    vox = []
    g = show.goat_voxels(ts)
    if g is not None:
        vox.append(g)
    hb = show.herobrine(ts)
    if hb is not None:
        vox.append(hb)
    if ts >= show.t_boom:
        vox.append(show.marbles.blast.shards(ts - show.t_boom))
    bits, puffs, flashes, lights = fx.render_data()
    if len(bits):
        vox.append(bits)
    vox = np.concatenate(vox) if vox else None
    light_arr = np.array(lights[:16], np.float32) if lights else np.zeros((0, 8), np.float32)
    near = cam['near']
    if near is None:
        near = show.marbles.hero_pos(ts)
    nh = 34.0
    if cam['shot'] in ('open', 'fill_c', 'done', 'after', 'burst'):
        nh = 60.0
    r.near_half = nh
    r.render(cam, voxels=vox, props=None, fx={'puffs': puffs, 'flashes': flashes, 'lights': light_arr},
             near_center=tuple(near), marbles=marbles, glass=show.glass(ts))
    img = r.finish()
    if preview:
        img = np.array(Image.fromarray(img).resize((W, H), Image.BILINEAR))
    return img


def step_effects(show, fx, ts, prev, state):
    if not state.get('ram') and ts >= show.t_ram:
        state['ram'] = True
        gs = show.goat_script
        fx.ram(ts, (gs.x, M.MY - 6.05, 1.0))
    if not state.get('boom') and ts >= show.t_boom:
        state['boom'] = True
        fx.boom(ts, show.marbles.creeper_c)
    if state.get('boom'):
        a, b = prev - show.t_boom, ts - show.t_boom
        for (th, nh, pts) in show.marbles.blast.hits:
            if a <= th < b and pts is not None:
                fx.landings(pts)
    fx.step(max(ts - prev, 0.0), ts)


def audio_cues(show):
    """Per frame: the story's time and moments, the camera, marbles coming out, hitting pegs, landing."""
    frames = []
    prev = show.frames[0][2] - 1.0 / FPS
    m = show.marbles
    spawn = np.asarray(show.sim['spawn_t'])
    count = m.count_t
    hits = np.asarray(show.sim['hero_hits']) + float(show.sim['events'][4])
    named = {'jam': show.t_jam, 'ram': show.t_ram, 'charge': show.goat_script.k['charge'],
             'bleat': show.goat_script.k['look_cam'] + 0.05, 'step_in': show.goat_script.k['enter'],
             'goat_stop': show.goat_script.k['stop'], 'back': show.goat_script.k['back'],
             'pop': show.t_ram + 0.05, 'burst': show.t_save, 'hero': float(show.sim['events'][4]),
             'land': show.t_land, 'fuse': show.t_land + TL.FUSE_AT, 'boom': show.t_boom,
             'gold_roll': show.hopper.t_roll, 'gold_thud': show.t_ram + 4.4}
    for i in range(show.n):
        s, u, ts = show.frames[i]
        cam = show.tl.camera(i)
        c = {'t': float(ts), 'dt': float(ts - prev), 'shot': s.name, 'v': (i + 0.5) / FPS}
        c['out'] = int(((spawn >= prev) & (spawn < ts)).sum())
        c['in'] = int(((count >= prev) & (count < ts)).sum())
        c['hero_hits'] = int(((hits >= prev) & (hits < ts)).sum())
        c['events'] = [k for k, te in named.items() if prev <= te < ts]
        c['cam'] = [float(v) for v in cam['eye']]
        c['n'] = m.counted(ts)
        if ts >= show.t_boom:
            a, b = prev - show.t_boom, ts - show.t_boom
            c['blast_hits'] = int(sum(nh for (th, nh, _) in m.blast.hits if a <= th < b))
        frames.append(c)
        prev = ts
    return frames


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
    n = show.n
    cues = audio_cues(show)
    meta = {'fps': FPS, 'frames': cues, 'starts': [float(v) for v in show.tl.starts],
            'shots': [s.name for s in show.tl.shots]}
    with open(os.path.join(args.out, 'cues.json'), 'w') as fh:
        json.dump(jsonable(meta), fh)
    if args.cues_only:
        if not args.no_audio:
            import audio
            audio.build(meta, os.path.join(args.out, 'audio.wav'))
        return

    if args.preview:
        r, _ = SC.make_renderer(width=W // 2, height=H // 2, ss=1.0, shadow_res=2048)
    else:
        r, _ = SC.make_renderer(width=W, height=H, ss=args.ss, shadow_res=4096)
    hud = HUDM.HUD(W, H, total=M.N)
    fx = FX.Effects()
    seg_dir = os.path.join(args.out, 'seg')
    os.makedirs(seg_dir, exist_ok=True)
    whole = args.f0 == 0 and args.f1 >= n and args.every == 1 and not args.stills
    state = {}
    prev = show.frames[0][2]
    t_start = time.time()
    n_out = 0
    wr = None
    from writer import Writer
    styles = {'white': dict(color=(255, 255, 255)), 'yellow': dict(grad=((255, 240, 120), (255, 190, 30))),
              'red': dict(grad=((255, 120, 110), (230, 30, 30))), 'green': dict(grad=((170, 255, 120), (60, 200, 40)))}
    for i in range(n):
        s, u, ts = show.frames[i]
        step_effects(show, fx, ts, prev, state)
        prev = ts
        if not (args.f0 <= i < args.f1) or i % args.every or (pick is not None and i not in pick):
            continue
        seg = os.path.join(seg_dir, f's{i // SEG:03d}.mp4')
        if not args.stills and whole and os.path.exists(seg):
            continue
        img = render_frame(r, show, fx, i, args.preview)
        if not args.no_hud:
            nm = show.marbles.counted(ts)
            done = (ts - show.t_land) if ts >= show.t_land else None
            a = 1.0 - _ss(show.t_boom - 0.05, show.t_boom + 0.3, ts)
            if a > 0:
                img = hud.draw(img, nm, done, 0.0, ts, alpha=a)
            for (text, ca, sty, age) in show.captions(i):
                y = 1640 if s.name in ('done', 'fuse') else 1360
                img = hud.caption(img, text, ca, y=y, pop=float(np.clip(age, 0, 1)), **styles[sty])
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
            print(f'  frame {i}/{n} ({s.name}, ts {ts:.2f})  {el:.0f}s  ({el / n_out:.2f}s/frame)', flush=True)
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
