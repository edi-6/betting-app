"""Render "1 vs 10 vs 100 vs 1,000 Block Tsunami vs Minecraft Village": four tsunamis, each ten times taller than the
last, against the same seaside village. The first one wets a fisherman's feet; the second floods the village; the
third wipes it off the map. The fourth is a kilometre-high wall that blots out the sun... and you put down a sponge.

Usage:
  python main.py --out OUT_DIR [--preview] [--ss 2] [--every N] [--from F --to F] [--frames a,b,c] [--sheet]
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

import blocks as BL
import edit as E
import hand as HD
import hud as HUDM
import playback as PB
import scene as SC
import village as VL
import wall as WL
import world as WD

W, H = 1080, 1920
FPS = E.FPS
SEG = FPS
TITLE = '1 vs 10 vs 100 vs 1,000 Block Tsunami vs Minecraft Village'
TOTAL_BLOCKS = 8022
# when each round's verdict lands: (shot, how far through it)
VERDICT = {1: ('r1', 0.86), 2: ('r2_flood', 0.62), 3: ('r3_gone', 0.42), 4: ('r4_hmm', 0.62)}
OK = {1: True, 2: True, 3: False, 4: True}


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return 'ffmpeg'


def encode_youtube(video_in, wav_in, out, workdir, bitrate='16M'):
    """Two-pass H.264 following YouTube's recommended upload settings (High profile, closed GOP of half the frame
    rate, 2 B-frames, BT.709 tags) muxed with 384 kbps AAC at 48 kHz, moov atom up front."""
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


def verdict_time(n):
    name, u = VERDICT[n]
    s = next(s for s in E.SHOTS if s.name == name)
    return s.start + u * s.dur


def _qz(ang):
    return np.array([0.0, 0.0, np.sin(ang / 2), np.cos(ang / 2)])


class Show:
    """Everything on screen at a moment of the video."""

    def __init__(self, r, wd):
        self.r = r
        t = time.time()
        self.vil = VL.build(wd['H'])
        self.pbs = {n: PB.Playback(n, self.vil) for n in (1, 2, 3)}
        self.wall = WL.Wall(wd['H'], sponge=(E.SPONGE[0], E.SPONGE[1], 1.0))
        self.arm = HD.Arm()
        self.intact = WD.mesh_blocks(self.vil.K, VL.ORIGIN, tint=self.vil.tint)
        self.base = dict(sky_amb=r.sky_amb.copy(), gnd_amb=r.gnd_amb.copy(), fog=float(r.fog))
        self.grid_of = None
        self.mesh_key = None
        self.v_sponge = BL.debris_variant(BL.KIND_ID['sponge'])
        self.v_wet = BL.debris_variant(BL.KIND_ID['wet_sponge'])
        rng = np.random.default_rng(21)
        self.st_ang = rng.uniform(np.radians(12), np.radians(168), 260)
        self.st_r = rng.uniform(6.0, 34.0, 260)
        self.st_t0 = rng.uniform(0.0, 1.0, 260)
        self.st_spin = rng.uniform(0.6, 1.6, 260) * rng.choice([-1, 1], 260)
        self.st_z = rng.uniform(0.2, 1.8, 260)
        print(f'[show] ready in {time.time() - t:.1f}s', flush=True)

    def _grid(self, key, grid):
        if self.grid_of != key:
            self.r.set_water_grid(*grid)
            self.grid_of = key

    def _village(self, key, mesh):
        if self.mesh_key != key:
            self.r.set_ground_region(*mesh)
            self.mesh_key = key

    # -- round 4: the sponge -------------------------------------------------------------------------------------------
    def drink(self, s, u):
        if s.extra.get('drink_done'):
            return 1.0
        d = s.extra.get('drink')
        if not d:
            return 0.0
        return float(np.clip((u - d[0]) / (d[1] - d[0]), 0.0, 1.0))

    def streaks(self, q):
        """Water sucked into the sponge: streaks spiralling in from the sea side."""
        if q <= 0.0 or q >= 0.75:
            return None
        S = np.array(E.SPONGE)
        life = 0.22
        out = []
        for k in range(len(self.st_ang)):
            ph = (q * 3.2 / life * 0.25 + self.st_t0[k]) % 1.0      # each streak's own trip, over and over
            if q < 0.05 * self.st_t0[k]:
                continue
            def at(p):
                rr = self.st_r[k] * (1.0 - p) ** 1.6 + 0.3
                a = self.st_ang[k] + self.st_spin[k] * (1.0 - p) ** 2
                z = S[2] + self.st_z[k] * (1.0 - p) * np.sin(np.pi * min(1.0, p + 0.2)) + 0.1
                return S + np.array([rr * np.cos(a), rr * np.sin(a), z - S[2]])
            head = at(ph)
            tail = at(max(0.0, ph - 0.18))
            alpha = np.sin(np.pi * ph) ** 0.7 * (1.0 - np.clip((q - 0.55) / 0.2, 0, 1)) * 0.9
            out.append([*head, *tail, 0.07 + 0.05 * (1 - ph), alpha])
        return np.array(out, np.float32) if out else None

    def frame(self, ts):
        s, u = E.shot_at(ts)
        return self.render_shot(s, u, ts)

    def render_shot(self, s, u, ts):
        r = self.r
        T = s.clock(u)
        cam = s.camera(u)
        info = {'shot': s, 'u': u, 'T': T, 'rnd': s.rnd, 'broken': 0}
        r.fog = s.fog if s.fog is not None else self.base['fog']
        r.sky_amb = self.base['sky_amb'].copy()
        r.gnd_amb = self.base['gnd_amb'].copy()
        props, vox, fx = {}, [], {}
        ov = s.people(u, T) if s.people else None
        if s.rnd in (1, 2, 3):
            pb = self.pbs[s.rnd]
            self._grid(s.rnd, pb.grid())
            n = pb.broken(T)
            info['broken'] = n
            self._village((s.rnd, n), pb.village_mesh(T))
            water = pb.water_at(T)
            props['block'] = pb.debris_at(T)
            p = pb.people(T, overrides=ov, hide=s.hide)
            if p is not None:
                vox.append(p)
            fx['puffs'] = pb.spray_at(T, **s.spray)
        else:
            self._grid(4, self.wall.grid())
            self._village('intact', self.intact)
            q = self.drink(s, u)
            info['q'] = q
            water = self.wall.water(T, q)
            k = WL.shade(T) * (1.0 - q)
            r.sky_amb = self.base['sky_amb'] * (1.0 - 0.45 * k)
            r.gnd_amb = self.base['gnd_amb'] * (1.0 - 0.5 * k)
            p = self.pbs[2].people(0.0, overrides=ov, hide=s.hide)
            if p is not None:
                vox.append(p)
            puffs = [self.wall.mist(T, q)]
            blocks = []
            place = s.extra.get('place')
            placed = s.extra.get('sponge_wet') or (place is not None and u >= place)
            wet = s.extra.get('sponge_wet') or q > 0.45
            if placed:
                pop = 1.0
                if place is not None:
                    a = (u - place) * s.dur
                    pop = 1.0 if a > 0.12 else 0.85 + 0.15 * a / 0.12
                blocks.append([*E.SPONGE, 0, 0, 0, 1, pop, self.v_wet if wet else self.v_sponge, 1.0, 0.0])
            if s.extra.get('arm'):
                t_sw = s.start + s.extra['swing'] * s.dur
                vox.append(self.arm.instances(cam['eye'], cam['target'], t_sw if ts >= t_sw - 0.4 else None, ts))
                if not placed:
                    f = np.asarray(cam['target'], float) - np.asarray(cam['eye'], float)
                    f /= np.linalg.norm(f)
                    rt = np.cross(f, [0, 0, 1.0])
                    rt /= np.linalg.norm(rt)
                    up = np.cross(rt, f)
                    sw = np.clip((ts - t_sw) / HD.SWING, 0.0, 1.0)
                    sws = np.sin(np.pi * sw)
                    pos = (np.asarray(cam['eye'], float) + f * (0.82 + 0.25 * sws) + rt * (0.40 - 0.10 * sws)
                           + up * (-0.36 - 0.12 * sws))
                    blocks.append([*pos, *_qz(np.radians(38.0)), 0.30, self.v_sponge, 1.0, 0.0])
            if s.extra.get('drink'):
                st = self.streaks(q)
                if st is not None:
                    fx['streaks'] = st
            if blocks:
                props['block'] = np.array(blocks, np.float32)
            fx['puffs'] = np.concatenate(puffs) if puffs else None
        r.water_time = T
        nc = s.nc if s.nc is not None else cam['target']
        r.render(cam, voxels=np.concatenate(vox) if vox else None, props=props, fx=fx, near_center=nc, water=water)
        return r.finish(), info


class Overlay:
    """The HUD over each frame (a pure function of the screen time, so any range of frames can be rendered on its
    own)."""

    def __init__(self, show, W_, H_):
        self.hud = HUDM.HUD(W_, H_)
        self.show = show
        self.verdicts = {n: verdict_time(n) for n in VERDICT}
        self.starts = {n: E.round_start(n) for n in (1, 2, 3, 4)}

    def half_at(self, ts):
        """Half hearts the village has left at screen time ts (refilling quickly when a new round starts)."""
        s, u = E.shot_at(max(ts, 0.0))
        n = s.rnd
        if s.name == 'open':
            return 20

        def raw(s_, u_):
            if s_.rnd not in (2, 3):
                return 20
            frac = 1.0 - self.show.pbs[s_.rnd].broken(s_.clock(u_)) / TOTAL_BLOCKS
            return int(np.ceil(20 * frac - 0.25)) if frac > 0 else 0
        half = raw(s, u)
        age = ts - self.starts[n]
        if n > 1 and age < 0.45:
            ps, pu = E.shot_at(self.starts[n] - 1e-3)
            prev = raw(ps, pu)
            half = int(round(prev + (20 - prev) * np.clip(age / 0.45, 0, 1)))
        return half

    def draw(self, img, ts, info):
        hud = self.hud
        out = img.astype(np.float32)
        s = info['shot']
        if s.name == 'open':
            a = 1.0 - np.clip((ts - (s.dur - 0.12)) / 0.12, 0, 1)
            hud.cold_open(out, ts, a)
            return np.clip(out, 0, 255).astype(np.uint8)
        n = s.rnd
        results = {i - 1: OK[i] for i in VERDICT if ts >= self.verdicts[i]}
        hud.tracker(out, n - 1, results, ts)
        half = self.half_at(ts)
        flash = self.half_at(ts - 0.15) > half and ts - 0.15 >= self.starts[n] + 0.45
        hud.hearts_row(out, half, flash=flash, shake=1.0 if flash else 0.0, t=ts)
        age = ts - self.starts[n]
        hud.title(out, n - 1, age)
        va = ts - self.verdicts[n]
        if va >= 0:
            hud.verdict(out, OK[n], va, text='SURVIVED' if OK[n] else 'DESTROYED')
        if n == 4 and va >= 0.5:
            hud.caption(out, 'WITH 1 SPONGE', 1335, px=8, grad=((255, 250, 160), (255, 205, 60)), pop=va - 0.5)
        return np.clip(out, 0, 255).astype(np.uint8)


def frame_times():
    return [i / FPS for i in range(E.N_FRAMES)]


def contact_sheet(show, overlay, out_path, per_shot=3, scale=0.25, only=None):
    tiles = []
    for s in E.SHOTS:
        if only and s.name not in only:
            continue
        row = []
        for j in range(per_shot):
            ts = s.start + s.dur * (j + 0.5) / per_shot
            img, info = show.frame(ts)
            if overlay is not None:
                img = overlay.draw(img, ts, info)
            im = Image.fromarray(img)
            row.append(np.array(im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)))
        tiles.append(np.concatenate(row, 1))
    # lay the shots out in columns of rows
    per_col = 5
    cols = []
    for c in range(0, len(tiles), per_col):
        col = tiles[c:c + per_col]
        while len(col) < per_col:
            col.append(np.zeros_like(tiles[0]))
        cols.append(np.concatenate(col, 0))
    Image.fromarray(np.concatenate(cols, 1)).save(out_path, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--preview', action='store_true')
    ap.add_argument('--ss', type=float, default=2.0)
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--stills', action='store_true')
    ap.add_argument('--from', dest='f0', type=int, default=0)
    ap.add_argument('--to', dest='f1', type=int, default=10 ** 9)
    ap.add_argument('--frames', default='')
    ap.add_argument('--sheet', action='store_true')
    ap.add_argument('--shots', default='')
    ap.add_argument('--crf', type=int, default=12)
    ap.add_argument('--no-audio', action='store_true')
    ap.add_argument('--no-hud', action='store_true')
    ap.add_argument('--cues-only', action='store_true')
    ap.add_argument('--encode-only', action='store_true')
    ap.add_argument('--segments', action='store_true', help='write one-second segments even for a partial range')
    ap.add_argument('--concat', action='store_true', help='join the segments, add the audio and encode')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    video_path = os.path.join(args.out, 'video_noaudio.mp4')
    wav = os.path.join(args.out, 'audio.wav')
    if args.concat:
        seg_dir = os.path.join(args.out, 'seg')
        segs = sorted(x for x in os.listdir(seg_dir) if x.endswith('.mp4') and not x.endswith('.part.mp4'))
        assert len(segs) == (E.N_FRAMES + SEG - 1) // SEG, f'{len(segs)} segments'
        with open(os.path.join(seg_dir, 'list.txt'), 'w') as fh:
            fh.writelines(f"file '{os.path.abspath(os.path.join(seg_dir, x))}'\n" for x in segs)
        subprocess.run([ffmpeg_exe(), '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i',
                        os.path.join(seg_dir, 'list.txt'), '-c', 'copy', video_path], check=True)
        import audio
        audio.build(wav)
        final = os.path.join(args.out, 'final.mp4')
        encode_youtube(video_path, wav, final, args.out)
        print('final:', final, flush=True)
        return
    if args.encode_only or args.cues_only:
        import audio
        audio.build(wav)
        if args.encode_only:
            encode_youtube(video_path, wav, os.path.join(args.out, 'final.mp4'), args.out)
            print('final:', os.path.join(args.out, 'final.mp4'), flush=True)
        return
    if args.preview:
        r, wd = SC.make_renderer(width=W // 2, height=H // 2, ss=1.0, shadow_res=2048, sky_res=(4096, 1024))
    else:
        r, wd = SC.make_renderer(width=W, height=H, ss=args.ss, shadow_res=4096)
    show = Show(r, wd)
    overlay = None if args.no_hud else Overlay(show, W // 2 if args.preview else W, H // 2 if args.preview else H)
    if args.sheet:
        contact_sheet(show, overlay, os.path.join(args.out, 'sheet.jpg'), per_shot=4 if args.shots else 3,
                      scale=0.5 if args.shots else 0.25, only=set(args.shots.split(',')) if args.shots else None)
        print('sheet:', os.path.join(args.out, 'sheet.jpg'), flush=True)
        return
    pick = {int(v) for v in args.frames.split(',')} if args.frames else None
    if pick:
        args.stills = True
    n = E.N_FRAMES
    seg_dir = os.path.join(args.out, 'seg')
    os.makedirs(seg_dir, exist_ok=True)
    whole = (args.segments or (args.f0 == 0 and args.f1 >= n)) and args.every == 1 and not args.stills
    from writer import Writer
    wr = None
    t_start = time.time()
    n_out = 0
    ow, oh = (W // 2, H // 2) if args.preview else (W, H)
    for i in range(n):
        if not (args.f0 <= i < args.f1) or i % args.every or (pick is not None and i not in pick):
            continue
        seg = os.path.join(seg_dir, f's{i // SEG:03d}.mp4')
        if not args.stills and whole and os.path.exists(seg):
            continue
        ts = i / FPS
        img, info = show.frame(ts)
        if overlay is not None:
            img = overlay.draw(img, ts, info)
        if args.stills:
            Image.fromarray(img).save(os.path.join(args.out, f'{i:04d}.jpg'), quality=92)
        else:
            if wr is None:
                wr = Writer(seg if whole else video_path, ow, oh, FPS / args.every, crf=args.crf)
            wr.write(img)
            if whole and (i % SEG == SEG - 1 or i == n - 1):
                wr.close()
                wr = None
        n_out += 1
        if n_out % 30 == 0:
            el = time.time() - t_start
            print(f'  frame {i}/{n} ({info["shot"].name}) {el:.0f}s ({el / n_out:.2f}s/frame)', flush=True)
    if wr is not None:
        wr.close()
    print(f'video frames: {n_out}, render time {time.time() - t_start:.0f}s', flush=True)
    if whole and not args.segments:
        segs = sorted(x for x in os.listdir(seg_dir) if x.endswith('.mp4'))
        with open(os.path.join(seg_dir, 'list.txt'), 'w') as fh:
            fh.writelines(f"file '{os.path.abspath(os.path.join(seg_dir, x))}'\n" for x in segs)
        subprocess.run([ffmpeg_exe(), '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i',
                        os.path.join(seg_dir, 'list.txt'), '-c', 'copy', video_path], check=True)
        if not args.no_audio:
            import audio
            audio.build(wav)
            final = os.path.join(args.out, 'final.mp4')
            encode_youtube(video_path, wav, final, args.out)
            print('final:', final, flush=True)


if __name__ == '__main__':
    main()
