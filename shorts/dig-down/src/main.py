"""Render the short: "I dug straight down" -- a Minecraft world in cross-section (x-ray view), Steve digging from the
grass to the diamonds, and one block too far.

python main.py --out ../output                                 # the video (resumable, 1 s segments) + soundtrack
python main.py --out ../stills --frames 0,300,900 --no-audio     # a few frames as stills
python main.py --out ../output --encode-only                    # redo the soundtrack and the encode
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
from PIL import Image

import chars as CH
import hud as HD
import render as RN
import story as ST
import world as WD

W, H = RN.W, RN.H
FPS = 60
SEG = 60
TITLE = 'I Dug Straight Down (X-Ray View)'


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return 'ffmpeg'


def encode_youtube(video_in, wav_in, out, workdir, bitrate='14M'):
    gop = FPS // 2
    common = ['-c:v', 'libx264', '-preset', 'slower', '-profile:v', 'high', '-level', '4.2',
              '-b:v', bitrate, '-maxrate', '24M', '-bufsize', '30M', '-pix_fmt', 'yuv420p', '-r', str(FPS),
              '-x264-params', f'keyint={gop}:min-keyint={gop}:scenecut=0:open-gop=0:bframes=2',
              '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv']
    log = os.path.join(workdir, 'x264pass')
    subprocess.run([ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', video_in] + common +
                   ['-pass', '1', '-passlogfile', log, '-an', '-f', 'null', os.devnull], check=True)
    subprocess.run([ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', video_in, '-i', wav_in, '-map', '0:v:0', '-map',
                    '1:a:0'] + common +
                   ['-pass', '2', '-passlogfile', log, '-c:a', 'aac', '-b:a', '384k', '-ar', '48000', '-ac', '2',
                    '-movflags', '+faststart', '-shortest', '-metadata', f'title={TITLE}', out], check=True)


class Show:
    def __init__(self):
        self.w = WD.World()
        self.s = ST.Story(self.w)
        self.r = RN.Renderer()
        self.cast = CH.Cast(self.s, self.r)
        self.hud = HD.Hud()
        self.n = int(round(self.s.T * FPS))
        yy, xx = np.mgrid[0:H, 0:W]
        rr = np.hypot((xx - W / 2) / (H / 2), (yy - H / 2) / (H / 2))
        self.vig = np.clip((rr - 0.55) / 0.55, 0, 1)[..., None].astype(np.float32)

    def frame(self, i, hud=True, cam=None):
        """Frame i as uint8 RGB. cam: optional (cx, cy, zoom) to use instead of the story's camera (the cover)."""
        s = self.s
        t = i / FPS
        ev = s.events
        te = min(t, ev['dead'] + 0.3)                     # the world freezes under the death screen
        cx, cy, zoom, shake = s.camera(te)
        if cam is not None:
            (cx, cy, zoom), shake = cam, 0.0
        if shake > 0:
            cx += shake * np.sin(t * 91.0)
            cy += shake * np.cos(t * 77.0)
        depth = np.clip((WD.SURFACE - cy) / 120.0, 0, 1)
        xray = 0.78 - 0.14 * depth
        img = self.r.frame(s.fg_at(te), self.w.bg, s.lights_at(te), s.deco_at(te), (cx, cy, zoom), te,
                           crack=s.crack_at(te),
                           draw_entities=lambda im, to_px, rend: self.cast.draw(im, to_px, rend, te),
                           xray=xray, dark=0.0)
        # the darkness effect, the hurt flash, the vignette (deeper and darker further down)
        d = s.darkness(te)
        v = 0.28 + 0.25 * depth + 0.6 * d
        img = img * (1 - v * self.vig)
        img *= 1 - 0.55 * d
        hurt = s.hurt_at(te) * (0.35 if te >= ev['lava'] else 1.0)      # (gentle in the lava: no fast flashing)
        if hurt > 0:
            img = img * (1 - 0.35 * hurt) + np.array([200, 0, 0]) * 0.35 * hurt * (0.5 + 0.5 * self.vig)
        if te >= ev['lava']:
            k = min(1.0, (te - ev['lava']) / 0.6)
            img = img * (1 - 0.25 * k) + np.array([255, 90, 10]) * 0.25 * k * self.vig
        img = np.clip(img, 0, 255)
        if hud:
            return self.hud.draw(img, s, t)
        return img.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--stills', action='store_true')
    ap.add_argument('--frames', default='')
    ap.add_argument('--crf', type=int, default=12)
    ap.add_argument('--no-audio', action='store_true')
    ap.add_argument('--no-hud', action='store_true')
    ap.add_argument('--encode-only', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    show = Show()
    n = show.n
    video_path = os.path.join(args.out, 'video_noaudio.mp4')
    with open(os.path.join(args.out, 'cues.json'), 'w') as fh:
        json.dump({'fps': FPS, 'frames': n, 'events': show.s.events}, fh, default=float)
    if args.encode_only:
        import sfx
        wav = os.path.join(args.out, 'audio.wav')
        sfx.build(show.s, wav)
        encode_youtube(video_path, wav, os.path.join(args.out, 'final.mp4'), args.out)
        print('final:', os.path.join(args.out, 'final.mp4'), flush=True)
        return
    pick = sorted({int(v) for v in args.frames.split(',')}) if args.frames else None
    if pick:
        args.stills = True
    print(f'[dig] {n} frames ({n / FPS:.2f}s); ' + ', '.join(f'{k} {v:.2f}' for k, v in show.s.events.items()),
          flush=True)
    seg_dir = os.path.join(args.out, 'seg')
    os.makedirs(seg_dir, exist_ok=True)
    whole = args.every == 1 and not args.stills
    from writer import Writer
    wr = None
    t0 = time.time()
    done = 0
    for i in (pick if pick is not None else range(n)):
        if pick is None and i % args.every:
            continue
        seg = os.path.join(seg_dir, f's{i // SEG:03d}.mp4')
        if whole and os.path.exists(seg):
            continue
        img = show.frame(i, hud=not args.no_hud)
        if args.stills:
            Image.fromarray(img).save(os.path.join(args.out, f'{i:04d}.jpg'), quality=92)
        else:
            if wr is None:
                wr = Writer(seg if whole else video_path, W, H, FPS, crf=args.crf)
            wr.write(img)
            if whole and (i % SEG == SEG - 1 or i == n - 1):
                wr.close()
                wr = None
        done += 1
        if done % 60 == 0:
            el = time.time() - t0
            print(f'  frame {i}/{n}  {el:.0f}s  ({el / done:.2f}s/frame)', flush=True)
    if wr is not None:
        wr.close()
    print(f'frames: {done}, render time {time.time() - t0:.0f}s', flush=True)
    if whole:
        segs = sorted(s for s in os.listdir(seg_dir) if s.endswith('.mp4') and not s.endswith('.part.mp4'))
        with open(os.path.join(seg_dir, 'list.txt'), 'w') as fh:
            fh.writelines(f"file '{os.path.abspath(os.path.join(seg_dir, s))}'\n" for s in segs)
        subprocess.run([ffmpeg_exe(), '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i',
                        os.path.join(seg_dir, 'list.txt'), '-c', 'copy', video_path], check=True)
        if not args.no_audio:
            import sfx
            wav = os.path.join(args.out, 'audio.wav')
            sfx.build(show.s, wav)
            final = os.path.join(args.out, 'final.mp4')
            encode_youtube(video_path, wav, final, args.out)
            print('final:', final, flush=True)


if __name__ == '__main__':
    main()
