"""Render the short: "How big is a Minecraft world?" -- one continuous zoom out, from a single pixel of Steve's eye
to the whole world, which turns out to be that pixel. The video loops.

python main.py --out ../output                                   # delivered quality, resumable (1 s segments)
python main.py --out ../stills --frames 0,600,1200 --no-audio      # a few frames as stills
python main.py --out ../preview --every 15 --stills --ss 1        # a quick look at the whole thing
python main.py --out ../output --encode-only                      # redo the soundtrack and the encode
"""
import argparse
import json
import os
import subprocess
import time

import cv2
import numpy as np
from PIL import Image

import zfx as FX
import zhud as ZH
import zoommap as ZM
import zscene as ZS
import ztimeline as TL

W, H = ZS.W, ZS.H
FPS = TL.FPS
SEG = 60
SHUTTER = 0.5                     # of a frame: the zoom's motion blur
TITLE = 'How Big Is a Minecraft World?'


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


class Zoom:
    def __init__(self, ss=2.0):
        self.tl = TL.Timeline()
        self.ss = ss
        sats = ZS.satellites()          # first: if it has to be built it makes (and drops) a GL context of its own
        self.r, wd = ZS.make_renderer(W, H, ss=ss, shadow_res=4096)
        self.sources = wd['smoke']
        self.fire = (-2.5, 1.5, 0.44)
        self.map = ZM.MapRenderer(64, 64, ctx=self.r.ctx)
        self.map.set_satellites(sats)
        self.pmap = ZM.MapRenderer(64, 64, ctx=self.r.ctx)          # the world in the pupil
        self.PW = 1024
        self._pw, self._pw_key = None, None
        self.hud = ZH.Hud(W, H)
        self.steve = ZS.STEVE.instances()
        yy, xx = np.mgrid[0:H, 0:W]
        rr = np.hypot((xx - W / 2 + 0.5) / (H / 2), (yy - H / 2 + 0.5) / (H / 2))
        self.vig = (1.0 - 0.16 * np.clip((rr - 0.45) / 0.6, 0, 1) ** 1.6)[..., None].astype(np.float32)

    # -- the pictures ----------------------------------------------------------------------------------------------
    def map_frame(self, L, t, w=W, h=H, clip=True, sat=True):
        """The map L metres wide (at the output's scale), w x h output pixels, supersampled: float RGBA 0..1."""
        k = max(1, int(round(self.ss)))
        m = self.map if sat else self.pmap
        m.resize(w * k, h * k)
        ex, ey, er = ZH.EARTH
        ea = ZH.earth_alpha(L)
        img = m.render(L * w / W, t=t, cloud=ZH.cloud_alpha(L), earth=(ex, ey, er, ea) if ea > 0 else None,
                       border=ZH.border_alpha(L), sat=sat, clip=1 if clip else 0)
        if k > 1:
            img = img.reshape(h, k, w, k, 4).mean((1, 3))
        return img

    def frame_3d(self, L3, t):
        r = self.r
        cam, nc = ZS.setup_frame(r, L3)
        fx = {}
        if L3 > 1.8:
            puffs = [FX.smoke(t, self.sources)]
            if L3 < 60:
                puffs.append(FX.flames(t, *self.fire))
            fx['puffs'] = np.concatenate(puffs)
        r.water_time = t
        r.render(cam, voxels=self.steve, near_center=nc, fx=fx)
        img = r.finish().astype(np.float32)
        # the pupil: the whole world, the size of one of his pixels
        pw = self.pupil_world(L3, t)
        if pw is not None:
            pm, x0, y0 = pw
            h, w = pm.shape[:2]
            xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
            pm = pm[ya - y0:yb - y0, xa - x0:xb - x0]
            a = pm[..., 3:4]
            img[ya:yb, xa:xb] = img[ya:yb, xa:xb] * (1 - a) + pm[..., :3] * 255.0 * a
        return img

    def pupil_world(self, L3, t):
        """The whole world as it shows in the pupil (RGBA 0..1, and where it goes) or None if it's under a pixel.
        Big, it's rendered at the size it is; small, the whole world is rendered at 1024 pixels and averaged down to
        its size (so it stays a little world map, with the right colours, at any size)."""
        s = TL.PX / L3 * W
        if s < 1.5:
            return None
        Lw = L3 * TL.RATIO
        k = max(1, int(round(self.ss)))
        if s * k >= self.PW:
            n = 2 * int(np.ceil(s / 2 + 1))
            w, h = min(n, W), min(n, H)
            return self.map_frame(Lw, t, w, h, clip=True, sat=False), (W - w) // 2, (H - h) // 2
        ba = round(ZH.border_alpha(Lw), 3)
        if self._pw_key != ba:
            self.pmap.resize(self.PW, self.PW)
            self._pw = self.pmap.render(TL.WORLD, border=ba, sat=False, clip=1).astype(np.float32)
            self._pw_key = ba
        img = self._pw
        while img.shape[0] > 2 * s + 2:
            img = cv2.pyrDown(img)
        n = img.shape[0]
        a = s / n
        x0, y0 = int(np.floor(W / 2 - s / 2)) - 1, int(np.floor(H / 2 - s / 2)) - 1
        m = int(np.ceil(s)) + 3
        cx, cy = W / 2 - s / 2 - x0, H / 2 - s / 2 - y0
        M = np.array([[a, 0, cx + 0.5 * a - 0.5], [0, a, cy + 0.5 * a - 0.5]], np.float32)
        out = cv2.warpAffine(img, M, (m, m), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return out, x0, y0

    def proj(self, L, ph):
        if ph == 'map':
            s = W / L
            return lambda x, y, z=0.0: (W / 2 + x * s, H / 2 - y * s)
        L3 = L / TL.RATIO if ph == 'eye' else L
        cam, hgt = ZS.camera(L3)
        th = np.tan(np.radians(cam['fov']) / 2) * ZS.ASPECT
        f = (W / 2) / th
        zc = ZS.PUPIL_Z + hgt
        return lambda x, y, z=0.0: (W / 2 + x * f / (zc - z), H / 2 - y * f / (zc - z))

    def zoom_blur(self, img, t):
        s = self.tl.speed_at(t) * SHUTTER / FPS                  # log-scale change while the shutter is open
        reach = s * np.hypot(W, H) / 2
        n = int(np.clip(np.ceil(reach * 1.5), 1, 16))
        if n < 2:
            return img
        acc = np.zeros_like(img)
        for k in range(n):
            a = float(np.exp(s * ((k + 0.5) / n - 0.5)))
            M = np.array([[a, 0, W / 2 * (1 - a)], [0, a, H / 2 * (1 - a)]], np.float32)
            acc += cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return acc / n

    def render(self, i, hud=True):
        t, L = self.tl.frame(i)
        ph, w3 = TL.phase(L)
        if ph == 'eye':
            img = self.frame_3d(L / TL.RATIO, t)
        elif ph == '3d':
            img = self.frame_3d(L, t)
        else:
            m = self.map_frame(L, t)[..., :3] * 255.0
            if ph == 'fade':
                img = self.frame_3d(L, t) * w3 + m * (1 - w3)
            else:
                img = m
        img = self.hud.world(img, L, ph, self.proj(L, ph))
        img = self.zoom_blur(img, t)
        img = img * self.vig
        img = np.clip(img, 0, 255)
        if hud:
            return self.hud.draw(img, L)
        return img.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--ss', type=float, default=2.0)
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--stills', action='store_true')
    ap.add_argument('--from', dest='f0', type=int, default=0)
    ap.add_argument('--to', dest='f1', type=int, default=10 ** 9)
    ap.add_argument('--frames', default='')
    ap.add_argument('--crf', type=int, default=12)
    ap.add_argument('--no-audio', action='store_true')
    ap.add_argument('--no-hud', action='store_true')
    ap.add_argument('--encode-only', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tl = TL.Timeline()
    n = tl.n
    video_path = os.path.join(args.out, 'video_noaudio.mp4')
    meta = {'fps': FPS, 'frames': n, 'T': tl.T}
    with open(os.path.join(args.out, 'cues.json'), 'w') as fh:
        json.dump(meta, fh)
    if args.encode_only:
        import zaudio
        wav = os.path.join(args.out, 'audio.wav')
        zaudio.build(tl, wav)
        encode_youtube(video_path, wav, os.path.join(args.out, 'final.mp4'), args.out)
        print('final:', os.path.join(args.out, 'final.mp4'), flush=True)
        return
    pick = sorted({int(v) for v in args.frames.split(',')}) if args.frames else None
    if pick:
        args.stills = True
    z = Zoom(ss=args.ss)
    print(f'[zoom] {n} frames ({tl.T:.2f}s)', flush=True)
    seg_dir = os.path.join(args.out, 'seg')
    os.makedirs(seg_dir, exist_ok=True)
    whole = args.f0 == 0 and args.f1 >= n and args.every == 1 and not args.stills
    from writer import Writer
    wr = None
    t0 = time.time()
    done = 0
    frames = pick if pick is not None else range(n)
    for i in frames:
        if not (args.f0 <= i < args.f1) or (pick is None and i % args.every):
            continue
        seg = os.path.join(seg_dir, f's{i // SEG:03d}.mp4')
        if whole and os.path.exists(seg):
            continue
        img = z.render(i, hud=not args.no_hud)
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
        if done % 30 == 0:
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
            import zaudio
            wav = os.path.join(args.out, 'audio.wav')
            zaudio.build(tl, wav)
            final = os.path.join(args.out, 'final.mp4')
            encode_youtube(video_path, wav, final, args.out)
            print('final:', final, flush=True)


if __name__ == '__main__':
    main()
