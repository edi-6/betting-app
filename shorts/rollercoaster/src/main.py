"""Render "Minecraft Rollercoaster Through All 3 Dimensions".

Usage:
  python main.py --out DIR [--preview] [--fps N] [--from SEC --to SEC] [--stills] [--every N] [--no-audio]
                 [--encode-only] [--cues-only]

--preview renders at half resolution; --fps renders a lighter preview at a lower frame rate (the timeline is the
same); --stills saves PNGs instead of a video; --every N keeps one frame in N.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

import director as DR
import props as PR
import ride as RD
import scene as SC
import world_end
import world_nether
import world_over


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return 'ffmpeg'


class Writer:
    """Raw frames piped to x264 (high quality intermediate; the YouTube encode comes after)."""

    def __init__(self, path, w, h, fps, crf=14):
        self.path = path
        self.tmp = path + '.part.mp4'
        cmd = [ffmpeg_exe(), '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r',
               str(fps), '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', str(crf), '-pix_fmt',
               'yuv420p', self.tmp]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, img):
        self.p.stdin.write(np.ascontiguousarray(img, np.uint8).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()
        os.replace(self.tmp, self.path)


def setup(preview=False):
    w, h = (RD.W // 2, RD.H // 2) if preview else (RD.W, RD.H)
    r = SC.make_renderer(w, h, skies=('golden', 'nether', 'end'), near_half=48.0, far_half=260.0)
    PR.register(r)
    DR.register(r)
    metas = {}
    metas['over'] = RD.load_world(r, 'over', world_over.build, ['world_over.py'])
    metas['nether'] = RD.load_world(r, 'nether', world_nether.build, ['world_nether.py'])
    metas['end'] = RD.load_world(r, 'end', world_end.build, ['world_end.py'])
    ride = RD.Ride(metas)
    return r, ride


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--preview', action='store_true')
    ap.add_argument('--fps', type=float, default=RD.FPS)
    ap.add_argument('--from', dest='t0', type=float, default=0.0)
    ap.add_argument('--to', dest='t1', type=float, default=None)
    ap.add_argument('--stills', action='store_true')
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--no-audio', action='store_true')
    ap.add_argument('--no-hud', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    r, ride = setup(a.preview)
    print(ride.summary(), flush=True)
    step = max(1, int(round(RD.FPS / a.fps))) * a.every
    f0 = int(a.t0 * RD.FPS)
    f1 = ride.nframes if a.t1 is None else min(ride.nframes, int(a.t1 * RD.FPS))
    frames = list(range(f0, f1, step))
    wr = None
    if not a.stills:
        wr = Writer(os.path.join(a.out, 'video.mp4'), r.W, r.H, RD.FPS / step)
    t0 = time.time()
    for n, f in enumerate(frames):
        img = DR.render_frame(r, ride, f, hud=not a.no_hud)
        if a.stills:
            Image.fromarray(img).save(os.path.join(a.out, f'f{f:05d}.png'))
        else:
            wr.write(img)
        if n % 30 == 0 or n == len(frames) - 1:
            el = time.time() - t0
            print(f'  frame {f} ({f / RD.FPS:.2f}s)  {n + 1}/{len(frames)}  {el / (n + 1):.2f}s/frame', flush=True)
    if wr:
        wr.close()
    print('done', flush=True)


if __name__ == '__main__':
    main()
