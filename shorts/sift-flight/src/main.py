"""Render "POV: your first elytra flight into Minecraft's new dimension".

Usage:
  python main.py --out DIR [--scale K] [--ss S] [--every N] [--from SEC --to SEC] [--stills] [--no-mblur]
                 [--no-audio] [--cues-only] [--encode-only]

--scale renders a smaller preview (0.5 = 540x960); --ss supersamples (2 = four samples a pixel); --every N keeps one
frame in N. After a whole render the soundtrack is built from the flight's cue sheet and the video is encoded for
YouTube (final.mp4), or for a preview just muxed (preview.mp4). --cues-only builds the cue sheet and the soundtrack,
--encode-only redoes the soundtrack and the encode from an existing render.
"""
import argparse
import os
import subprocess
import time

import numpy as np
from PIL import Image

import director as DR
import flight as FL

TITLE = 'POV: Your First Elytra Flight in the Sift'
NFRAMES = int(round(FL.DURATION * FL.FPS))


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return 'ffmpeg'


class Writer:
    """Raw frames piped to x264 (a near-lossless intermediate; the YouTube encode comes after)."""

    def __init__(self, path, w, h, fps, crf=10):
        self.path = path
        self.tmp = path + '.part.mp4'
        cmd = [ffmpeg_exe(), '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r',
               str(fps), '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', str(crf), '-pix_fmt',
               'yuv444p', self.tmp]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, img):
        self.p.stdin.write(np.ascontiguousarray(img, np.uint8).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()
        os.replace(self.tmp, self.path)


def encode_youtube(video_in, wav_in, out, workdir, fps, bitrate='20M'):
    """Two-pass H.264 following YouTube's recommended upload settings (High profile, closed GOP of half the frame
    rate, 2 B-frames, BT.709 tags) muxed with 384 kbps AAC at 48 kHz, moov atom up front. 20 Mbps: well above the
    recommendation for 1080p60 - the ichor's swirls, the leaves and the fireworks' sparks need it."""
    gop = int(round(fps / 2))
    common = ['-c:v', 'libx264', '-preset', 'slower', '-profile:v', 'high', '-level', '4.2',
              '-b:v', bitrate, '-maxrate', '30M', '-bufsize', '40M', '-pix_fmt', 'yuv420p', '-r', f'{fps:g}',
              '-x264-params', f'keyint={gop}:min-keyint={gop}:scenecut=0:open-gop=0:bframes=2',
              '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv']
    log = os.path.join(workdir, 'x264pass')
    subprocess.run([ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', video_in] + common +
                   ['-pass', '1', '-passlogfile', log, '-an', '-f', 'null', os.devnull], check=True)
    subprocess.run([ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', video_in, '-i', wav_in, '-map', '0:v:0', '-map',
                    '1:a:0'] + common +
                   ['-pass', '2', '-passlogfile', log, '-c:a', 'aac', '-b:a', '384k', '-ar', '48000', '-ac', '2',
                    '-movflags', '+faststart', '-shortest', '-metadata', f'title={TITLE}', out], check=True)


def soundtrack(out):
    """The cue sheet and the soundtrack from it: out/cues.json, out/audio.wav."""
    import audio as AU
    import cues as CU
    cs = CU.write(os.path.join(out, 'cues.json'))
    return AU.build(cs, os.path.join(out, 'audio.wav'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--ss', type=float, default=2.0)
    ap.add_argument('--from', dest='t0', type=float, default=0.0)
    ap.add_argument('--to', dest='t1', type=float, default=None)
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--stills', action='store_true')
    ap.add_argument('--no-mblur', action='store_true')
    ap.add_argument('--no-audio', action='store_true')
    ap.add_argument('--cues-only', action='store_true')
    ap.add_argument('--encode-only', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    video = os.path.join(a.out, 'video.mp4')
    if a.cues_only or a.encode_only:
        wav = soundtrack(a.out)
        if a.encode_only:
            encode_youtube(video, wav, os.path.join(a.out, 'final.mp4'), a.out, FL.FPS)
            print('encoded', os.path.join(a.out, 'final.mp4'), flush=True)
        return
    r, fl, meta = DR.setup(scale=a.scale, ss=a.ss)
    print(fl.summary().splitlines()[0], flush=True)
    f0 = int(round(a.t0 * FL.FPS))
    f1 = NFRAMES if a.t1 is None else min(NFRAMES, int(round(a.t1 * FL.FPS)))
    frames = list(range(f0, f1, a.every))
    wr = None if a.stills else Writer(video, r.W, r.H, FL.FPS / a.every)
    t0 = time.time()
    for n, f in enumerate(frames):
        img = DR.render_frame(r, fl, meta, f, mblur=not a.no_mblur)
        if a.stills:
            Image.fromarray(img).save(os.path.join(a.out, f'f{f:04d}.png'))
        else:
            wr.write(img)
        if n % 30 == 0 or n == len(frames) - 1:
            el = time.time() - t0
            left = el / (n + 1) * (len(frames) - n - 1)
            print(f'  frame {f} ({f / FL.FPS:.2f}s)  {n + 1}/{len(frames)}  {el / (n + 1):.2f}s/frame  '
                  f'{left / 60:.1f} min left', flush=True)
    if wr:
        wr.close()
    print('rendered', flush=True)
    whole = f0 == 0 and f1 == NFRAMES and not a.stills
    if whole and not a.no_audio:
        wav = soundtrack(a.out)
        if a.every == 1 and a.scale == 1.0:
            encode_youtube(video, wav, os.path.join(a.out, 'final.mp4'), a.out, FL.FPS)
            print('encoded', os.path.join(a.out, 'final.mp4'), flush=True)
        else:
            prev = os.path.join(a.out, 'preview.mp4')
            subprocess.run([ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', video, '-i', wav, '-map', '0:v:0',
                            '-map', '1:a:0', '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', '-c:a', 'aac',
                            '-b:a', '192k', '-shortest', prev], check=True)
            print('preview', prev, flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
