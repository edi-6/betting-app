"""Render "Minecraft Rollercoaster Through All 3 Dimensions".

Usage:
  python main.py --out DIR [--preview] [--fps N] [--from SEC --to SEC] [--stills] [--every N] [--no-audio]
                 [--encode-only] [--cues-only]

--preview renders at half resolution; --fps renders a lighter preview at a lower frame rate (the timeline is the
same); --stills saves PNGs instead of a video; --every N keeps one frame in N. After a full render the soundtrack is
built from the ride's cue sheet and the video is encoded for YouTube (final.mp4); --cues-only builds just the cue
sheet and the soundtrack, --encode-only redoes the soundtrack and the encode from an existing render.
"""
import argparse
import os
import subprocess
import time

import numpy as np
from PIL import Image

import audio as AU
import cues as CU
import director as DR
import props as PR
import ride as RD
import scene as SC
import world_deep
import world_end
import world_nether
import world_over
import world_sift

TITLE = 'Minecraft Rollercoaster Through All 4 Dimensions'
SKIES = ('golden', 'nether', 'end', 'deep', 'sift')
WORLDS = (('over', world_over, 'world_over.py'), ('nether', world_nether, 'world_nether.py'),
          ('end', world_end, 'world_end.py'), ('deep', world_deep, 'world_deep.py'),
          ('sift', world_sift, 'world_sift.py'))


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


def encode_youtube(video_in, wav_in, out, workdir, fps, bitrate='16M'):
    """Two-pass H.264 following YouTube's recommended upload settings (High profile, closed GOP of half the frame
    rate, 2 B-frames, BT.709 tags) muxed with 384 kbps AAC at 48 kHz, moov atom up front. 16 Mbps: above the
    recommendation for 1080p60, the lava and the particles need it."""
    gop = int(round(fps / 2))
    common = ['-c:v', 'libx264', '-preset', 'slower', '-profile:v', 'high', '-level', '4.2',
              '-b:v', bitrate, '-maxrate', '24M', '-bufsize', '32M', '-pix_fmt', 'yuv420p', '-r', f'{fps:g}',
              '-x264-params', f'keyint={gop}:min-keyint={gop}:scenecut=0:open-gop=0:bframes=2',
              '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv']
    log = os.path.join(workdir, 'x264pass')
    subprocess.run([ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', video_in] + common +
                   ['-pass', '1', '-passlogfile', log, '-an', '-f', 'null', os.devnull], check=True)
    subprocess.run([ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', video_in, '-i', wav_in, '-map', '0:v:0', '-map',
                    '1:a:0'] + common +
                   ['-pass', '2', '-passlogfile', log, '-c:a', 'aac', '-b:a', '384k', '-ar', '48000', '-ac', '2',
                    '-movflags', '+faststart', '-shortest', '-metadata', f'title={TITLE}', out], check=True)


def soundtrack(out, ride=None):
    """The cue sheet and the soundtrack from it: out/cues.json, out/audio.wav."""
    if ride is None:
        ride = RD.Ride(CU.load_metas())
    cs = CU.write(os.path.join(out, 'cues.json'), ride)
    return AU.build(cs, os.path.join(out, 'audio.wav'))


def setup(preview=False, ss=1.0):
    w, h = (RD.W // 2, RD.H // 2) if preview else (RD.W, RD.H)
    r = SC.make_renderer(w, h, ss=ss, skies=SKIES, near_half=48.0, far_half=260.0)
    PR.register(r)
    DR.register(r)
    metas = {key: RD.load_world(r, key, mod.build, [src]) for key, mod, src in WORLDS}
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
    ap.add_argument('--cues-only', action='store_true')
    ap.add_argument('--encode-only', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    video = os.path.join(a.out, 'video.mp4')
    if a.cues_only or a.encode_only:
        wav = soundtrack(a.out)
        if a.encode_only:
            encode_youtube(video, wav, os.path.join(a.out, 'final.mp4'), a.out, RD.FPS)
            print('encoded', os.path.join(a.out, 'final.mp4'), flush=True)
        return
    r, ride = setup(a.preview)
    print(ride.summary(), flush=True)
    step = max(1, int(round(RD.FPS / a.fps))) * a.every
    f0 = int(a.t0 * RD.FPS)
    f1 = ride.nframes if a.t1 is None else min(ride.nframes, int(a.t1 * RD.FPS))
    frames = list(range(f0, f1, step))
    wr = None
    if not a.stills:
        wr = Writer(video, r.W, r.H, RD.FPS / step)
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
    print('rendered', flush=True)
    whole = f0 == 0 and f1 == ride.nframes and not a.stills
    if whole and not a.no_audio:
        wav = soundtrack(a.out, ride)
        if step == 1 and not a.preview:
            encode_youtube(video, wav, os.path.join(a.out, 'final.mp4'), a.out, RD.FPS)
            print('encoded', os.path.join(a.out, 'final.mp4'), flush=True)
        else:
            # a preview: just put the soundtrack on it
            prev = os.path.join(a.out, 'preview.mp4')
            subprocess.run([ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', video, '-i', wav, '-map', '0:v:0',
                            '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-shortest', prev],
                           check=True)
            print('preview', prev, flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
