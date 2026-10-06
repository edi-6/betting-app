"""Automated checks on a rendered video: frame count/format, black or corrupt frames, flicker.

The video is one continuous camera move, so a big brightness jump from one frame to the next is a glitch unless the
action explains it: the creeper's fuse flashing, the blast (and the smoke and dirt in front of the lens), and the
lightning at the end.

python qa.py VIDEO.mp4 CUES.json
"""
import json
import subprocess
import sys

import numpy as np

from main import ffmpeg_exe


def frames(path, w=270, h=480):
    cmd = [ffmpeg_exe(), '-v', 'error', '-i', path, '-vf', f'scale={w}:{h}', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = w * h * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield np.frombuffer(b, np.uint8).reshape(h, w, 3)


def exempt_frames(meta):
    ok = set()
    lit = None
    for i, c in enumerate(meta['frames']):
        if 'lit' in c['events']:
            lit = i
        if 'blast' in c['events'] and lit is not None:
            ok.update(range(lit, i))                  # the fuse's flashing (at most 3 a second, by design)
        if 'blast' in c['events']:
            ok.update(range(i, i + 40))
        if 'strike' in c['events']:
            ok.update(range(i - 2, i + 20))
    return ok


def main(video, cues_path):
    with open(cues_path) as fh:
        meta = json.load(fh)
    cues = meta['frames']
    ok = exempt_frames(meta)
    issues = []
    lum_prev = None
    n = 0
    for i, f in enumerate(frames(video)):
        n += 1
        g = f.astype(np.float32)
        lum = g.mean()
        if lum < 10:
            issues.append((i, f'very dark frame (mean {lum:.1f})'))
        if (f.max(axis=2) - f.min(axis=2) < 3).mean() > 0.95:
            issues.append((i, 'frame is almost uniform grey'))
        mag = ((g[..., 0] > 240) & (g[..., 1] < 15) & (g[..., 2] > 240)).mean()
        if mag > 0.001:
            issues.append((i, f'magenta pixels {mag:.4f}'))
        if lum_prev is not None and i not in ok and abs(lum - lum_prev) > 12:
            issues.append((i, f'brightness jump {lum_prev:.0f}->{lum:.0f}'))
        lum_prev = lum
    print(f'frames decoded: {n}, cues: {len(cues)}')
    if n != len(cues):
        issues.append((-1, f'frame count mismatch video {n} vs cues {len(cues)}'))
    for i, msg in issues[:80]:
        print(f'  frame {i}: {msg}')
    print('issues:', len(issues))
    return issues


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
