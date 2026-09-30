"""Check a rendered video: its streams and durations, and frame by frame (decoded small) for black or blown-out
frames and sudden jumps that aren't one of the ride's planned cuts (the portals, the loop).

python qa.py VIDEO.mp4 [CUES.json]
"""
import json
import subprocess
import sys

import numpy as np

from main import ffmpeg_exe


def probe(path):
    out = subprocess.run([ffmpeg_exe(), '-hide_banner', '-i', path], capture_output=True, text=True).stderr
    return [l.strip() for l in out.splitlines() if 'Stream #' in l or 'Duration' in l]


def frames(path, w=90, h=160):
    cmd = [ffmpeg_exe(), '-v', 'error', '-i', path, '-vf', f'scale={w}:{h}', '-f', 'rawvideo', '-pix_fmt', 'gray', '-']
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(np.float32)


def main():
    path = sys.argv[1]
    cues = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else None
    for l in probe(path):
        print(l)
    F = frames(path)
    fps = cues['fps'] if cues else 60
    mean = F.mean(axis=(1, 2))
    diff = np.abs(np.diff(F, axis=0)).mean(axis=(1, 2))
    cuts = []
    if cues:
        cuts = [cues['events'][k] for k in ('portal_a', 'portal_b', 'portal_c', 'explosion', 'land')]
    print(f'{len(F)} frames ({len(F) / fps:.2f} s), mean luma {mean.min():.0f}..{mean.max():.0f}')
    bad = 0
    for i, m in enumerate(mean):
        if (m < 6 or m > 250) and not any(abs(i / fps - c) < 0.5 for c in cuts):
            print(f'  frame {i} ({i / fps:.2f}s): {"black" if m < 6 else "white"} (luma {m:.0f})')
            bad += 1
    thr = np.median(diff) * 6 + 8
    for i, d in enumerate(diff):
        t = (i + 1) / fps
        if d > thr and not any(abs(t - c) < 0.5 for c in cuts):
            print(f'  jump into frame {i + 1} ({t:.2f}s): {d:.1f} (typical {np.median(diff):.1f})')
            bad += 1
    seam = np.abs(F[-1] - F[0]).mean()
    print(f'loop seam (last frame -> first): {seam:.1f} (typical frame step {np.median(diff):.1f})')
    print('OK' if bad == 0 else f'{bad} things to look at')


if __name__ == '__main__':
    main()
