"""Check a rendered video: its streams and durations, and frame by frame (decoded small) for black or blown-out
frames and sudden jumps (the flight has no cuts), and the loop seam (the last frame into the first).

python qa.py VIDEO.mp4
"""
import subprocess
import sys

import numpy as np

from main import ffmpeg_exe


def probe(path):
    out = subprocess.run([ffmpeg_exe(), '-hide_banner', '-i', path], capture_output=True, text=True).stderr
    return [line.strip() for line in out.splitlines() if 'Stream #' in line or 'Duration' in line]


def frames(path, w=90, h=160):
    cmd = [ffmpeg_exe(), '-v', 'error', '-i', path, '-vf', f'scale={w}:{h}', '-f', 'rawvideo', '-pix_fmt', 'gray', '-']
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(np.float32)


def main():
    path = sys.argv[1]
    for line in probe(path):
        print(line)
    F = frames(path)
    fps = 60
    mean = F.mean(axis=(1, 2))
    diff = np.abs(np.diff(F, axis=0)).mean(axis=(1, 2))
    print(f'{len(F)} frames ({len(F) / fps:.2f} s), mean luma {mean.min():.0f}..{mean.max():.0f}')
    bad = 0
    for i, m in enumerate(mean):
        if m < 6 or m > 250:
            print(f'  frame {i} ({i / fps:.2f}s): {"black" if m < 6 else "white"} (luma {m:.0f})')
            bad += 1
    thr = np.median(diff) * 6 + 8
    for i, d in enumerate(diff):
        if d > thr:
            print(f'  jump into frame {i + 1} ({(i + 1) / fps:.2f}s): {d:.1f} (typical {np.median(diff):.1f})')
            bad += 1
    seam = np.abs(F[-1] - F[0]).mean()
    near = np.median(diff[-30:])
    print(f'loop seam (last frame -> first): {seam:.2f} (the frame steps around it {near:.2f}, '
          f'typical over the video {np.median(diff):.1f})')
    print('OK' if bad == 0 else f'{bad} things to look at')


if __name__ == '__main__':
    main()
