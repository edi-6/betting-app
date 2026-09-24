"""Automated checks on a rendered video: frame count/format, black or corrupt frames, flicker within shots.

Brightness jumps that belong to the action are expected and are not reported: cuts, the rewind, blocks giving
way, the TNT flashing white and going off, the hoses bursting (the alarm), and the press blowing up.

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


def main(video, cues_path):
    cues = json.load(open(cues_path))['frames']
    cut = set()
    flash = set()
    for i in range(1, len(cues)):
        c, p = cues[i], cues[i - 1]
        ev = c.get('events', {}) or {}
        if (c.get('flash') or 0.0) > 0.02 or (p.get('flash') or 0.0) > 0.02 or \
                any(k in ev for k in ('explode', 'press_break', 'break', 'hoses', 'prime')) or \
                c.get('seg') != p.get('seg') or c.get('seg') == 'rewind' or \
                (c.get('kind') == 'tnt' and c.get('state') == 'whole'):
            flash.update((i, i + 1, i + 2))
    first_main = next((i for i, c in enumerate(cues) if c.get('seg') == 'main'), 0)
    flash.update(range(max(0, first_main - 3), first_main + 3))
    prev_key = None
    for i, c in enumerate(cues):
        key = (c['seg'], c.get('shot'))
        if key != prev_key or c.get('cut'):
            cut.add(i)
        prev_key = key
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
        # magenta/NaN style garbage: pixels far outside the palette of the scene
        mag = ((g[..., 0] > 240) & (g[..., 1] < 15) & (g[..., 2] > 240)).mean()
        if mag > 0.001:
            issues.append((i, f'magenta pixels {mag:.4f}'))
        if lum_prev is not None and i not in cut and i not in flash:
            if abs(lum - lum_prev) > 18:
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
