"""Checks on the rendered video: the format, black or frozen frames, jumps from one frame to the next other than the
flash at the win, fast flashing, and the loudness and true peak of the sound (the WAV and the AAC in the MP4).
Also writes a contact sheet (a frame every half second) to look over.

python qa.py OUT_DIR
"""
import json
import os
import subprocess
import sys
import wave

import numpy as np
from PIL import Image, ImageDraw

from writer import ffmpeg_exe

FPS = 60


def frames(path, w=270, h=480):
    cmd = [ffmpeg_exe(), '-v', 'error', '-i', path, '-vf', f'scale={w}:{h}', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = w * h * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield np.frombuffer(b, np.uint8).reshape(h, w, 3)


def decoded_audio(path):
    cmd = [ffmpeg_exe(), '-v', 'error', '-i', path, '-vn', '-f', 's16le', '-ac', '2', '-ar', '48000', '-']
    raw = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(raw, np.int16).reshape(-1, 2) / 32768.0


def loudness(x, label):
    from scipy.signal import resample_poly
    from sound import integrated_lufs
    tp = max(np.abs(resample_poly(x[:, 0], 4, 1)).max(), np.abs(resample_poly(x[:, 1], 4, 1)).max())
    print(f'{label}: {len(x) / 48000:.2f}s, {integrated_lufs(x[:, 0], x[:, 1]):.1f} LUFS, '
          f'true peak {20 * np.log10(tp):.2f} dBFS')


def main(out_dir):
    video = os.path.join(out_dir, 'country-royale.mp4')
    cues = json.load(open(os.path.join(out_dir, 'cues.json')))
    win = int(round(cues['win'] * FPS))
    probe = subprocess.run([ffmpeg_exe(), '-hide_banner', '-i', video], capture_output=True, text=True).stderr
    print('\n'.join(l.strip() for l in probe.splitlines() if 'Duration' in l or 'Stream' in l))
    prev = None
    diffs, lumas, sheet = [], [], []
    for i, f in enumerate(frames(video)):
        g = f.astype(np.float32)
        lumas.append(float(g.mean()))
        diffs.append(float(np.abs(g - prev).mean()) if prev is not None else 0.0)
        prev = g
        if i % (FPS // 2) == 0:
            sheet.append((i, f))
    diffs, lumas = np.array(diffs), np.array(lumas)
    n = len(diffs)
    print(f'frames {n} (expected {int(round(cues["duration"] * FPS))}); black frames {(lumas < 4).sum()}; '
          f'frozen frames {(diffs[1:] < 0.02).sum()}')
    med = float(np.median(diffs[1:]))
    print(f'frame-to-frame change: median {med:.2f}, 99th pct {np.percentile(diffs[1:], 99):.2f}, '
          f'max {diffs.max():.2f} at frame {int(diffs.argmax())}')
    big = [i for i in range(1, n) if diffs[i] > max(6 * med, 12.0)]
    unexplained = [i for i in big if abs(i - win) > 2]
    print('big jumps:', big[:40])
    print('unexplained (not the win flash):', unexplained[:40])
    dl = np.diff(lumas)
    swings = np.where(np.abs(dl) > 20)[0]
    worst = max((np.sum((swings >= s) & (swings < s + FPS)) for s in range(0, n, FPS // 4)), default=0)
    print(f'luma swings > 20: {len(swings)}; most in any second: {worst}')
    wav = os.path.join(out_dir, 'sound.wav')
    if os.path.exists(wav):
        with wave.open(wav) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2) / 32768.0
        loudness(x, 'sound.wav')
    loudness(decoded_audio(video), 'AAC in the MP4')
    cols = 10
    w, h = 162, 288
    rows = (len(sheet) + cols - 1) // cols
    img = Image.new('RGB', (cols * w, rows * h))
    for k, (i, f) in enumerate(sheet):
        im = Image.fromarray(f).resize((w, h))
        ImageDraw.Draw(im).text((4, 4), f'{i / FPS:.1f}', fill=(255, 255, 0))
        img.paste(im, ((k % cols) * w, (k // cols) * h))
    path = os.path.join(out_dir, 'contact.jpg')
    img.save(path, quality=85)
    print('contact sheet:', path)


if __name__ == '__main__':
    main(sys.argv[1])
