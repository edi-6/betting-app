"""Deliverables, once the film is rendered, composed and its sound built.

    python deliver.py master     # output/THE_PLAYER_WHO_NEVER_LOGGED_OUT.mp4 (+ _clean): the full-quality films
    python deliver.py git        # release/film/: compact HEVC copies split into < 95 MB parts, with join scripts
    python deliver.py web DIR    # DIR: the clean film as an HLS stream (fMP4 pieces < 15 MB) for the release page
    python deliver.py preview    # output/preview_<30MB.mp4: a small copy for a quick look on a phone
"""
import hashlib
import os
import subprocess
import sys

import film as FM

ROOT = FM.ROOT
OUT = os.path.join(ROOT, 'output')
REL = os.path.join(ROOT, 'release', 'film')
NAME = 'THE_PLAYER_WHO_NEVER_LOGGED_OUT'
PART = 95_000_000


def ff(*args):
    cmd = [FM.ffmpeg_exe(), '-y', '-hide_banner', '-v', 'error', '-stats'] + [str(a) for a in args]
    print('$', ' '.join(cmd[4:]), flush=True)
    subprocess.run(cmd, check=True)


def src(clean=False):
    v = os.path.join(OUT, 'parts_clean' if clean else 'parts', 'list.txt')
    a = os.path.join(OUT, 'film_audio.wav')
    for p in (v, a):
        if not os.path.exists(p):
            sys.exit(f'missing {p}')
    return v, a


def video_in(v):
    return ['-f', 'concat', '-safe', '0', '-i', v]


def master():
    for clean in (False, True):
        v, a = src(clean)
        out = os.path.join(OUT, NAME + ('_clean' if clean else '') + '.mp4')
        ff(*video_in(v), '-i', a, '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '256k',
           '-movflags', '+faststart', '-shortest', out)
        print(out, os.path.getsize(out) // 2 ** 20, 'MiB')


def hevc(clean, kbps, out):
    """One pass, average bitrate with a ceiling (two passes of x265 would take hours here)."""
    v, a = src(clean)
    x265 = f'log-level=error:aq-mode=3:psy-rd=1.5:deblock=-1,-1:vbv-maxrate={int(kbps * 1.6)}:vbv-bufsize={kbps * 3}'
    ff(*video_in(v), '-i', a, '-map', '0:v', '-map', '1:a', '-c:v', 'libx265', '-preset', 'fast', '-b:v', f'{kbps}k',
       '-x265-params', x265, '-pix_fmt', 'yuv420p', '-tag:v', 'hvc1', '-c:a', 'aac', '-b:a', '160k',
       '-movflags', '+faststart', '-shortest', out)


def split(path, dst_dir):
    """Byte-split into parts (joined back exactly by `cat` / `copy /b`), with the checksum of the whole."""
    os.makedirs(dst_dir, exist_ok=True)
    base = os.path.basename(path)
    h = hashlib.sha256()
    parts = []
    with open(path, 'rb') as fh:
        k = 1
        while True:
            b = fh.read(PART)
            if not b:
                break
            h.update(b)
            p = os.path.join(dst_dir, f'{base}.part{k}')
            with open(p, 'wb') as o:
                o.write(b)
            parts.append(os.path.basename(p))
            k += 1
    digest = h.hexdigest()
    with open(os.path.join(dst_dir, f'{base}.sha256'), 'w') as o:
        o.write(f'{digest}  {base}\n')
    return parts, digest


def git_release():
    os.makedirs(REL, exist_ok=True)
    tmp = os.path.join(OUT, 'git')
    os.makedirs(tmp, exist_ok=True)
    made = []
    for clean, kbps in ((False, 2600), (True, 2000)):
        name = NAME + ('_no_subtitles' if clean else '') + '.mp4'
        full = os.path.join(tmp, name)
        hevc(clean, kbps, full)
        parts, digest = split(full, REL)
        made.append((name, parts, os.path.getsize(full)))
        print(name, os.path.getsize(full) // 2 ** 20, 'MiB ->', parts, flush=True)
    # join scripts
    with open(os.path.join(REL, 'join.sh'), 'w') as o:
        o.write('#!/bin/sh\n# Joins the parts back into the films (macOS / Linux): sh join.sh\ncd "$(dirname "$0")"\n')
        for name, parts, _ in made:
            o.write(f'cat {" ".join(parts)} > {name} && echo "{name}: $(wc -c < {name}) bytes"\n')
        o.write('command -v shasum >/dev/null && shasum -a 256 -c *.sha256\n')
    with open(os.path.join(REL, 'join.bat'), 'w', newline='\r\n') as o:
        o.write('@echo off\nrem Joins the parts back into the films (Windows): double-click join.bat\ncd /d "%~dp0"\n')
        for name, parts, _ in made:
            o.write(f'copy /b {"+".join(parts)} {name}\n')
        o.write('echo Done.\npause\n')
    with open(os.path.join(REL, 'README.md'), 'w') as o:
        o.write('# The film\n\nGitHub limits files to 100 MB, so each film is split into parts. To get the films back, '
                'download this folder and run `join.sh` (macOS / Linux) or double-click `join.bat` (Windows). The '
                'result is byte-for-byte the original file; the `.sha256` files are there to check it.\n\n')
        for name, parts, size in made:
            o.write(f'- **{name}** ({size / 2 ** 20:.0f} MB, 1920x1080, 24 fps, HEVC + AAC): '
                    f'{"no burned-in subtitles, for your own voice-over and captions" if "no_sub" in name else "with his lines as burned-in subtitles"}\n')


def web(dst):
    """H.264 in fragmented-MP4 pieces of ~30 s (at most ~12 MB at the 3 Mbit/s ceiling), their HLS playlist (as
    playlist.txt: a type any host serves) and manifest.json listing them. The page streams the pieces and joins them
    for its download button (init.mp4 followed by every piece is itself one playable MP4). About 185 MiB in all: under
    the 200 MiB a phone app will save, and the 256 MB a page may hold."""
    import json
    v, a = src(True)                    # the clean film: the page shows his lines as captions it can turn off
    os.makedirs(dst, exist_ok=True)
    ff(*video_in(v), '-i', a, '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-preset', 'slow', '-b:v', '1900k',
       '-maxrate', '3000k', '-bufsize', '6000k', '-profile:v', 'high', '-pix_fmt', 'yuv420p', '-g', '48',
       '-keyint_min', '48', '-sc_threshold', '0', '-c:a', 'aac', '-b:a', '128k', '-f', 'hls', '-hls_time', '30',
       '-hls_playlist_type', 'vod', '-hls_segment_type', 'fmp4', '-hls_fmp4_init_filename', 'init.mp4',
       '-hls_segment_filename', os.path.join(dst, 'seg%03d.mp4'), os.path.join(dst, 'film.m3u8'))
    segs = []
    dur = None
    for ln in open(os.path.join(dst, 'film.m3u8')).read().splitlines():
        if ln.startswith('#EXTINF:'):
            dur = float(ln[8:].split(',')[0])
        elif ln and not ln.startswith('#'):
            segs.append({'file': ln, 'dur': dur, 'bytes': os.path.getsize(os.path.join(dst, ln))})
    os.replace(os.path.join(dst, 'film.m3u8'), os.path.join(dst, 'playlist.txt'))
    init = os.path.getsize(os.path.join(dst, 'init.mp4'))
    total = init + sum(s['bytes'] for s in segs)
    man = {'init': 'init.mp4', 'segments': segs, 'bytes': total, 'seconds': round(sum(s['dur'] for s in segs), 2),
           'filename': NAME + '_no_subtitles.mp4'}
    with open(os.path.join(dst, 'manifest.json'), 'w') as fh:
        json.dump(man, fh)
    big = [s['file'] for s in segs if s['bytes'] > 15_000_000]
    print('web:', len(segs), 'pieces,', total // 2 ** 20, 'MiB', ('OVER 15 MB: ' + str(big)) if big else '')


def preview():
    v, a = src(False)
    out = os.path.join(OUT, 'preview_small.mp4')
    dur = 766.0
    budget = 29.0 * 8 * 2 ** 20 / dur / 1000 - 48          # kbit/s of video that fits 29 MiB with 48 kbit/s audio
    for p in (1, 2):
        ff(*video_in(v), '-i', a, '-map', '0:v', '-map', '1:a', '-vf', 'scale=854:480:flags=lanczos', '-c:v', 'libx264',
           '-preset', 'slow', '-b:v', f'{int(budget)}k', '-pass', p, '-passlogfile', out + '.log',
           *(['-an', '-f', 'null', '/dev/null'] if p == 1 else
             ['-c:a', 'aac', '-b:a', '48k', '-ac', '1', '-movflags', '+faststart', '-shortest', out]))
    print(out, os.path.getsize(out) / 2 ** 20, 'MiB')


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'master'
    if what == 'master':
        master()
    elif what == 'git':
        git_release()
    elif what == 'web':
        web(sys.argv[2])
    elif what == 'preview':
        preview()
