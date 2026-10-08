"""Countries Battle Royale: the edit (video time to simulation time, slow motion for the final elimination, the camera)
and the render: 1080x1920 at 60 fps in 1-second segments (resumable, rendered by 4 processes), then the sound and
the YouTube encode.

python main.py --out ../output                    # render everything, then sound and encode
python main.py --out ../preview --stills 60,900   # a few frames as PNG
python main.py --out ../output --encode-only      # redo the sound and the encode
"""
import argparse
import math
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

import hud as HD
import render as RD
import sim as SIM
from writer import Writer, ffmpeg_exe

FPS = 60
SEED = int(os.environ.get('ROYALE_SEED', '1485'))
TITLE = 'Countries Battle Royale: Last One Inside Wins'
TAIL = 6.2            # seconds of celebration after the last elimination


def window(x, a, b, ramp):
    """1 inside [a, b], easing to 0 over `ramp` outside it."""
    up = np.clip((x - (a - ramp)) / ramp, 0, 1)
    dn = np.clip(((b + ramp) - x) / ramp, 0, 1)
    up, dn = up * up * (3 - 2 * up), dn * dn * (3 - 2 * dn)
    return np.minimum(up, dn)


class Timeline:
    def __init__(self, st):
        self.st = st
        t_end = (len(st.t) - 2) / SIM.FPS
        dt = 1 / 1200
        g = np.arange(0, t_end, dt)
        speed = np.ones_like(g)
        te = [t for t, i in st.elims]
        # quiet stretches (no one going out for a while) run a little faster
        for a, b in zip([0.0] + te[:-4], te[:-3]):
            if b - a > 2.3:
                speed = np.maximum(speed, 1 + 0.32 * window(g, a + 0.7, b - 0.7, 0.45))
        # the last elimination in slow motion
        tw = st.t_win
        w = window(g, tw - 0.42, tw + 0.2, 0.32)
        speed = speed * (1 - w) + 0.26 * w
        vt = np.concatenate([[0.0], np.cumsum(dt / speed[:-1])])
        self.g, self.vt = g, vt
        self.vt_win = float(np.interp(tw, g, vt))
        self.duration = self.vt_win + TAIL
        self.n_frames = int(round(self.duration * FPS))
        # where the last one goes out, for the camera
        _, x, y, vx, vy, r = st.out_state[st.elims[-1][1]]
        self.exit_pt = np.array([x, y])
        self.shakes = [(t, 3.0) for t, i in st.elims[:-1]] + [(tw, 9.0)]

    def at(self, tv):
        """(sim time, seconds since the win in video time)."""
        if tv <= self.vt[-1]:
            ts = float(np.interp(tv, self.vt, self.g))
        else:
            ts = float(self.g[-1])
        return ts, tv - self.vt_win

    def video_time(self, ts):
        return float(np.interp(ts, self.g, self.vt))

    def camera(self, ts, tc):
        tw = self.st.t_win
        k = float(window(np.array([ts]), tw - 0.75, tw + 0.15, 0.45)[0])
        if tc > 0:
            k *= max(0.0, 1 - tc / 0.9)
        c = self.exit_pt * 0.42 * k
        zoom = 1 + 0.24 * k
        sx = sy = 0.0
        for t0, amp in self.shakes:
            d = ts - t0
            if 0 <= d < 0.35:
                e = amp * math.exp(-d / 0.09)
                sx += e * math.sin(d * 97 + t0 * 13)
                sy += e * math.cos(d * 83 + t0 * 7)
        return {'c': c, 'zoom': zoom, 'shake': np.array([sx, sy])}


class Show:
    def __init__(self, seed=SEED):
        t0 = time.time()
        self.st = RD.Story(SIM.load(seed))
        self.r = RD.Renderer(self.st)
        self.tl = Timeline(self.st)
        self.hud = HD.HUD(self.st, self.r, self.tl)
        self.n = self.tl.n_frames
        print(f'[show] seed {seed}: {self.n} frames ({self.tl.duration:.1f}s), winner '
              f'{self.st.names[self.st.winner]}, built in {time.time() - t0:.1f}s', flush=True)

    def winner_state(self, ts, tc):
        """The winner glides to the middle of the ring and grows during the celebration."""
        st = self.st
        pos, vel, rad, ang, gap = st.sample(ts)
        i = st.winner
        k = RD.ease((tc - 0.25) / 0.9)
        p = pos[i] * (1 - k) + np.array([0.0, 0.02]) * k
        r = rad[i] * (1 - k) + 0.5 * k
        return p, r

    def frame(self, k):
        tv = k / FPS
        ts, tc = self.tl.at(tv)
        cam = self.tl.camera(ts, tc)
        R, st = self.r, self.st
        frame = R.bg.copy()
        col = R.ring_color(ts)
        frame += (np.clip(1.05 - R.bg_d, 0, 1) ** 2 * 0.07)[..., None] * col
        glow = np.zeros((RD.H // 4, RD.W // 4, 3), np.float32)
        won = tc >= 0
        if won:
            p, r = self.winner_state(ts, tc)
            center = R.to_screen(p, cam)
            self.hud.celebration_back(frame, tc, center, r * RD.S * cam['zoom'])
        R.draw_ring(frame, glow, ts, cam)
        R.trails(frame, glow, ts, cam)
        R.draw_balls(frame, ts, cam, winner_override=True if won else None)
        R.sparks(frame, glow, ts, cam)
        if won:
            amt, nx, ny = st.squash(ts, st.winner) if tc < 0.3 else (0.0, 1.0, 0.0)
            bob = 0.0 if tc < 1.2 else 6 * math.sin((tc - 1.2) * 3.2)
            R.draw_ball(frame, st.codes[st.winner], center + np.array([0, bob]), r * RD.S * cam['zoom'],
                        look=(0.0, 0.0), mood='happy', squash=(amt, nx, ny))
        R.draw_board(frame, ts, alpha=1.0 - 0.75 * (cam['zoom'] - 1) / 0.24)
        R.draw_out_balls(frame, ts, cam)
        R.bursts(frame, glow, ts)
        R.finish_glow(frame, glow)
        self.hud.names(frame, ts, cam)
        self.hud.banners(frame, ts)
        self.hud.title(frame, tv, ts, tc if won else None)
        self.hud.counter(frame, tv, ts, tc if won else None)
        if won:
            self.hud.crown(frame, tc, center + np.array([0, 0 if tc < 1.2 else 6 * math.sin((tc - 1.2) * 3.2)]),
                           r * RD.S * cam['zoom'])
            self.hud.confetti(frame, tc, R.A.colors[st.codes[st.winner]])
            self.hud.comment_cta(frame, tc)
        # one soft flash as the last one goes out
        if -0.02 < tc < 0.25:
            frame += 0.22 * (1 - max(0.0, tc) / 0.25)
        return RD.to8(frame)


# ----------------------------------------------------------------------------------------------------------------

def _render_segments(args):
    out_dir, segs, seed = args
    show = Show(seed)
    for s in segs:
        path = os.path.join(out_dir, 'seg', f's{s:03d}.mp4')
        if os.path.exists(path):
            continue
        w = Writer(path, RD.W, RD.H, FPS)
        t0 = time.time()
        for k in range(s * FPS, min(show.n, (s + 1) * FPS)):
            w.write(show.frame(k))
        w.close()
        print(f'[seg {s:03d}] {time.time() - t0:.1f}s', flush=True)


def render_all(out_dir, seed, procs=4):
    from multiprocessing import Pool
    os.makedirs(os.path.join(out_dir, 'seg'), exist_ok=True)
    show = Show(seed)
    nseg = (show.n + FPS - 1) // FPS
    todo = [s for s in range(nseg) if not os.path.exists(os.path.join(out_dir, 'seg', f's{s:03d}.mp4'))]
    print(f'[render] {len(todo)} of {nseg} segments to do', flush=True)
    chunks = [todo[i::procs] for i in range(procs)]
    with Pool(procs) as p:
        p.map(_render_segments, [(out_dir, c, seed) for c in chunks if c])
    lst = os.path.join(out_dir, 'seg', 'list.txt')
    with open(lst, 'w') as f:
        for s in range(nseg):
            f.write(f"file 's{s:03d}.mp4'\n")
    video = os.path.join(out_dir, 'video.mp4')
    subprocess.run([ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst, '-c', 'copy',
                    video], check=True)
    return show, video


def encode_youtube(video_in, wav_in, out, workdir, bitrate='12M'):
    """Two-pass H.264 following YouTube's recommended upload settings (High profile, closed GOP of half the frame
    rate, 2 B-frames, BT.709 tags) muxed with 384 kbps AAC at 48 kHz, moov atom up front."""
    gop = FPS // 2
    common = ['-c:v', 'libx264', '-preset', 'slower', '-profile:v', 'high', '-level', '4.2',
              '-b:v', bitrate, '-maxrate', '20M', '-bufsize', '24M', '-pix_fmt', 'yuv420p', '-r', str(FPS),
              '-x264-params', f'keyint={gop}:min-keyint={gop}:scenecut=0:open-gop=0:bframes=2',
              '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv']
    log = os.path.join(workdir, 'x264pass')
    subprocess.run([ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', video_in] + common +
                   ['-pass', '1', '-passlogfile', log, '-an', '-f', 'null', os.devnull], check=True)
    subprocess.run([ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', video_in, '-i', wav_in, '-map', '0:v:0', '-map',
                    '1:a:0'] + common +
                   ['-pass', '2', '-passlogfile', log, '-c:a', 'aac', '-b:a', '384k', '-ar', '48000', '-ac', '2',
                    '-movflags', '+faststart', '-shortest', '-metadata', f'title={TITLE}', out], check=True)


def cues(show):
    """Everything the sound needs, in video seconds."""
    st, tl = show.st, show.tl
    c = {'duration': tl.duration, 'win': tl.vt_win,
         'elims': [(tl.video_time(t), int(i), int(st.place[i]), float(st.out_state[i][1]))
                   for t, i in st.elims],
         'lands': [(tl.video_time(t + RD.OUT_T), int(st.place[i])) for t, i in st.elims],
         'milestones': [(tl.video_time(t), label) for t, label, sub in show.hud.milestones],
         'impacts': []}
    for row in st.impacts:
        t, a, b, imp, x, y = row[:6]
        if t > st.t_win + 0.3:
            continue
        c['impacts'].append((tl.video_time(float(t)), int(a), int(b), float(imp), float(x), float(y),
                             st.n_alive(float(t))))
    # how fast time runs in each frame (slow motion pitches the impacts down)
    c['speed'] = [float(np.interp(tl.at(k / FPS)[0] + 1e-3, tl.g, np.gradient(tl.g) / np.gradient(tl.vt)))
                  for k in range(0, tl.n_frames, 6)]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='../output')
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--stills', default=None)
    ap.add_argument('--encode-only', action='store_true')
    ap.add_argument('--procs', type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.stills:
        show = Show(a.seed)
        for k in [int(x) for x in a.stills.split(',')]:
            t0 = time.time()
            img = show.frame(k)
            Image.fromarray(img).save(os.path.join(a.out, f'still_{k:05d}.png'))
            print(f'frame {k} ({k / FPS:.2f}s) in {time.time() - t0:.2f}s', flush=True)
        return
    if a.encode_only:
        show = Show(a.seed)
        video = os.path.join(a.out, 'video.mp4')
    else:
        show, video = render_all(a.out, a.seed, a.procs)
    import json
    import audio as AU
    wav = os.path.join(a.out, 'sound.wav')
    c = cues(show)
    with open(os.path.join(a.out, 'cues.json'), 'w') as fh:
        json.dump(c, fh)
    AU.build(c, wav)
    final = os.path.join(a.out, 'country-royale.mp4')
    encode_youtube(video, wav, final, a.out)
    print('done:', final, flush=True)


if __name__ == '__main__':
    main()
