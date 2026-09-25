"""Render "Minecraft Mob Marble Race": eight mobs as marbles on a marble run down a cliff, three elimination rounds
and a final over lava, with the broadcast HUD; then the music and sound design and the YouTube encode.

Usage:
  python main.py --out OUT_DIR [--preview] [--ss 1.5] [--every N] [--stills] [--from F --to F]
                 [--no-audio] [--cues-only] [--encode-only]
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
from PIL import Image

import course as C
import director as D
import effects as FX
import props as PR
import race as RC
import scene as SC
import timeline as T

W, H = 1080, 1920
FPS = T.FPS


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return 'ffmpeg'


def encode_youtube(video_in, wav_in, out, workdir, bitrate='12M'):
    """Two-pass H.264 encode following YouTube's recommended upload settings (High profile, closed GOP of
    half the frame rate, 2 B-frames, BT.709 tags) muxed with 384 kbps AAC at 48 kHz, moov atom up front."""
    common = ['-c:v', 'libx264', '-preset', 'slower', '-profile:v', 'high', '-level', '4.1',
              '-b:v', bitrate, '-maxrate', '20M', '-bufsize', '24M', '-pix_fmt', 'yuv420p',
              '-x264-params', 'keyint=15:min-keyint=15:scenecut=0:open-gop=0:bframes=2',
              '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv']
    log = os.path.join(workdir, 'x264pass')
    subprocess.run([ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', video_in] + common +
                   ['-pass', '1', '-passlogfile', log, '-an', '-f', 'null', os.devnull], check=True)
    subprocess.run([ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', video_in, '-i', wav_in, '-map', '0:v:0', '-map',
                    '1:a:0'] + common +
                   ['-pass', '2', '-passlogfile', log, '-c:a', 'aac', '-b:a', '384k', '-ar', '48000', '-ac', '2',
                    '-movflags', '+faststart', '-shortest', '-metadata', 'title=Minecraft Mob Marble Race', out],
                   check=True)


# ---------------------------------------------------------------------------------------------
# pass 1: run the race on the edit's schedule; everything else (camera, HUD, sound) is read off it
# ---------------------------------------------------------------------------------------------
HIT_KIND = {'fence': 'peg', 'marble': 'marble', 'ice': 'ice', 'slime': 'slime', 'trapdoor': 'wood', 'planks': 'wood',
            'gold': 'metal', 'iron': 'metal', 'piston': 'wood', 'tnt': 'wood'}


def record(course):
    race = RC.Race(course, T.LINEUP, seed=T.SEED)
    rec = []
    t_prev = 0.0
    for n in T.schedule():
        ev = race.step(n)
        st = race.state()
        st['dt'] = st['t'] - t_prev
        t_prev = st['t']
        hits = {}
        events = []
        for e in ev:
            if e[0] == 'hit':
                kind = HIT_KIND.get(e[2], 'stone')
                h = hits.setdefault(kind, [0, 0.0, 0.0])
                h[0] += 1
                h[1] = max(h[1], e[3])
                h[2] += e[4][0]
            else:
                events.append(e)
        for k, h in hits.items():
            h[2] /= h[0]
        st['hits'] = hits
        st['events'] = events
        st['standings'] = race.standings()
        rec.append(st)
    return rec


def jsonable(e):
    out = []
    for x in e:
        if isinstance(x, (tuple, list)):
            out.append([float(v) for v in x])
        elif isinstance(x, (np.floating, float)):
            out.append(float(x))
        elif isinstance(x, (np.integer, int)):
            out.append(int(x))
        else:
            out.append(str(x))
    return out


def advance_fx(fx, st, i, story, winner):
    """The particles for frame i: the race's events, the winner's confetti and fireworks, then a step."""
    for e in st['events']:
        fx.event(e, st)
    if winner is not None and story['win'] is not None:
        k = i - story['win']
        if k == 0 or k in (6, 18, 30, 44):
            w = [m for m in st['marbles'] if m['name'] == winner][0]
            if k == 0:
                fx.confetti(w['x'], w['z'])
            else:
                col = FX.CONFETTI[(k // 6) % len(FX.CONFETTI)]
                fx.firework(w['x'] + (-3.0 if (k // 12) % 2 == 0 else 3.0), w['z'] + 5.0 + (k % 7), col)
    fx.step(st['dt'], st)


def render_frame(r, course, st, cam, marbles, fx, preview):
    t = st['t']
    vox_m = marbles.instances(st, t)
    bits, puffs, flashes, lights = fx.render_data()
    vox = np.concatenate([vox_m, bits]) if len(bits) else vox_m
    v, i = PR.dynamic_mesh(course, st, t, t)
    r.set_ground_region(v, i)
    rows = PR.lava_lights(course, t) + lights
    light_arr = np.array(rows[:16], np.float32) if rows else np.zeros((0, 8), np.float32)
    r.dof = None
    tgt = np.array(cam['target'])
    r.render(cam, voxels=vox, props={}, fx={'puffs': puffs, 'flashes': flashes, 'lights': light_arr},
             near_center=tgt, giants=None, glow=3.0, sculk=(0.0, 0.0))
    img = r.finish()
    if preview:
        img = np.array(Image.fromarray(img).resize((W, H), Image.BILINEAR))
    return img


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--preview', action='store_true')
    ap.add_argument('--ss', type=float, default=1.5, help='supersampling factor (1.5 = delivered quality)')
    ap.add_argument('--every', type=int, default=1, help='render every Nth frame only (stills)')
    ap.add_argument('--stills', action='store_true', help='write JPEG stills instead of a video')
    ap.add_argument('--from', dest='f0', type=int, default=0)
    ap.add_argument('--to', dest='f1', type=int, default=10 ** 9)
    ap.add_argument('--crf', type=int, default=15)
    ap.add_argument('--no-audio', action='store_true')
    ap.add_argument('--no-hud', action='store_true')
    ap.add_argument('--cues-only', action='store_true',
                    help='sound work: write OUT/cues.json and OUT/audio.wav without rendering a frame')
    ap.add_argument('--encode-only', action='store_true',
                    help='rebuild audio from OUT/cues.json and redo the final encode of OUT/video_noaudio.mp4')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.encode_only:
        import audio
        with open(os.path.join(args.out, 'cues.json')) as fh:
            d = json.load(fh)
        wav = os.path.join(args.out, 'audio.wav')
        audio.build(d, wav)
        final = os.path.join(args.out, 'final.mp4')
        encode_youtube(os.path.join(args.out, 'video_noaudio.mp4'), wav, final, args.out)
        print('final:', final)
        return

    course = C.build()
    t0 = time.time()
    rec = record(course)
    alive = [m['name'] for m in rec[-1]['marbles'] if m['out'] is None]
    winner = alive[0] if len(alive) == 1 else None
    print(f'[record] {len(rec)} frames in {time.time() - t0:.0f}s; winner {winner}', flush=True)
    cams, track = D.cameras(rec, course, winner)
    import hud as HUDM
    story = HUDM.story(rec, course, winner)
    hud = HUDM.HUD(W, H)

    r = None
    if not args.cues_only:
        if args.preview:
            r = SC.make_renderer(course, width=W // 2, height=H // 2, ss=1.0, shadow_res=2048)
        else:
            r = SC.make_renderer(course, width=W, height=H, ss=args.ss, shadow_res=4096)
    marbles = PR.Marbles(T.LINEUP)
    fx = FX.Effects(course)
    video_path = os.path.join(args.out, 'video_noaudio.mp4')
    enc = None
    if not args.stills and not args.cues_only:
        cmd = [ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
               '-r', str(FPS), '-i', '-', '-vf', 'scale=out_color_matrix=bt709:out_range=tv',
               '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
               '-c:v', 'libx264', '-preset', 'slow', '-crf', str(args.crf),
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', video_path]
        enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    ban_at = {f0: [kind, name] for (f0, dur, kind, name, extra) in story['banners']}
    cues = []
    t_start = time.time()
    n_out = 0
    for i, st in enumerate(rec):
        advance_fx(fx, st, i, story, winner)
        cue = {'t': st['t'], 'dt': st['dt'], 'events': [jsonable(e) for e in st['events']], 'hits': st['hits'],
               'cam': list(cams[i]['eye']), 'tgt': list(cams[i]['target']),
               'speed': [float(np.hypot(m['vx'], m['vz'])) if m['out'] is None else 0.0 for m in st['marbles']],
               'names': [m['name'] for m in st['marbles']], 'pos': [[m['x'], m['z']] for m in st['marbles']],
               'hud': dict(story['cues'].get(i, {}), **({'banner': ban_at[i]} if i in ban_at else {}))}
        cues.append(cue)
        if args.cues_only or not (args.f0 <= i < args.f1) or i % args.every:
            continue
        img = render_frame(r, course, st, cams[i], marbles, fx, args.preview)
        if not args.no_hud:
            img = hud.draw(img, HUDM.state_at(i, story, rec))
        if enc is not None:
            enc.stdin.write(np.ascontiguousarray(img).tobytes())
        else:
            Image.fromarray(img).save(os.path.join(args.out, f'{i:04d}.jpg'), quality=90)
        n_out += 1
        if n_out % 15 == 0:
            el = time.time() - t_start
            print(f'  frame {i}/{len(rec)}  {el:.0f}s  ({el / n_out:.2f}s/frame)', flush=True)
    if enc is not None:
        enc.stdin.close()
        enc.wait()
    meta = {'fps': FPS, 'frames': cues, 'lineup': T.LINEUP, 'winner': winner, 'go': story['go'],
            'win': story['win'], 'rounds': story['rounds'],
            'pistons': [[p['x'], p['z']] for p in course.pistons], 'tnt': [[t['x'], t['z']] for t in course.tnt],
            'lava': [list(b) for b in course.lava] + [list(course.lake)]}
    with open(os.path.join(args.out, 'cues.json'), 'w') as fh:
        json.dump(meta, fh)
    print(f'video frames: {n_out}, render time {time.time() - t_start:.0f}s', flush=True)
    if not args.no_audio and (enc is not None or args.cues_only) and args.every == 1 and args.f0 == 0:
        import audio
        wav = os.path.join(args.out, 'audio.wav')
        audio.build(meta, wav)
        if enc is not None:
            final = os.path.join(args.out, 'final.mp4')
            encode_youtube(video_path, wav, final, args.out)
            print('final:', final)


if __name__ == '__main__':
    main()
