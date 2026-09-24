"""Render "Hydraulic Press vs Minecraft": a cold open of the bedrock test at full pressure, a rewind, then nine
blocks under the press one after another (grass, glass, melon, slime, TNT, chest, diamond, obsidian, bedrock) with
the Minecraft-style HUD; then the sound design and the YouTube encode.

Usage:
  python main.py --out OUT_DIR [--preview] [--ss 1.5] [--shots a,b] [--every N] [--stills]
                 [--no-audio] [--encode-only]
"""
import argparse
import io
import json
import os
import subprocess
import time

import numpy as np
from PIL import Image

import press as P
import scene as SC
import timeline as T
from hud import HUD

W, H = 1080, 1920
FPS = T.FPS
KINDS = T.KINDS
BOSS_COL = {'grass': 'green', 'glass': 'white', 'melon': 'red', 'slime': 'green', 'tnt': 'red', 'chest': 'yellow',
            'diamond': 'blue', 'obsidian': 'purple', 'bedrock': 'pink'}
TITLES = {'grass': ('CRUSHED!', 'red'), 'glass': ('SHATTERED!', 'aqua'), 'melon': ('SPLAT!', 'red'),
          'slime': ('SQUASHED!', 'green'), 'tnt': ('BOOM!', 'gold'), 'chest': ('JACKPOT!', 'yellow'),
          'diamond': ('CRUSHED!', 'aqua'), 'obsidian': ('CRUSHED!', 'light_purple')}
BLOCK_C = np.array([0.0, 0.0, SC.Z0 + 2.0])
SPARK_COL = np.array([1.0, 0.62, 0.22]) * 7.0


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
                    '-movflags', '+faststart', '-shortest', '-metadata', 'title=Hydraulic Press vs Minecraft',
                    out], check=True)


def smooth01(u):
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3 - 2 * u)


def compact(ev):
    """Events of a frame as plain JSON (name -> dict with a count and scalars)."""
    out = {}
    for k, v in ev.items():
        if isinstance(v, dict):
            out[k] = {kk: (vv if isinstance(vv, (int, float, str)) else None) for kk, vv in v.items()}
        elif isinstance(v, list):
            out[k] = {'n': int(v[0]), 'speed': float(v[1])}
    return out


def snapshot(w):
    it = w.item
    return {'t': float(w.t), 'pressure': float(w.pressure), 'integrity': float(w.integrity),
            'press_health': float(getattr(w, 'press_health', 1.0)), 'cur': int(w.cur),
            'kind': it['kind'] if it else None, 'state': it['state'] if it else None, 'ram_z': float(w.ram_z),
            'ram_vz': float(w.ram_vz), 'ram_broken': bool(w.ram_broken), 'shake': float(w.shake),
            'flash': float(w.flash), 'n_frags': len(w.frags), 'n_bits': len(w.bits.p),
            'n_sparks': len(w.sparks.p)}


# ---------------------------------------------------------------------------------------------
# pass 1: the whole story on the edit's schedule without rendering; the HUD's story comes from it
# ---------------------------------------------------------------------------------------------
def advance(w, shot, f, scale):
    """Step the story one video frame; a shot that brings on the next block sweeps the platen as it cuts in."""
    if f == 0 and shot.get('clear'):
        w.clear_platen()
    return w.step_frame(scale)


def analyse(sched):
    w = P.Press(T.plan())
    shots = T.shots()
    recs = []
    t0 = time.time()
    for i, (si, f, scale) in enumerate(sched):
        ev = compact(advance(w, shots[si], f, scale))
        s = snapshot(w)
        s['ev'] = ev
        recs.append(s)
        if i % 120 == 0:
            print(f'  [analyse] frame {i}/{len(sched)}  sim t={w.t:.2f}  ({time.time() - t0:.0f}s)', flush=True)
    return recs


def story(recs):
    """Frame numbers of what the HUD does: placements, breaks, titles, the press bar, the toast."""
    n = len(recs)
    cuts = dict(zip([s['name'] for s in T.shots()], T.cut_frames()))
    place = {}
    brk = {}
    bounce = None
    hoses = None
    press_break = None
    for i, r in enumerate(recs):
        ev = r['ev']
        if 'place' in ev:
            place[ev['place']['index']] = i
        if 'break' in ev:
            brk[r['cur']] = i
        if 'bounce' in ev:
            bounce = i
        if 'hoses' in ev:
            hoses = i
        if 'press_break' in ev:
            press_break = i
    titles = []
    for k, fb in sorted(brk.items()):
        kind = KINDS[k]
        text, col = TITLES[kind]
        nxt = place.get(k + 1, n)
        dur = float(np.clip((nxt - fb) / FPS + 0.1, 0.7, 1.35))
        tons = P.MATS[kind]['tons']
        titles.append(dict(f=fb, text=text, col=col, sub=f'{tons:,} TONS', dur=dur))
    if bounce is not None:
        titles.append(dict(f=bounce, text='BOING!', col='green', sub=None, dur=0.75))
    survivor = None
    if press_break is not None:
        survivor = cuts['survivor'] + T.sec(0.2)
        titles.append(dict(f=survivor, text='UNBREAKABLE', col='light_purple', sub='100,000 TONS', dur=9.0, y=470))
        brk[8] = press_break
    titles.sort(key=lambda d: d['f'])
    # displayed integrity and pressure (smoothed; the count-up of the pressure is quick)
    integ = np.ones(n)
    tons = np.zeros(n)
    for i, r in enumerate(recs):
        pi = integ[i - 1] if i else 1.0
        integ[i] = r['integrity'] if r['integrity'] > pi or 'place' in r['ev'] else pi + (r['integrity'] - pi) * 0.45
        pt = tons[i - 1] if i else 0.0
        tons[i] = r['pressure'] if (r['pressure'] < pt or 'place' in r['ev']) else pt + (r['pressure'] - pt) * 0.5
    return dict(place=place, brk=brk, bounce=bounce, hoses=hoses, press_break=press_break, titles=titles,
                survivor=survivor, integ=integ, tons=tons)


def hud_state(i, st, recs):
    r = recs[i]
    s = {'vignette': 0.28}
    cur = max([k for k, f in st['place'].items() if f <= i], default=-1)
    # the boss bar
    if cur >= 0:
        kind = KINDS[cur]
        fp = st['place'][cur]
        pop = (i - fp) / FPS
        bars = []
        fb = st['brk'].get(cur)
        if kind == 'bedrock':
            bars.append(dict(name='Bedrock', frac=1.0, col=BOSS_COL[kind], pop=pop))
            fh = st['press_break']
            contact = next((j for j in range(fp, len(recs)) if recs[j]['ram_z'] <= P.ZC + 0.05), None)
            if contact is not None and i >= contact:
                a = smooth01((i - contact) / 6.0)
                if fh is not None and i > fh:
                    a *= 1.0 - smooth01((i - fh - 12) / 8.0)
                ph = r['press_health'] if (fh is None or i < fh) else 0.0
                bars.append(dict(name='Hydraulic Press', frac=ph, col='red', alpha=a,
                                 pop=(i - contact) / FPS, shake=0.0 if fh is not None and i > fh else (1 - ph) ** 2))
        else:
            a = 1.0
            if fb is not None and i > fb:
                a = 1.0 - smooth01((i - fb - 10) / 6.0)
            frac = st['integ'][i] if (fb is None or i < fb) else 0.0
            bars.append(dict(name=HUD_LABELS[kind], frac=frac,
                             col=BOSS_COL[kind], alpha=a, pop=pop))
        s['boss'] = bars
    # the pressure
    fh = st['press_break']
    tons = st['tons'][i]
    if cur >= 0:
        kind = KINDS[cur]
        fb = st['brk'].get(cur)
        if kind == 'bedrock':
            u = min(1.0, tons / 100000.0)
            danger = min(1.0, u * 1.6)
            pulse = 0.0
        else:
            danger = float(np.clip((1.0 - st['integ'][i] - 0.35) / 0.6, 0, 1)) if (fb is None or i < fb) else 1.0
            pulse = float(np.exp(-(i - fb) / 5.0)) if (fb is not None and i >= fb) else 0.0
        a = 1.0
        if fh is not None and i > fh:
            a = 1.0 - smooth01((i - fh - 6) / 10.0)
        s['xp'] = dict(tons=tons, danger=danger, pulse=pulse, alpha=a)
    else:
        s['xp'] = dict(tons=0.0, danger=0.0)
    # the hotbar
    sel = 0.0
    for k in sorted(st['place']):
        fp = st['place'][k]
        if i >= fp:
            sel = (k - 1) + smooth01((i - fp) / 4.0) if k else 0.0
    crossed = {k: (i - f) / FPS for k, f in st['brk'].items() if i >= f and k < 8}
    glint = (i - st['survivor']) / FPS if st['survivor'] is not None and i >= st['survivor'] else None
    bump = (i - st['place'][cur]) / FPS if cur >= 0 else None
    s['hotbar'] = dict(sel=sel, crossed=crossed, glint=glint, bump=bump)
    # titles and the toast
    for d in st['titles']:
        t = (i - d['f']) / FPS
        if 0 <= t < d['dur']:
            s['title'] = dict(text=d['text'], sub=d['sub'], col=d['col'], t=t, dur=d['dur'], y=d.get('y'))
    if st['survivor'] is not None and i >= st['survivor'] + T.sec(0.5):
        s['toast'] = (i - st['survivor'] - T.sec(0.5)) / FPS
    # flashes and the alarm
    s['flash'] = min(0.85, r['flash'] * 1.3)
    if st['hoses'] is not None and i >= st['hoses'] and (fh is None or i < fh + 3):
        s['red'] = 0.18 * (0.5 + 0.5 * np.sin(2 * np.pi * 2.2 * (i - st['hoses']) / FPS))
    return s


HUD_LABELS = {k: v['label'] for k, v in P.PA.make_items().items()}


# ---------------------------------------------------------------------------------------------
# cameras, fx
# ---------------------------------------------------------------------------------------------
class Shaker:
    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.ph = rng.uniform(0, 2 * np.pi, (3, 3))
        self.fr = np.array([7.3, 11.1, 17.9])

    def offset(self, t, amp):
        o = np.zeros(3)
        for a in range(3):
            o[a] = sum(np.sin(2 * np.pi * self.fr[k] * t + self.ph[a, k]) / (k + 1) for k in range(3))
        return o * amp


def shake_kick(ev):
    k = 0.0
    if 'break' in ev:
        k += {'grass': 0.8, 'glass': 0.7, 'melon': 0.9, 'slime': 1.0, 'chest': 1.0, 'diamond': 1.6,
              'obsidian': 2.2}.get(ev['break'].get('kind'), 1.0)
    if 'bounce' in ev:
        k += 1.2
    if 'explode' in ev:
        k += 4.0
    if 'press_break' in ev:
        k += 7.0
    if 'hoses' in ev:
        k += 1.5
    if 'land' in ev:
        k += min(0.3, 0.004 * ev['land']['n'])
    return k


def alarm_lights(i, st, fh):
    """The redstone lamps flash red once the hoses give."""
    if st is None or st['hoses'] is None or i < st['hoses'] or (fh is not None and i > fh + T.sec(2.0)):
        return []
    ph = 0.5 + 0.5 * np.sin(2 * np.pi * 2.2 * (i - st['hoses']) / FPS)
    return [[lx, ly - 3.0, lz, 14.0 * ph, 44.0, 1.0, 0.06, 0.03] for (lx, ly, lz) in SC.LAMPS] + \
        [[0.0, -12.0, 20.0, 5.0 * ph, 30.0, 1.0, 0.05, 0.03]]


def render_frame(r, w, cam, preview, extra_lights=(), focus=None):
    vox, giants, fx = w.instances()
    eye = np.asarray(cam['eye'], float)
    nb = giants[-1][1] if giants else 0
    if len(vox) > nb:
        # debris flying at the lens shrinks away instead of blacking out the frame
        tail = vox[nb:]
        dist = np.linalg.norm(tail['pos'].astype(np.float64) - eye, axis=1)
        u = np.clip((dist - 1.5) / 3.0, 0.0, 1.0)
        tail = tail[u > 0].copy()
        tail['scale'] *= (u * u * (3 - 2 * u))[u > 0]
        vox = np.concatenate([vox[:nb], tail])
    if len(extra_lights):
        fx['lights'] = np.concatenate([fx['lights'], np.array(extra_lights, np.float32)])
    f = focus if focus is not None else float(np.linalg.norm(np.asarray(cam['target']) - eye))
    r.dof = {'focus': f, 'k': 14.0, 'maxr': 10.0}
    r.streak_col = SPARK_COL
    r.render(cam, voxels=vox, props={}, fx=fx, near_center=SC.NEAR_CENTER, giants=giants, glow=4.0,
             sculk=(0.0, 0.0))
    img = r.finish()
    if preview:
        img = np.array(Image.fromarray(img).resize((W, H), Image.BILINEAR))
    return img


def rewind_fx(img, k, n):
    """VHS rewind look: washed out, a tracking band rolling up, a little horizontal tearing."""
    out = img.astype(np.float32)
    g = out @ np.array([0.299, 0.587, 0.114], np.float32)
    out = out * 0.45 + g[..., None] * 0.55
    out = out * np.array([0.92, 0.98, 1.08]) + 6.0
    rng = np.random.default_rng(100 + k)
    y0 = int((1.0 - k / max(1, n - 1)) * H * 1.2) % H
    band = slice(max(0, y0 - 60), min(H, y0 + 60))
    out[band] = out[band] * 0.7 + 70.0
    for _ in range(6):
        yy = int(rng.integers(0, H - 8))
        out[yy:yy + int(rng.integers(2, 8))] = np.roll(out[yy:yy + 8][:1], int(rng.integers(-40, 40)), axis=1) + 25.0
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--preview', action='store_true')
    ap.add_argument('--ss', type=float, default=1.5, help='supersampling factor (1.5 = delivered quality)')
    ap.add_argument('--shots', default=None, help='render only these shots (comma separated names, or "cold")')
    ap.add_argument('--every', type=int, default=1, help='render every Nth frame only (stills)')
    ap.add_argument('--stills', action='store_true', help='write JPEG stills instead of a video')
    ap.add_argument('--crf', type=int, default=15)
    ap.add_argument('--no-audio', action='store_true')
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

    shots = T.shots()
    sched = T.schedule()
    only = set(args.shots.split(',')) if args.shots else None

    # ---- pass 1
    t0 = time.time()
    recs = analyse(sched)
    st = story(recs)
    print(f'[analyse] {len(sched)} frames in {time.time() - t0:.0f}s; breaks {st["brk"]}; places {st["place"]}; '
          f'press break {st["press_break"]}', flush=True)
    with open(os.path.join(args.out, 'story.json'), 'w') as fh:
        json.dump({'brk': st['brk'], 'place': st['place'], 'titles': st['titles'], 'hoses': st['hoses'],
                   'press_break': st['press_break'], 'survivor': st['survivor']}, fh)

    # ---- renderer / encoder
    if args.cues_only:
        r = None
    elif args.preview:
        r = SC.make_renderer(width=W // 2, height=H // 2, ss=1.0, shadow_res=2048)
    else:
        r = SC.make_renderer(width=W, height=H, ss=args.ss, shadow_res=4096)
    hud = HUD(W, H)
    video_path = os.path.join(args.out, 'video_noaudio.mp4')
    enc = None
    if not args.stills and not args.cues_only:
        cmd = [ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
               '-r', str(FPS), '-i', '-', '-vf', 'scale=out_color_matrix=bt709:out_range=tv',
               '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
               '-c:v', 'libx264', '-preset', 'slow', '-crf', str(args.crf),
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', video_path]
        enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    cues = []
    n_out = [0]
    t_start = time.time()

    def emit(img, cue, tag):
        if enc is not None:
            enc.stdin.write(np.ascontiguousarray(img).tobytes())
        else:
            Image.fromarray(img).save(os.path.join(args.out, f'{tag}.jpg'), quality=90)
        cues.append(cue)
        n_out[0] += 1
        if n_out[0] % 15 == 0:
            el = time.time() - t_start
            print(f'  frame {n_out[0]}  {el:.0f}s  ({el / n_out[0]:.2f}s/frame)', flush=True)

    def wanted(name, f):
        return (only is None or name in only) and f % args.every == 0

    # ---- the cold open: the bedrock test at full pressure, shown first; then rewound
    cold = T.cold_plan()
    w = P.Press(cold)
    ts = cold['tests'][0]
    tau0, tau1 = T.COLD['tau']
    while w.t < tau0 - 1e-6:
        w.step_frame(1.0)
    nf = T.sec(T.COLD['dur'])
    n_rw = T.sec(T.REWIND)
    keep_idx = {int(round((nf - 1) * (1.0 - j / max(1, n_rw - 1)))) for j in range(n_rw)}
    kept = {}
    shaker = Shaker(11)
    acc = 0.0
    sh = T.cold_shot()
    hoses_f = None
    cold_hoses = ts['t0'] + P.event_offsets(ts)['hoses']
    for k in range(nf):
        ev = compact(w.step_frame((tau1 - tau0) / T.COLD['dur']))
        snap = snapshot(w)
        if hoses_f is None and w.t >= cold_hoses:
            hoses_f = k
        eye, tgt, fov = T.cam_path(sh['keys'], k / FPS)
        acc = acc * 0.86 + shake_kick(ev)
        off = shaker.offset(k / FPS, 0.1 * np.tanh(acc / 3.0) + 0.03 * w.shake)
        u = min(1.0, w.pressure / 100000.0)
        hs = {'boss': [dict(name='Bedrock', frac=1.0, col='pink'),
                       dict(name='Hydraulic Press', frac=snap['press_health'], col='red',
                            shake=(1 - snap['press_health']) ** 2)],
              'xp': dict(tons=w.pressure, danger=min(1.0, 0.65 + 1.5 * u)),
              'hotbar': dict(sel=8.0, crossed={j: 9.0 for j in range(8)}),
              'vignette': 0.3, 'flash': min(0.85, w.flash * 1.3)}
        if hoses_f is not None:
            hs['red'] = 0.18 * (0.5 + 0.5 * np.sin(2 * np.pi * 2.2 * (k - hoses_f) / FPS))
        cue = {'seg': 'cold', 'shot': 'cold', 'shot_frame': k, 'cut': k == 0, 'events': ev, 'scale': 1.0,
               'cam': eye.tolist(), 'hud': {}, **snap}
        if args.cues_only:
            cues.append(cue)
            continue
        if wanted('cold', k) or k in keep_idx:
            cam = {'eye': eye + off, 'target': tgt + off * 0.6, 'fov': fov, 'roll': float(off[0] * 0.6)}
            lights = alarm_lights(k, {'hoses': hoses_f}, None) if hoses_f is not None else []
            img = render_frame(r, w, cam, args.preview, lights)
            img = hud.draw(img, hs)
            if k in keep_idx:
                buf = io.BytesIO()
                Image.fromarray(img).save(buf, format='JPEG', quality=92)
                kept[k] = buf.getvalue()
            if wanted('cold', k):
                emit(img, cue, f'cold_{k:03d}')
    for j in range(n_rw):
        k = int(round((nf - 1) * (1.0 - j / max(1, n_rw - 1))))
        cue = {'seg': 'rewind', 'shot': 'rewind', 'shot_frame': j, 'cut': j == 0, 'events': {}, 'scale': -4.0,
               'hud': {}}
        if args.cues_only:
            cues.append(cue)
            continue
        if k not in kept or not wanted('cold', j):
            continue
        img = np.asarray(Image.open(io.BytesIO(kept[k])).convert('RGB'))
        img = rewind_fx(img, j, n_rw)
        img = hud.draw(img, {'rewind': j / FPS})
        emit(img, cue, f'rewind_{j:03d}')
    print(f'[render] cold open + rewind done ({time.time() - t_start:.0f}s)', flush=True)

    # ---- the main edit
    w = P.Press(T.plan())
    shaker = Shaker(3)
    acc = 0.0
    punch = 0.0
    prev_si = None
    for i, (si, f, scale) in enumerate(sched):
        shot = shots[si]
        ev = compact(advance(w, shot, f, scale))
        eye, tgt, fov = T.cam_path(shot['keys'], f / FPS)
        kick = shake_kick(ev)
        acc = acc * 0.86 + kick
        if 'break' in ev or 'explode' in ev or 'press_break' in ev or 'bounce' in ev:
            punch = 1.0
        punch *= 0.72
        amp = 0.1 * np.tanh(acc / 3.0) + 0.02 * w.shake
        off = shaker.offset(i / FPS, amp)
        hs = hud_state(i, st, recs)
        cue = {'seg': 'main', 'shot': shot['name'], 'shot_frame': f, 'cut': si != prev_si, 'events': ev,
               'scale': float(scale), 'cam': eye.tolist(), 'hud': {}, **snapshot(w)}
        for d in st['titles']:
            if d['f'] == i:
                cue['hud']['title'] = d['text']
        if st['survivor'] is not None and i == st['survivor'] + T.sec(0.5):
            cue['hud']['toast'] = True
        if args.cues_only:
            cues.append(cue)
        elif wanted(shot['name'], f):
            cam = {'eye': eye + off, 'target': tgt + off * 0.6, 'fov': fov * (1.0 - 0.05 * punch),
                   'roll': float(off[0] * 0.6)}
            img = render_frame(r, w, cam, args.preview, alarm_lights(i, st, st['press_break']))
            img = hud.draw(img, hs)
            emit(img, cue, f'{i:04d}_{shot["name"]}_{f:03d}')
        prev_si = si
    print(f'[render] main done ({time.time() - t_start:.0f}s)', flush=True)

    if enc is not None:
        enc.stdin.close()
        enc.wait()
    g0, beat = T.beat_grid()
    meta = {'fps': FPS, 'frames': cues, 'grid': [g0, beat], 'main_start': T.MAIN_START}
    with open(os.path.join(args.out, 'cues.json'), 'w') as fh:
        json.dump(meta, fh)
    print(f'video frames: {n_out[0]}, render time {time.time() - t_start:.0f}s', flush=True)

    if not args.no_audio and (enc is not None or args.cues_only) and only is None and args.every == 1:
        import audio
        wav = os.path.join(args.out, 'audio.wav')
        audio.build(meta, wav)
        if enc is not None:
            final = os.path.join(args.out, 'final.mp4')
            encode_youtube(video_path, wav, final, args.out)
            print('final:', final)


if __name__ == '__main__':
    main()
