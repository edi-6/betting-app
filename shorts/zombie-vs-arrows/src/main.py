"""Render "Zombie vs 10,000 Arrows": hook montage + five rounds, HUD, then sound design and final mux.

Usage:
  python main.py --out OUT_DIR [--preview] [--ss 1.5] [--only r3] [--no-audio] [--encode-only]
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
from PIL import Image

import arrows as AR
import scene
import sim as S
import timeline as T
from hud import HUD
from mathutil import quat_rotate

W, H = 1080, 1920
FPS = T.FPS
UP = np.array([0.0, 0.0, 1.0])


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
    subprocess.run([ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', video_in, '-i', wav_in, '-map', '0:v:0', '-map', '1:a:0']
                   + common + ['-pass', '2', '-passlogfile', log, '-c:a', 'aac', '-b:a', '384k', '-ar', '48000', '-ac', '2',
                               '-movflags', '+faststart', '-shortest', '-metadata', 'title=Zombie vs 10,000 Arrows', out],
                   check=True)


def compact_events(ev):
    return {k: [int(cnt), (psum / pn).tolist() if pn else None] for k, (cnt, psum, pn) in ev.items()}


# ---------------------------------------------------------------------------------------------
# pass 1: simulate a round without rendering (same time-scale schedule), measure damage/events
# ---------------------------------------------------------------------------------------------
def make_sim(rd):
    """Round simulation, pre-rolled so arrows can already be in the air when the round starts."""
    sim = S.RoundSim(rd['formation'], **rd['sim'])
    for _ in range(T.sec(rd.get('preroll', 0.0))):
        sim.step_frame(1.0)
    return sim


def analyse_round(rd):
    sim = make_sim(rd)
    n = T.sec(rd['duration'])
    scales = T.frame_scales(rd.get('scale'), n)
    destroyed = np.zeros(n)
    events = []
    for f in range(n):
        ev = sim.step_frame(scales[f])
        destroyed[f] = sim.destroyed
        events.append(compact_events(ev))
    return {'destroyed': destroyed, 'events': events, 'total': sim.g.total, 'scales': scales,
            'stuck_end': int((sim.state == S.STUCK).sum())}


def hp_curve(rd, an):
    d = an['destroyed']
    n = len(d)
    stamp_f = min(n - 1, T.sec(rd['stamp_at']))
    if rd.get('death_fraction'):
        ref = rd['death_fraction'] * an['total']
    else:
        ref = max(d[stamp_f], 1.0)
    loss = rd['hp_loss'] * np.clip(d / ref, 0, 1)
    steps = np.ceil(loss - 1e-6).astype(int)
    steps = np.minimum(steps, rd['hp_loss'])
    if rd.get('death_fraction'):
        # make sure the zombie is dead before the check mark lands, whatever the sim did
        f_dead = np.nonzero(steps >= rd['hp_loss'])[0]
        deadline = max(0, stamp_f - T.sec(0.6))
        if len(f_dead) == 0 or f_dead[0] > deadline:
            first = np.nonzero(steps > 0)[0]
            f0 = first[0] if len(first) else deadline // 2
            ramp = np.arange(n)
            forced = np.ceil(rd['hp_loss'] * np.clip((ramp - f0) / max(1, deadline - f0), 0, 1)).astype(int)
            steps = np.maximum(steps, forced)
    steps[stamp_f:] = steps[stamp_f]
    return 20 - steps


def shake_curve(rd, an):
    n = len(an['destroyed'])
    s = np.zeros(n)
    acc = 0.0
    for f in range(n):
        ev = an['events'][f]
        imp = ev.get('impact', [0])[0]
        carve = ev.get('carve', [0])[0]
        shat = ev.get('shatter', [0])[0]
        kick = np.log1p(imp) * 0.3 + np.log1p(carve) * 0.04 + np.log1p(shat) * 0.3
        acc = acc * 0.86 + kick
        s[f] = acc
    return rd['shake'] * np.tanh(s / 3.0)


# ---------------------------------------------------------------------------------------------
# cameras
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


class FollowCam:
    """Arrow-cam: rides just behind one arrow. With `stop`, the camera brakes smoothly shortly before impact and
    halts `stop` blocks short of it, so the hit is seen rather than flown into; after the hit it swings round to
    the side (`pull_side`) and eases back (`pull`) to show the arrow sticking out of him."""

    def __init__(self, sim, spec):
        self.sim = sim
        self.i = spec['arrow']
        self.back = spec.get('back', 3.0)
        self.up = spec.get('up', 1.0)
        self.side = spec.get('side', 1.2)
        self.look = spec.get('look', 12.0)
        self.fov = spec.get('fov', 58.0)
        self.stop = spec.get('stop', 0.0)
        self.soft = spec.get('soft', 1.5)
        self.pull = spec.get('pull', 0.0)
        self.pull_side = spec.get('pull_side', 0.0)
        self.pull_up = spec.get('pull_up', 0.0)
        self.pull_time = spec.get('pull_time', 1.0)
        self.pull_fov = spec.get('pull_fov', 0.0)
        self.hit = None
        self.H = self.dH = None
        if self.stop > 0:
            self.H, self.dH = self._predict_hit()

    def _predict_hit(self):
        """Where the arrow's tip will first touch the zombie (or the ground), by integrating its flight."""
        s = self.sim
        p = s.p[self.i].copy()
        v = (s.v0 if s.state[self.i] == S.WAIT else s.v)[self.i].copy()
        dt = 1.0 / 480
        for _ in range(480 * 6):
            p = p + v * dt + 0.5 * S.GRAV * dt * dt
            v = v + S.GRAV * dt
            d = v / np.linalg.norm(v)
            tip = p + d * S.TIP
            ii, jj, kk = S.Giant.world_to_index(tip[None])
            if s.g.occupied(ii, jj, kk)[0] or tip[2] < 0.0:
                return tip, d
        return None, None

    def _pose(self, p, d, side):
        eye = p - d * self.back + UP * self.up + side * self.side
        tgt = p + d * self.look
        if self.H is not None:
            x = float(np.dot(self.H - eye, self.dH))          # how far the eye still is from the impact
            xs = self.stop + self.soft * np.log1p(np.exp(np.clip((x - self.stop) / self.soft, -30, 30)))
            eye = eye + self.dH * (x - xs)                     # never closer than `stop`, eased in
            w = np.clip((xs - x) / 4.0, 0.0, 1.0)
            w = w * w * (3 - 2 * w)
            tgt = tgt * (1 - w) + self.H * w
        return eye, tgt

    def __call__(self, t):
        s = self.sim
        i = self.i
        if s.state[i] in (S.WAIT, S.FLY):
            eye, tgt = self._pose(s.p[i], s.d[i], s.side[i])
            return eye, tgt, self.fov, True
        if self.hit is None:
            eye, tgt = self._pose(s.p[i], s.d[i], s.side[i])
            tip = s.p[i] + s.d[i] * S.TIP
            self.hit = (eye, tgt, s.d[i].copy(), s.side[i].copy(), tip - s.d[i] * 0.8, t)
        eye0, tgt0, d0, side0, aim, t0 = self.hit
        u = np.clip((t - t0) / self.pull_time, 0.0, 1.0)
        u = u * u * (3 - 2 * u)
        eye = eye0 - d0 * self.pull * u + side0 * self.pull_side * u + UP * self.pull_up * u
        tgt = tgt0 * (1 - u) + aim * u
        return eye, tgt, self.fov + self.pull_fov * u, False


def swarm_info(sim, eye, exclude=None):
    """Flying arrows within 30 blocks of the camera, the closest one's distance and position (for fly-by
    whooshes); `exclude`: the arrow-cam's own arrow."""
    f = sim.state == S.FLY
    if exclude is not None:
        f[exclude] = False
    if not f.any():
        return 0, 999.0, None
    p = sim.p[f]
    d = np.linalg.norm(p - eye, axis=1)
    k = int(np.argmin(d))
    return int((d < 30.0).sum()), float(d[k]), p[k].tolist()


HURT_GAP = 18      # frames between damage flashes, so a hail of hits reads as a pulse rather than solid red


def hurt_amount(frames_since_hit):
    """Red damage overlay, on for a quarter second after a hit and then off, like the game."""
    return 1.0 if 0 <= frames_since_hit < 8 else 0.0


def fade_near(arrows, eye, d0=0.8, d1=2.0, keep=None):
    """Screen-door fade for arrows passing close to the lens (between d1 and d0; gone inside d0), so the near
    plane never slices one into big flat shards. `keep`: position of an arrow that must stay (the arrow-cam's)."""
    if not len(arrows):
        return arrows
    p = arrows[:, :3].astype(np.float64)
    d = quat_rotate(arrows[:, 3:7].astype(np.float64), np.tile([1.0, 0.0, 0.0], (len(arrows), 1)))
    a = p - d * AR.TIP
    ab = d * (2 * AR.TIP)
    t = np.clip(np.sum((np.asarray(eye) - a) * ab, 1) / np.sum(ab * ab, 1), 0.0, 1.0)
    dist = np.linalg.norm(a + ab * t[:, None] - eye, axis=1)
    u = np.clip((dist - d0) / (d1 - d0), 0.0, 1.0)
    fade = u * u * (3 - 2 * u)
    if keep is not None:
        fade[np.linalg.norm(p - keep, axis=1) < 1e-3] = 1.0
    out = arrows[fade > 0].copy()
    out[:, 9] = fade[fade > 0]
    return out


def render_frame(r, cam, sim, preview, hurt=0.0, follow=None):
    """follow: the FollowCam in use (its arrow stays sharp, the rest of the swarm fades out around the lens)."""
    vox, arrows, fx = sim.instances()
    if follow is not None:
        arrows = fade_near(arrows, np.asarray(cam['eye'], float), 3.0, 8.0, keep=sim.p[follow.i])
    else:
        arrows = fade_near(arrows, np.asarray(cam['eye'], float))
    r.render(cam, voxels=vox, arrows=arrows, fx=fx, hurt=(hurt, sim.n_static))
    img = r.finish()
    if preview:
        img = np.array(Image.fromarray(img).resize((W, H), Image.BILINEAR))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--preview', action='store_true')
    ap.add_argument('--ss', type=float, default=1.5, help='supersampling factor (1.5 = delivered quality)')
    ap.add_argument('--only', default=None, help='comma separated list of segments: hook,r1..r5')
    ap.add_argument('--crf', type=int, default=15)
    ap.add_argument('--no-audio', action='store_true')
    ap.add_argument('--encode-only', action='store_true',
                    help='rebuild audio from OUT/cues.json and redo the final encode of OUT/video_noaudio.mp4')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.encode_only:
        import audio
        with open(os.path.join(args.out, 'cues.json')) as fh:
            d = json.load(fh)
        wav = os.path.join(args.out, 'audio.wav')
        audio.build(d['frames'], d['fps'], wav)
        final = os.path.join(args.out, 'final.mp4')
        encode_youtube(os.path.join(args.out, 'video_noaudio.mp4'), wav, final, args.out)
        print('final:', final)
        return

    rounds = T.round_defs()
    rmap = {rd['key']: rd for rd in rounds}
    only = set(args.only.split(',')) if args.only else None

    # ---- pass 1
    t0 = time.time()
    analysis = {}
    for rd in rounds:
        if only is not None and rd['key'] not in only:
            continue
        a = analyse_round(rd)
        a['hp'] = hp_curve(rd, a)
        a['shake'] = shake_curve(rd, a)
        analysis[rd['key']] = a
        print(f"[analyse] {rd['key']}: destroyed {int(a['destroyed'][-1])}/{a['total']}  hp_end {a['hp'][-1]}  "
              f"stuck {a['stuck_end']}  ({time.time() - t0:.0f}s)", flush=True)

    # ---- renderer
    if args.preview:
        r, _ = scene.make_renderer(width=W // 2, height=H // 2, ss=1.0, shadow_res=2048)
    else:
        r, _ = scene.make_renderer(width=W, height=H, ss=args.ss, shadow_res=4096)
    hud = HUD(W, H)
    video_path = os.path.join(args.out, 'video_noaudio.mp4')
    cmd = [ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
           '-r', str(FPS), '-i', '-', '-vf', 'scale=out_color_matrix=bt709:out_range=tv',
           '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
           '-c:v', 'libx264', '-preset', 'slow', '-crf', str(args.crf),
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', video_path]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    cues = []          # per output frame: dict of audio cues
    out_frame = 0
    t_start = time.time()

    def emit(img, cue):
        nonlocal out_frame
        enc.stdin.write(np.ascontiguousarray(img).tobytes())
        cues.append(cue)
        out_frame += 1
        if out_frame % 15 == 0:
            el = time.time() - t_start
            print(f'  frame {out_frame}  {el:.0f}s  ({el / out_frame:.2f}s/frame)', flush=True)
        if out_frame % 30 == 1:
            Image.fromarray(img).save(os.path.join(args.out, f'thumb_{out_frame:04d}.jpg'), quality=88)

    # ---- hook
    if only is None or 'hook' in only:
        for hi, hs in enumerate(T.hook_shots()):
            rd = rmap[hs['round']]
            sim = S.RoundSim(rd['formation'], **rd['sim'])
            for _ in range(T.sec(hs['sim_start'])):
                sim.step_frame(1.0)
            nf = T.sec(hs['dur'])
            scales = T.frame_scales(hs.get('scale'), nf)
            follow = FollowCam(sim, hs['follow']) if 'follow' in hs else None
            for f in range(nf):
                ev = sim.step_frame(scales[f])
                t = f / FPS
                if follow is not None:
                    eye, tgt, fov, riding = follow(t)
                else:
                    eye, tgt, fov = T.cam_path(hs['keys'], t)
                    riding = False
                img = render_frame(r, {'eye': eye, 'target': tgt, 'fov': fov}, sim, args.preview, follow=follow)
                img = hud.draw(img, {'hp': 20})
                n_near, dmin, cpos = swarm_info(sim, eye, None if follow is None else follow.i)
                emit(img, {'seg': 'hook', 'shot': hi, 'shot_frame': f, 'round': rd['key'],
                           'events': compact_events(ev), 'cam': eye.tolist(), 'tgt': np.asarray(tgt).tolist(),
                           'scale': float(scales[f]), 'swarm': n_near, 'swarm_dmin': dmin, 'swarm_pos': cpos,
                           'riding': bool(riding), 'cut': f == 0})

    # ---- rounds
    for rd in rounds:
        if only is not None and rd['key'] not in only:
            continue
        a = analysis[rd['key']]
        sim = make_sim(rd)
        shaker = Shaker(sum(map(ord, rd['key'])))       # deterministic across runs
        n = T.sec(rd['duration'])
        stamp_f = T.sec(rd['stamp_at'])
        flash_start = -100
        follows = {}
        prev_si = None
        for f in range(n):
            sim.step_frame(a['scales'][f])
            t = f / FPS
            si = max(k for k, s in enumerate(rd['shots']) if t >= s['start'] - 1e-9)
            shot = rd['shots'][si]
            riding = False
            follow = None
            if 'follow' in shot:
                if si not in follows:
                    follows[si] = FollowCam(sim, shot['follow'])
                follow = follows[si]
                eye, tgt, fov, riding = follow(t - shot['start'])
            else:
                eye, tgt, fov = T.cam_path(shot['keys'], t)
            amp = a['shake'][f]
            off = shaker.offset(t, amp)
            eye = eye + off
            tgt = tgt + off * 0.6
            roll = float(off[0] * 0.8)
            hp = int(a['hp'][f])
            if f > 0 and hp < a['hp'][f - 1] and f - flash_start >= HURT_GAP:
                flash_start = f
            hurt = hurt_amount(f - flash_start)
            img = render_frame(r, {'eye': eye, 'target': tgt, 'fov': fov, 'roll': roll}, sim, args.preview,
                               hurt=hurt, follow=follow)
            flash = (f - flash_start) < 12 and ((f - flash_start) // 3) % 2 == 0
            st = {'hp': hp, 'flash': flash, 'jiggle_seed': out_frame}
            lab_in, lab_out = 0.0, 1.75
            if lab_in <= t < lab_out:
                st['label'] = rd['label']
                st['label_t'] = min(1.0, (t - lab_in) / 0.3)
                st['label_alpha'] = min(1.0, max(0.0, (lab_out - t) / 0.2))
            if f >= stamp_f:
                st['stamp'] = rd['stamp']
                st['stamp_t'] = min(1.0, (f - stamp_f) / (0.3 * FPS))
            img = hud.draw(img, st)
            n_near, dmin, cpos = swarm_info(sim, eye, None if follow is None else follow.i)
            emit(img, {'seg': rd['key'], 'shot': si, 'round_frame': f, 'events': a['events'][f], 'cam': eye.tolist(),
                       'tgt': np.asarray(tgt).tolist(), 'hp': hp, 'hp_prev': int(a['hp'][f - 1]) if f else 20,
                       'label': f == T.sec(lab_in), 'stamp': rd['stamp'] if f == stamp_f else None,
                       'cut': f == 0 or si != prev_si, 'scale': float(a['scales'][f]),
                       'n_arrows': int(sim.n), 'swarm': n_near, 'swarm_dmin': dmin, 'swarm_pos': cpos,
                       'riding': bool(riding), 'hurt': hurt})
            prev_si = si
        print(f"[render] {rd['key']} done", flush=True)

    enc.stdin.close()
    enc.wait()
    with open(os.path.join(args.out, 'cues.json'), 'w') as fh:
        json.dump({'fps': FPS, 'frames': cues}, fh)
    print(f'video frames: {out_frame}, render time {time.time() - t_start:.0f}s', flush=True)

    if not args.no_audio:
        import audio
        wav = os.path.join(args.out, 'audio.wav')
        audio.build(cues, FPS, wav)
        final = os.path.join(args.out, 'final.mp4')
        encode_youtube(video_path, wav, final, args.out)
        print('final:', final)


if __name__ == '__main__':
    main()
