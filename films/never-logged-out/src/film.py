"""The film: a timeline of shots, the 3D render of each shot into an intermediate video, and the compose pass that
draws the interface, the subtitles and the film look over it and encodes the result.

    python film.py list                                  # the shot list with times
    python film.py render [--shots a,b] [--preview]      # 3D intermediates (resumable: finished shots are skipped)
    python film.py compose [--preview] [--from T --to T] # frames -> output/film_noaudio.mp4
    python film.py stills OUT_DIR [--every S] [--preview] # composed stills for review
    python film.py cues                                  # the cue sheet for the sound (output/cues.json)

A shot is 3D (rendered every frame), 'still' (one 3D frame reused: the world behind a book), '2d' (drawn entirely
by its overlay) or 'black'.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

FPS = 24
W, H = 1920, 1080
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
CACHE = os.environ.get('NLO_CACHE', os.path.join(ROOT, 'cache'))


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return 'ffmpeg'


# ---------------------------------------------------------------------------------------------
# shots and the timeline
# ---------------------------------------------------------------------------------------------
class Shot:
    def __init__(self, name, dur, kind='3d', scene=None, overlay=None, fx=None, cues=(), subs=(), still=0.0,
                 hud=None, chat=False, world=None, tags=(), late=None, sub_y=None, feedback=False):
        self.late_fn = late
        self.feedback = feedback     # compose each frame as soon as it is rendered (for screens showing the video)
        self.sub_y = sub_y          # drawn after the HUD and chat (GUIs that cover them)
        self.name = name
        self.dur = float(dur)
        self.kind = kind
        self.scene_fn = scene
        self.overlay_fn = overlay
        self.fx_fn = fx
        self.cue_list = list(cues)
        self.sub_list = list(subs)
        self.still = still
        self.hud = hud               # None or callable(t) -> dict for ui.HUD.draw (or a dict)
        self.chat = chat
        self.world = world
        self.tags = tags
        self.start = 0.0

    @property
    def nframes(self):
        return int(round((self.start + self.dur) * FPS)) - int(round(self.start * FPS))

    @property
    def f0(self):
        return int(round(self.start * FPS))

    def scene(self, t):
        return self.scene_fn(t, self.start + t) if self.scene_fn else None

    def fx(self, t):
        d = {'grain': 0.032, 'vignette': 0.0}
        if self.fx_fn:
            d.update(self.fx_fn(t))
        return d


class Film:
    def __init__(self, shots):
        self.shots = shots
        t = 0.0
        for s in shots:
            s.start = t
            t += s.dur
        self.duration = t
        self.nframes = int(round(t * FPS))
        self.chat_log = []           # (t_global, text, colour)
        self.subs = []
        self.cues = []
        for s in shots:
            for (a, b, text) in s.sub_list:
                self.subs.append((s.start + a, s.start + b, text))
            for c in s.cue_list:
                t0, name = c[0], c[1]
                kw = c[2] if len(c) > 2 else {}
                self.cues.append((s.start + t0, name, kw))
        self.cues.sort(key=lambda c: c[0])

    def shot_at(self, frame):
        for s in self.shots:
            if s.f0 <= frame < s.f0 + s.nframes:
                return s, (frame - s.f0) / FPS
        return self.shots[-1], self.shots[-1].dur

    def by_name(self, name):
        for s in self.shots:
            if s.name == name:
                return s
        raise KeyError(name)

    def sub_at(self, t):
        for (a, b, text) in self.subs:
            if a <= t < b:
                fade = min(1.0, (t - a) / 0.15, (b - t) / 0.2)
                return text, max(0.0, fade)
        return None, 0.0


# ---------------------------------------------------------------------------------------------
# shared context: worlds, renderer, decals, skins, interface
# ---------------------------------------------------------------------------------------------
class Ctx:
    def __init__(self, preview=False, need_gl=True, variants=('village', 'changed', 'finale')):
        import worldgen as WG
        import skins as SK
        import ui
        self.preview = preview
        t = time.time()
        self.worlds = {v: WG.build(v) for v in variants}
        print(f'[ctx] worlds built {time.time() - t:.1f}s', flush=True)
        self.skins = SK.make_skins()
        self.hud = ui.HUD()
        self.desk = ui.Desktop()
        self.r = None
        self.dec = {}
        import maps as MP
        self.maps = MP.make_maps(self.worlds['village'])
        if need_gl:
            self._gl(variants)

    def _gl(self, variants):
        import decals as DC
        import entities as EN
        import scene as SC
        w_, h_ = (960, 540) if self.preview else (W, H)
        self.r = SC.make_renderer(w_, h_, skies=('sunset', 'dusk', 'night', 'wrong'),
                                  shadow_res=2048 if self.preview else 4096)
        for v in variants:
            SC.register_world(self.r, v, self.worlds[v], cache_name=v + '_v1')
            self.dec[v] = DC.Decals(self.r, self.worlds[v], maps=self.maps, prefix=v + ':')
        EN.register(self.r)
        import screen as SCR
        SCR.register(self.r)
        self.item_names = self.r._item_names
        self.last3d = None               # the previous main 3D frame (what the in-game monitors can show)
        self.last_composed = None        # the previous finished frame of a feedback shot (the video itself)

    def points(self, variant='village'):
        return self.worlds[variant].points


def build_instances(ctx, sc):
    import entities as EN
    world = sc.get('world', 'village')
    inst = {}
    if world in ctx.dec:
        inst.update(ctx.dec[world].instances(exclude=sc.get('exclude', ())))
    extra = list(sc.get('props', [])) + list(sc.get('fp', []))
    acts = EN.collect(sc.get('actors', []), extra)
    for k, v in acts.items():
        inst[k] = np.concatenate([inst[k], v]) if k in inst else v
    return inst


def render_scene(ctx, sc, frame_no=0, main=True):
    import looks
    r = ctx.r
    r.use_world(sc.get('world', 'village'))
    e = sc['env'] if not isinstance(sc['env'], str) else looks.get(sc['env'])
    r.time = sc.get('time', frame_no / FPS)
    r.dof = sc.get('dof')
    if sc.get('prep'):
        sc['prep'](r)
    r.render(sc['cam'], e, instances=build_instances(ctx, sc), lights=sc.get('lights'),
             particles=sc.get('particles'), streaks=sc.get('streaks'), clip_z=sc.get('clip_z'),
             near_center=sc.get('near_center'))
    img = r.finish(e)
    if ctx.preview:
        from PIL import Image
        img = np.asarray(Image.fromarray(img).resize((W, H), Image.BILINEAR))
    if main:
        ctx.last3d = img
    return img


def project(cam, pts):
    """World points -> (N, 2) pixel positions in the 1920 x 1080 frame and view depth (for name tags)."""
    import gfx
    eye = np.asarray(cam['eye'], float)
    view = gfx.look_at(eye, cam['target'], cam.get('up', (0, 0, 1)))
    if cam.get('roll', 0.0):
        rr = np.radians(cam['roll'])
        rz = np.eye(4)
        rz[0, 0], rz[0, 1], rz[1, 0], rz[1, 1] = np.cos(rr), -np.sin(rr), np.sin(rr), np.cos(rr)
        view = rz @ view
    proj = gfx.perspective(cam['fov'], W / H, 0.05, 1200.0)
    p = np.c_[np.asarray(pts, float).reshape(-1, 3), np.ones(len(np.asarray(pts).reshape(-1, 3)))]
    v = p @ view.T
    c = v @ proj.T
    ndc = c[:, :2] / c[:, 3:4]
    return np.stack([(ndc[:, 0] * 0.5 + 0.5) * W, (1 - (ndc[:, 1] * 0.5 + 0.5)) * H], -1), -v[:, 2]


# ---------------------------------------------------------------------------------------------
# video io
# ---------------------------------------------------------------------------------------------
class Writer:
    def __init__(self, path, w=W, h=H, crf=12, preset='veryfast', fps=FPS):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.tmp = path + '.part.mp4'
        cmd = [ffmpeg_exe(), '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r',
               str(fps), '-i', '-', '-c:v', 'libx264', '-preset', preset, '-crf', str(crf), '-pix_fmt', 'yuv420p',
               self.tmp]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, img):
        self.p.stdin.write(np.ascontiguousarray(img, np.uint8).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()
        os.replace(self.tmp, self.path)


def read_frames(path, w=W, h=H):
    cmd = [ffmpeg_exe(), '-v', 'error', '-i', path, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = w * h * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield np.frombuffer(b, np.uint8).reshape(h, w, 3)


def count_frames(path):
    try:
        out = subprocess.run([ffmpeg_exe(), '-v', 'error', '-i', path, '-map', '0:v:0', '-c', 'copy', '-f', 'null',
                              '-'], capture_output=True, text=True)
        del out
    except Exception:
        return -1
    return -1


# ---------------------------------------------------------------------------------------------
# render (3D intermediates) and compose
# ---------------------------------------------------------------------------------------------
def frames_dir(preview):
    return os.path.join(CACHE, 'frames_preview' if preview else 'frames')


def render_shots(film, ctx, names=None, preview=False, force=False):
    d = frames_dir(preview)
    for s in film.shots:
        if s.kind not in ('3d', 'still'):
            continue
        if names and s.name not in names:
            continue
        path = os.path.join(d, s.name + '.mp4')
        if os.path.exists(path) and not force:
            continue
        t0 = time.time()
        n = s.nframes if s.kind == '3d' else 1
        wr = Writer(path, crf=20 if preview else 12)
        for k in range(n):
            t = k / FPS if s.kind == '3d' else s.still
            sc = s.scene(t)
            img = render_scene(ctx, sc, s.f0 + k)
            if s.feedback:
                ctx.last_composed = compose_frame(film, ctx, s, t, img, s.f0 + k)
            wr.write(img)
            if k % 48 == 47:
                el = time.time() - t0
                print(f'  {s.name}: {k + 1}/{n}  {el / (k + 1):.2f}s/frame', flush=True)
        wr.close()
        print(f'[render] {s.name}: {n} frames in {time.time() - t0:.0f}s', flush=True)


def compose_frame(film, ctx, s, t, img3d, frame):
    import post
    import ui
    tg = s.start + t
    img = img3d.copy() if img3d is not None else np.zeros((H, W, 3), np.uint8)
    hud_state = s.hud(t) if callable(s.hud) else s.hud
    if s.overlay_fn:
        s.overlay_fn(img, t, tg, ctx, film)
    if hud_state:
        ctx.hud.draw(img, **hud_state)
    if s.chat:
        ui.chat(img, film.chat_log, tg, open_=False)
    if getattr(s, 'late_fn', None):
        s.late_fn(img, t, tg, ctx, film)
    text, a = film.sub_at(tg)
    if text:
        y = s.sub_y if s.sub_y is not None else (ui.H - 250 if hud_state else ui.H - 90)
        ui.subtitle(img, text, alpha=a, y=y)
    return post.apply(img, frame, tg, s.fx(t))


def compose(film, ctx, out_path, preview=False, t_from=0.0, t_to=None, every=1, stills_dir=None):
    d = frames_dir(preview)
    wr = None if stills_dir else Writer(out_path, crf=16 if not preview else 22, preset='medium' if not preview else
                                       'veryfast', fps=FPS / every)
    t_to = film.duration if t_to is None else t_to
    t0 = time.time()
    count = 0
    for s in film.shots:
        if s.start + s.dur <= t_from or s.start >= t_to:
            continue
        src = None
        still = None
        if s.kind in ('3d', 'still'):
            path = os.path.join(d, s.name + '.mp4')
            if not os.path.exists(path):
                print(f'[compose] missing {s.name}, using black', flush=True)
            elif s.kind == 'still':
                still = next(read_frames(path))
            else:
                src = read_frames(path)
        for k in range(s.nframes):
            frame = s.f0 + k
            img3d = None
            if src is not None:
                try:
                    img3d = next(src)
                except StopIteration:
                    src = None
            elif still is not None:
                img3d = still
            t = k / FPS
            tg = s.start + t
            if tg < t_from or tg >= t_to or frame % every:
                continue
            out = compose_frame(film, ctx, s, t, img3d, frame)
            if stills_dir:
                from PIL import Image
                Image.fromarray(out).save(os.path.join(stills_dir, f'{frame:06d}_{s.name}.jpg'), quality=88)
            else:
                wr.write(out)
            count += 1
            if count % 240 == 0:
                print(f'  composed {count} frames ({tg:.1f}s)  {(time.time() - t0) / count:.3f}s/frame', flush=True)
    if wr:
        wr.close()
    print(f'[compose] {count} frames in {time.time() - t0:.0f}s', flush=True)


def load_film(ctx):
    import story
    return story.film(ctx)


def make_assets(ctx):
    """The launcher's banner (and the world's icon): a still of the village at sunset."""
    import looks
    from PIL import Image
    import common as C
    cam = dict(eye=(46, -46, 12), target=(2, -14, 3), fov=55)
    sc = dict(world='village', env=looks.get('sunset'), cam=cam, actors=C.Village().villagers(40.0) +
              C.animals(40.0), props=C.world_props(ctx, 'village'))
    img = render_scene(ctx, sc)
    os.makedirs(CACHE, exist_ok=True)
    Image.fromarray(img).resize((960, 540), Image.LANCZOS).save(os.path.join(CACHE, 'banner.png'))
    print('[assets] banner.png')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd')
    ap.add_argument('arg', nargs='?')
    ap.add_argument('--shots', default='')
    ap.add_argument('--preview', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--every', type=float, default=0.0)
    ap.add_argument('--from', dest='t_from', type=float, default=0.0)
    ap.add_argument('--to', dest='t_to', type=float, default=None)
    a = ap.parse_args()
    ctx = Ctx(preview=a.preview, need_gl=a.cmd in ('render', 'assets'))
    film = load_film(ctx)
    if a.cmd == 'assets':
        make_assets(ctx)
        return
    if a.cmd == 'list':
        for s in film.shots:
            print(f'{s.start:8.2f} {s.start + s.dur:8.2f}  {s.kind:6s} {s.name}')
        print(f'total {film.duration:.2f}s = {int(film.duration // 60)}:{film.duration % 60:05.2f}, '
              f'{film.nframes} frames; 3D frames: {sum(s.nframes for s in film.shots if s.kind == "3d")}')
        return
    if a.cmd == 'cues':
        os.makedirs(os.path.join(ROOT, 'output'), exist_ok=True)
        with open(os.path.join(ROOT, 'output', 'cues.json'), 'w') as fh:
            json.dump({'fps': FPS, 'duration': film.duration, 'cues': film.cues, 'subs': film.subs,
                       'chat': film.chat_log}, fh, indent=1, default=float)
        print(len(film.cues), 'cues')
        return
    names = set(x for x in a.shots.split(',') if x)
    if a.cmd == 'probe':
        # composed stills of chosen shots at chosen fractions of their length: --shots a,b --every 0.25
        from PIL import Image
        os.makedirs(a.arg, exist_ok=True)
        fr = a.every or 0.5
        for s in film.shots:
            if names and s.name not in names:
                continue
            for u in np.arange(0.0, 1.0 + 1e-6, fr):
                t = min(u * s.dur, s.dur - 1.0 / FPS)
                img3d = None
                if s.kind in ('3d', 'still'):
                    ctx2 = ctx if ctx.r is not None else None
                    if ctx2 is None:
                        ctx.preview = a.preview
                        ctx._gl(tuple(ctx.worlds))
                    img3d = render_scene(ctx, s.scene(t if s.kind == '3d' else s.still), s.f0)
                out = compose_frame(film, ctx, s, t, img3d, s.f0 + int(t * FPS))
                ctx.last_composed = out
                Image.fromarray(out).save(os.path.join(a.arg, f'{s.name}_{t:05.2f}.jpg'), quality=88)
                print('[probe]', s.name, f'{t:.2f}', flush=True)
        return
    if a.cmd == 'render':
        render_shots(film, ctx, names or None, a.preview, a.force)
    elif a.cmd == 'compose':
        out = os.path.join(ROOT, 'preview' if a.preview else 'output', 'film_noaudio.mp4')
        compose(film, ctx, out, a.preview, a.t_from, a.t_to)
    elif a.cmd == 'stills':
        os.makedirs(a.arg, exist_ok=True)
        every = max(1, int(round((a.every or 1.0) * FPS)))
        compose(film, ctx, None, a.preview, a.t_from, a.t_to, every=every, stills_dir=a.arg)
    else:
        print('unknown command', a.cmd)
        sys.exit(1)


if __name__ == '__main__':
    main()
