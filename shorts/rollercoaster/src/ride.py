"""The ride as a film: the timeline of segments (which world, which stretch of which track, in slow motion or not),
the worlds (built once and cached), and the scene of any frame.

Segments, in order (the video loops from the last back to the first without a seam):
  A  OVERWORLD  from the lift hill's crest, the drop, the canyon, into the ruined portal
  B  NETHER     out of the portal on the ledge ... the jump (in slow motion) ... into the End portal
  C  THE END    off the obsidian platform ... the dragon ... into the exit portal
  E  DEEP DARK  out into the ancient city, round its tower, into the great portal as it wakes
  F  THE SIFT   the new dimension: off the cliff into the Singer's Meadow ... under the fossil ... into the rift
  D  OVERWORLD  back on the lift hill, clicking up to the crest: the frame after the last is the first

Everything moves in ride time (tau, per track, from the track's physics); the video time maps onto it, slower where
a segment has slow motion.
"""
import hashlib
import os
import time

import numpy as np

import looks
import paths
import props as PR
import rider
import track as TK
import voxel as VX

FPS = 60
W, H = 1080, 1920
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, '..', 'cache')
STANDARD = {'size', 'origin', 'ids', 'state', 'hidden', 'tint', 'signs', 'frames'}


# ---------------------------------------------------------------------------------------------
# worlds
# ---------------------------------------------------------------------------------------------
def cache_path(key, sources):
    h = hashlib.md5()
    for f in sources + ['blocks.py', 'voxel.py', 'textures.py', 'paths.py', 'track.py']:
        with open(os.path.join(HERE, f), 'rb') as fh:
            h.update(fh.read())
    return os.path.join(CACHE, f'world_{key}_{h.hexdigest()[:10]}.npz')


def _build_cached(path, build):
    w = build()
    mesh, light = w.mesh(), w.light()
    meta = {k: v for k, v in w.__dict__.items() if k not in STANDARD}
    meta['origin'] = w.origin
    meta['size'] = w.size
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, mesh=np.array(mesh, dtype=object), light=light, meta=np.array(meta, dtype=object))
    return mesh, light, meta


def load_world(r, key, build, sources):
    """Build (or load from the cache) a world, upload it, and return its metadata (portal positions etc.)."""
    t = time.time()
    path = cache_path(key, sources)
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        mesh, light, meta = d['mesh'].item(), d['light'], d['meta'].item()
    else:
        mesh, light, meta = _build_cached(path, build)
    r.add_world(key, mesh, light, meta['origin'])
    n = sum(len(v[1]) // 3 for v in mesh.values())
    print(f'[ride] world {key}: {n} triangles, {time.time() - t:.1f}s', flush=True)
    return meta


def load_meta(key, build, sources):
    """Just a world's metadata (for the soundtrack's cue sheet), from the cache, building the world if need be."""
    path = cache_path(key, sources)
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)['meta'].item()
    return _build_cached(path, build)[2]


# ---------------------------------------------------------------------------------------------
# the timeline
# ---------------------------------------------------------------------------------------------
class Segment:
    def __init__(self, name, world, tr, tau0, tau1, slow=()):
        """slow: list of (tau_a, tau_b, factor, ramp_in, ramp_out): between tau_a and tau_b the ride plays at
        `factor` of real speed, easing in before tau_a and out after tau_b over the ramps (seconds of ride time)."""
        self.name, self.world, self.tr = name, world, tr
        self.tau0, self.tau1 = tau0, tau1
        taus = np.linspace(tau0, tau1, int((tau1 - tau0) * 2000) + 2)
        k = np.ones_like(taus)
        for (a, b, f, ramp_in, ramp_out) in slow:
            up = np.clip((taus - (a - ramp_in)) / ramp_in, 0, 1)
            dn = np.clip(((b + ramp_out) - taus) / ramp_out, 0, 1)
            w = np.minimum(up, dn)
            w = w * w * (3 - 2 * w)
            k = np.minimum(k, 1 - (1 - f) * w)
        dt = np.diff(taus) / (0.5 * (k[1:] + k[:-1]))
        self.taus = taus
        self.tv = np.concatenate([[0.0], np.cumsum(dt)])
        self.duration = float(self.tv[-1])
        self.k = k

    def tau(self, t):
        return float(np.interp(t, self.tv, self.taus))

    def speed_factor(self, t):
        return float(np.interp(t, self.tv, self.k))


def crossing(tr, point, normal):
    """Arc length where the track crosses the plane through point with the given normal."""
    d = (tr.P - np.asarray(point)) @ np.asarray(normal, float)
    idx = np.nonzero((d[:-1] < 0) & (d[1:] >= 0))[0]
    if not len(idx):
        return tr.length - 0.5
    i = idx[0]
    u = -d[i] / (d[i + 1] - d[i])
    return float(tr.s[i] + u * TK.STEP)


class Ride:
    def __init__(self, metas=None):
        self.tracks = {'over': paths.overworld(), 'nether': paths.nether(), 'end': paths.the_end(),
                       'deep': paths.deep(), 'sift': paths.sift()}
        self.metas = metas or {}
        o, n, e = self.tracks['over'], self.tracks['nether'], self.tracks['end']
        dp, sf = self.tracks['deep'], self.tracks['sift']
        md, ms = self.metas.get('deep', {}), self.metas.get('sift', {})
        if 'gate' in md:
            s_cut_e = crossing(dp, md['gate']['center'], md['gate']['normal']) + 0.8
        else:
            s_cut_e = dp.marks['portal'] + 1.0
        if 'portals' in ms:
            rift = ms['portals']['rift']
            s_cut_f = crossing(sf, rift['center'], rift['normal']) + 0.8
        else:
            s_cut_f = sf.marks['rift'] + 1.3
        crest = o.time(o.marks['crest'])
        mo = self.metas.get('over', {})
        if 'portal' in mo:
            s_cut_a = crossing(o, mo['portal']['center'], (0, 1, 0))
        else:
            s_cut_a = o.marks['portal'] + 1.0
        s_cut_b = n.marks['portal'] + 7.0
        s_cut_c = e.marks['portal'] + 8.0
        jump0, jump1 = n.time(n.marks['takeoff']), n.time(n.marks['land'])
        self.segments = [
            Segment('A', 'over', o, crest, o.time(s_cut_a)),
            Segment('B', 'nether', n, n.time(0.6), n.time(s_cut_b),
                    slow=[(jump0 - 0.08, jump1 - 0.14, 0.3, 0.38, 0.14)]),
            Segment('C', 'end', e, e.time(0.6), e.time(s_cut_c)),
            Segment('E', 'deep', dp, dp.time(0.6), dp.time(s_cut_e)),
            Segment('F', 'sift', sf, sf.time(0.6), sf.time(s_cut_f)),
            Segment('D', 'over', o, crest - 3.4, crest),
        ]
        self.starts = np.cumsum([0.0] + [sg.duration for sg in self.segments])
        self.duration = float(self.starts[-1])
        self.nframes = int(round(self.duration * FPS))

    def locate(self, f):
        """Frame index -> (segment, time within the segment, ride time tau)."""
        tv = f / FPS
        i = int(np.searchsorted(self.starts, tv, side='right') - 1)
        i = min(max(i, 0), len(self.segments) - 1)
        sg = self.segments[i]
        t = tv - self.starts[i]
        return sg, t, sg.tau(t)

    def video_time(self, name, tau):
        """Video time of ride time tau in the segment called name."""
        i = [sg.name for sg in self.segments].index(name)
        sg = self.segments[i]
        return float(self.starts[i] + np.interp(tau, sg.taus, sg.tv))

    def summary(self):
        out = [f'{self.duration:.2f} s, {self.nframes} frames at {FPS} fps']
        for sg, t0 in zip(self.segments, self.starts):
            out.append(f'  {sg.name} {sg.world:6s} {t0:6.2f} .. {t0 + sg.duration:6.2f}  ({sg.duration:5.2f} s video, '
                       f'{sg.tau1 - sg.tau0:5.2f} s ride)')
        return '\n'.join(out)


# ---------------------------------------------------------------------------------------------
# what the rider does
# ---------------------------------------------------------------------------------------------
def look_dist(world, tr, s):
    """At the top of a big drop the rider looks down it (the view the video opens on)."""
    if world == 'over':
        c = tr.marks['crest']
        w = np.clip((s - (c - 7.0)) / 7.0, 0, 1) * np.clip(1 - (s - (c + 8.0)) / 14.0, 0, 1)
        return None if w <= 0 else float(w * 24.0 + (1 - w) * np.clip(4.0 + tr.speed(s) * 0.26, 4, 14))
    if world in ('nether', 'sift'):
        c = tr.marks['drop']
        w = np.clip(1 - np.abs(s - c) / 12.0, 0, 1)
        return None if w <= 0 else float(w * 20.0 + (1 - w) * np.clip(4.0 + tr.speed(s) * 0.26, 4, 14))
    return None


def hands_up(world, tr, s):
    """Arms thrown up on the big drops."""
    if world == 'over':
        a, b = tr.marks['drop'] + 2.0, tr.marks['bottom'] - 6.0
    elif world == 'nether':
        a, b = tr.marks['drop'] + 2.0, tr.marks['lava'] - 8.0
    elif world == 'sift':
        a, b = tr.marks['drop'] + 2.0, tr.marks['meadow'] - 6.0
    else:
        return 0.0
    return float(np.clip((s - a) / 5.0, 0, 1) * np.clip((b - s) / 6.0, 0, 1))


BED = {'over': 'spruce_planks', 'nether': 'nether_bricks', 'end': 'end_stone_bricks', 'deep': 'deepslate_bricks',
       'sift': 'siftslate_bricks'}


def base_scene(ride, f, extra=None):
    """The scene of frame f without the per-world events: dict(world, env, cam, instances, rows, lights,
    particles, seg, t, tau, s, v, g)."""
    sg, t, tau = ride.locate(f)
    tr = sg.tr
    s = tr.s_at(tau)
    import fx
    cam, eye, (F, R, U), (T, Rt, Ut), v, g = rider.view(tr, s, tau, look_dist=look_dist(sg.world, tr, s),
                                                        attend=fx.attention(sg.world, tr, tau),
                                                        jolt=fx.jolt(sg.world, tr, tau))
    at = VX.atlas()
    bed = BED[sg.world] if BED[sg.world] in at.layer else 'spruce_planks'
    rows = PR.track_rows(tr, s - 24.0, s + 280.0, at[bed])
    rows += PR.cart_rows(tr.pos(s), T, Rt, Ut)
    rows += PR.rider_hands(eye, F, R, U, up=hands_up(sg.world, tr, s))
    meta = ride.metas.get(sg.world, {})
    if 'portal' in meta:
        pt = meta['portal']
        rows += PR.nether_portal_rows(pt['center'], pt['width'], pt['height'], pt['normal'], tau)
    return dict(world=sg.world, env=looks.get(sg.world), cam=cam, rows=rows, lights=[], particles=None,
                seg=sg, t=t, tau=tau, s=s, v=v, g=g, eye=eye, axes=(F, R, U), frame=(T, Rt, Ut), tr=tr)
