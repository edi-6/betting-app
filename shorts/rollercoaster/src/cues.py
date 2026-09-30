"""The cue sheet the soundtrack is built from: for every video frame, where the cart is and how it moves (speed,
g-force, on the lift chain or in the air, how slow the slow motion is), where the camera looks (to place sounds
left and right), where the things that make noise are (the ghasts, the fireballs, the dragon, the fire on the
bridge, the lava falls, the crystals); and the moments the sound has to hit, in video seconds.

It needs no renderer: the ride is deterministic, the worlds' metadata comes from the cache.
"""
import json

import numpy as np

import fx
import ride as RD
import rider


def load_metas():
    from main import WORLDS
    return {key: RD.load_meta(key, mod.build, [src]) for key, mod, src in WORLDS}


def _nearest(points, eye):
    if not points:
        return None
    P = np.asarray([p[:3] for p in points], float)
    d = np.linalg.norm(P - eye, axis=1)
    i = int(np.argmin(d))
    return P[i]


def build(ride=None):
    ride = ride or RD.Ride(load_metas())
    fx.RD_META = ride.metas
    o, n, e = ride.tracks['over'], ride.tracks['nether'], ride.tracks['end']
    dp, sf = ride.tracks['deep'], ride.tracks['sift']
    ns = fx.NetherStory(n, ride.metas.get('nether', {}))
    es = fx.EndStory(e, ride.metas.get('end', {}))
    ds = fx.DeepStory(dp, ride.metas.get('deep', {}))
    fs = fx.SiftStory(sf, ride.metas.get('sift', {}))
    fx._STORY.update(nether=ns, end=es, deep=ds, sift=fs)
    mdp, msf = ride.metas.get('deep', {}), ride.metas.get('sift', {})
    mo, mn, me = ride.metas.get('over', {}), ride.metas.get('nether', {}), ride.metas.get('end', {})
    falls = [(x, y, 60.0) for (x, y, wd) in mn.get('lava_falls', [])]
    fires = list(mn.get('fires', []))
    crystals = [(x, y, z + 1.0) for (x, y, z, r, c) in me.get('pillars', [])]
    frames = []
    for f in range(ride.nframes):
        sg, t, tau = ride.locate(f)
        tr = sg.tr
        s = tr.s_at(tau)
        cam, eye, (F, R, U), _, v, g = rider.view(tr, s, tau, look_dist=RD.look_dist(sg.world, tr, s),
                                                  attend=fx.attention(sg.world, tr, tau),
                                                  jolt=fx.jolt(sg.world, tr, tau))
        c = dict(w=sg.world, seg=sg.name, tau=round(tau, 5), s=round(s, 3), v=round(v, 3), g=round(float(g), 3),
                 k=round(sg.speed_factor(t), 4), air=bool(tr.in_gap(s)),
                 lift=bool(any(a <= s <= b for (a, b, _) in tr.lift)), pw=bool(tr.is_powered(s)),
                 eye=[round(x, 3) for x in eye], F=[round(x, 4) for x in F], R=[round(x, 4) for x in R],
                 U=[round(x, 4) for x in U])
        src = {}
        if sg.world == 'over' and 'waterfall' in mo:
            Pc, rt, th = mo['waterfall']
            d = float(np.dot(eye - np.asarray(Pc), np.asarray(th)))
            lat = float(abs(np.dot(eye - np.asarray(Pc), np.asarray(rt))))
            src['fall'] = [float(Pc[0]), float(Pc[1]), 24.0]
            c['curtain'] = round(d, 3) if lat < 8.0 else None
        if sg.world == 'nether':
            src['ghast1'] = ns.ghast_pos(1, tau).tolist()
            src['ghast2'] = ns.ghast_pos(2, tau).tolist()
            for which in (1, 2):
                p, d, alive = ns.fireball(which, tau)
                if alive:
                    src[f'fb{which}'] = p.tolist()
            q = _nearest(falls, eye)
            if q is not None:
                src['lavafall'] = [float(q[0]), float(q[1]), float(min(max(eye[2], 32.0), 90.0))]
            q = _nearest(fires, eye)
            if q is not None:
                src['fire'] = q.tolist()
        if sg.world == 'end':
            src['dragon'] = np.asarray(es.path(tau)[0]).tolist()
            q = _nearest(crystals, eye)
            if q is not None:
                src['crystal'] = q.tolist()
        if sg.world == 'deep' and 'gate' in mdp:
            src['gate'] = np.asarray(mdp['gate']['center'], float).tolist()
        if sg.world == 'sift':
            q = _nearest([b[0] for b in fs.blubs], eye)
            if q is not None and np.linalg.norm(q - eye) < 30.0:
                src['blub'] = q.tolist()
            if 'portals' in msf:
                src['rift'] = np.asarray(msf['portals']['rift']['center'], float).tolist()
        c['src'] = {k: [round(x, 2) for x in v_] for k, v_ in src.items()}
        frames.append(c)

    vt = ride.video_time
    ev = {
        'crest': 0.0,
        'bottom': vt('A', o.time(o.marks['bottom'])),
        'airtime': vt('A', o.time(o.marks['arch'] + 32.0)),
        'portal_a': float(ride.starts[1]),
        'nether_drop': vt('B', n.time(n.marks['drop'])),
        'ghast_moan': vt('B', max(ns.t_shoot - 1.5, n.time(1.0))),
        'ghast_shoot': vt('B', ns.t_shoot),
        'explosion': vt('B', ns.t_hit),
        'takeoff': vt('B', ns.t_take),
        'fb2_shoot': vt('B', ns.t_shoot2),
        'fb2_pass': vt('B', ns.t_mid),
        'land': vt('B', ns.t_land),
        'bridge': vt('B', n.time(n.marks['bridge'])),
        'portal_b': float(ride.starts[2]),
        'swoop': vt('C', es.kt[4]),
        'perch': vt('C', min(es.t_perch, ride.segments[2].tau1)),
        'roar': vt('C', min(es.t_roar, ride.segments[2].tau1)),
        'portal_c': float(ride.starts[3]),
        'wake': vt('E', ds.t_wake),
        'gate': float(ride.starts[4]),
        'sift_drop': vt('F', sf.time(sf.marks['drop'])),
        'meadow': vt('F', sf.time(sf.marks['meadow'])),
        'lake': vt('F', sf.time(sf.marks['lake'])),
        'fossil': vt('F', sf.time(sf.marks['fossil'])),
        'rift': float(ride.starts[5]),
        'end': float(ride.duration),
        'toast_b': float(ride.starts[1]) + 0.7,
        'toast_c': float(ride.starts[2]) + 0.7,
        'toast_f': float(ride.starts[4]) + 0.8,
    }
    # the waterfall: when the eye goes through the curtain
    for i in range(1, len(frames)):
        a, b = frames[i - 1].get('curtain'), frames[i].get('curtain')
        if a is not None and b is not None and a < 0.25 <= b:
            ev['waterfall'] = i / RD.FPS
            break
    # debris hitting the lava: the times (video) of the span's pieces going in
    sinks = sorted(pc['t_sink'] for pc in ns.pieces if pc['t_sink'] < 1e8)
    ev['sinks'] = [vt('B', t) for t in sinks if t < ride.segments[1].tau1]
    return dict(fps=RD.FPS, nframes=ride.nframes, duration=ride.duration, events=ev, frames=frames,
                segments=[dict(name=sg.name, world=sg.world, start=float(t0), dur=sg.duration)
                          for sg, t0 in zip(ride.segments, ride.starts)])


def write(path, ride=None):
    cs = build(ride)
    with open(path, 'w') as fh:
        json.dump(cs, fh)
    return cs


if __name__ == '__main__':
    import sys
    cs = write(sys.argv[1] if len(sys.argv) > 1 else 'cues.json')
    for k, v in cs['events'].items():
        if k != 'sinks':
            print(f'  {k:12s} {v:7.3f}')
    print('  sinks', len(cs['events']['sinks']), 'from', round(min(cs['events']['sinks']), 2), 'to',
          round(max(cs['events']['sinks']), 2))
