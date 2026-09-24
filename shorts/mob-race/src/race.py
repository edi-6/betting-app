"""The race, simulated with pymunk (Chipmunk2D) in the plane of the course: eight marbles, the course's blocks,
beams and pegs, and its machinery (the starting floor, the trapdoors, pistons, TNT), stepped per video frame in
fixed sub-steps so a race is deterministic for a given seed and line-up.

Each trapdoor counts the marbles that roll over it; once `keep` are across it swings open and whatever comes after
drops into the lava basin under it. A marble in lava is out (it keeps sinking and burning for the renderer).
"""
import numpy as np
import pymunk

import course as C

FPS = 30
SUB = 10                     # physics sub-steps per video frame
G = 34.0                     # gravity (blocks / s^2)
GO_STEP = 150                # the stalls open after this many physics steps
R = C.R
MATS = {                     # friction, elasticity (pymunk multiplies them with the marble's)
    'stone_bricks': (0.8, 0.35), 'stone': (0.8, 0.35), 'netherrack': (0.8, 0.3), 'nether_bricks': (0.7, 0.3),
    'gold': (0.9, 0.2), 'iron_bars': (0.4, 0.4), 'planks': (0.7, 0.35), 'fence': (0.3, 0.7), 'ice': (0.02, 0.25),
    'slime': (0.9, 1.45), 'piston': (0.6, 0.6), 'tnt': (0.6, 0.3),
}
MARBLE_F, MARBLE_E = 0.9, 0.78


class Race:
    def __init__(self, course, lineup, seed=0, jitter=0.02):
        """lineup: marble names in stall order (left to right)."""
        self.c = course
        self.names = list(lineup)
        self.rng = np.random.default_rng(seed)
        sp = pymunk.Space()
        sp.gravity = (0.0, -G)
        sp.iterations = 30
        sp.damping = 0.985
        self.space = sp
        self.t = 0.0
        self.frame = 0
        self.steps = 0
        self.events = []
        self._ev = []
        self.pending = []
        # static geometry
        sb = sp.static_body
        for p in course.pieces:
            if p['kind'] == 'box':
                shape = pymunk.Poly(sb, [(p['x0'], p['z0']), (p['x1'], p['z0']), (p['x1'], p['z1']),
                                         (p['x0'], p['z1'])])
            elif p['kind'] == 'beam':
                shape = pymunk.Segment(sb, p['a'], p['b'], p['t'] / 2.0)
            else:
                shape = pymunk.Circle(sb, p['r'], offset=p['c'])
            shape.friction, shape.elasticity = MATS[p['mat']]
            shape.collision_type = 1
            shape.mat = p['mat']
            sp.add(shape)
        # the starting floor
        x0, z0, x1, z1 = course.start['floor']
        self.floor = pymunk.Poly(sb, [(x0, z0), (x1, z0), (x1, z1), (x0, z1)])
        self.floor.friction, self.floor.elasticity = 0.8, 0.2
        self.floor.mat = 'planks'
        sp.add(self.floor)
        self.go = False
        self.go_t = None
        # trapdoors: kinematic plates hinged at their upstream end
        self.traps = []
        for tr in course.traps:
            body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
            body.position = tr['hinge']
            body.angle = tr['angle']
            seg = pymunk.Segment(body, (0.0, 0.0), (tr['length'], 0.0), 0.25)
            seg.friction, seg.elasticity = 0.7, 0.3
            seg.collision_type = 1
            seg.mat = 'trapdoor'
            sp.add(body, seg)
            self.traps.append(dict(tr, body=body, count=[], open_at=None))
        # holding pens: floors that go at the end of a round
        self.pens = []
        for pn in course.pens:
            seg = pymunk.Segment(sb, pn['a'], pn['b'], 0.25)
            seg.friction, seg.elasticity = MATS[pn['mat']]
            seg.collision_type = 1
            seg.mat = pn['mat']
            sp.add(seg)
            self.pens.append(dict(pn, shape=seg, release=None))
        # pistons
        self.pistons = []
        for p in course.pistons:
            body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
            body.position = (p['x'] - p['dir'] * 0.5, p['z'])        # at rest: inside the wall
            shape = pymunk.Poly.create_box(body, (1.0, 1.0))
            shape.friction, shape.elasticity = MATS['piston']
            shape.collision_type = 1
            shape.mat = 'piston'
            sp.add(body, shape)
            self.pistons.append(dict(p, body=body, ext=0.0))
        # TNT
        self.tnt = []
        for t in course.tnt:
            shape = pymunk.Poly(sb, [(t['x'] - 0.5, t['z'] - 0.5), (t['x'] + 0.5, t['z'] - 0.5),
                                     (t['x'] + 0.5, t['z'] + 0.5), (t['x'] - 0.5, t['z'] + 0.5)])
            shape.friction, shape.elasticity = MATS['tnt']
            shape.collision_type = 1
            shape.mat = 'tnt'
            sp.add(shape)
            self.tnt.append(dict(t, shape=shape, lit=None, boom=None))
        # the marbles
        self.marbles = []
        for k, name in enumerate(self.names):
            body = pymunk.Body(1.0, 0.4 * R * R)
            body.position = (course.start['xs'][k] + self.rng.uniform(-jitter, jitter), course.start['z'] + 0.02)
            body.angle = self.rng.uniform(-0.3, 0.3)
            shape = pymunk.Circle(body, R)
            shape.friction, shape.elasticity = MARBLE_F, MARBLE_E
            shape.collision_type = 2
            shape.marble = k
            sp.add(body, shape)
            self.marbles.append(dict(name=name, body=body, shape=shape, out=None, stage=0, finish=None,
                                     sink=None))
        sp.on_collision(2, 1, post_solve=self._hit)
        sp.on_collision(2, 2, post_solve=self._hit)

    # ------------------------------------------------------------------------------------------
    def _hit(self, arb, space, data):
        imp = arb.total_impulse.length
        if imp < 0.6:
            return
        a, b = arb.shapes
        k = a.marble
        other = getattr(b, 'mat', None) if not hasattr(b, 'marble') else 'marble'
        p = arb.contact_point_set.points[0].point_a if arb.contact_point_set.points else a.body.position
        self._ev.append(('hit', k, other, float(imp), (float(p.x), float(p.y))))

    def start(self):
        if not self.go:
            self.space.remove(self.floor)
            self.go = True
            self.go_t = self.t
            self.pending = [('go',)]

    def alive(self):
        return [m for m in self.marbles if m['out'] is None]

    def _piston(self, p, t):
        """Extension 0..1 over one period: a long rest, then a fast punch out, a short hold and back."""
        u = ((t / p['period']) + p['phase']) % 1.0
        if u < 0.70:
            return 0.0
        if u < 0.78:
            return (u - 0.70) / 0.08
        if u < 0.88:
            return 1.0
        if u < 0.98:
            return 1.0 - (u - 0.88) / 0.10
        return 0.0

    def _sub(self, dt):
        t = self.t
        # pistons: kinematic, moved by velocity so they shove what they hit
        for p in self.pistons:
            e = self._piston(p, t + dt) if self.go else 0.0
            if e > 0.0 and p['ext'] == 0.0:
                self._ev.append(('piston', self.pistons.index(p)))
            target = p['x'] + p['dir'] * (-0.5 + p['reach'] * e)
            p['body'].velocity = ((target - p['body'].position.x) / dt, 0.0)
            p['ext'] = e
        # trapdoors: once open, swing down to hang from the hinge
        for tr in self.traps:
            if tr['open_at'] is not None:
                w = (-np.pi / 2 - tr['body'].angle) / dt
                tr['body'].angular_velocity = float(np.clip(w, -14.0, 14.0))
            else:
                tr['body'].angular_velocity = 0.0
        self.space.step(dt)
        self.t += dt
        # triggers
        for m in self.alive():
            x, z = m['body'].position
            for ti, tr in enumerate(self.traps):
                if m['name'] in tr['count'] or m['stage'] != ti:
                    continue
                if abs(z - tr['hinge'][1]) < 3.0 and (x - tr['count_x']) * tr['dir'] > 0:
                    tr['count'].append(m['name'])
                    m['stage'] = ti + 1
                    if tr['open_at'] is not None:
                        # made it across as the trapdoor went: safe
                        tr.setdefault('saved', []).append(m['name'])
                        self._ev.append(('saved', ti, m['name']))
                    else:
                        self._ev.append(('through', ti, m['name']))
                    if tr['open_at'] is None and len(tr['count']) >= tr['keep']:
                        tr['open_at'] = self.t
                        self._ev.append(('open', ti))
            for (x0, z0, x1, z1) in self.c.lava:
                if x0 < x < x1 and z0 < z < z1:
                    m['out'] = self.t
                    m['sink'] = (float(x), float(z), float(m['body'].velocity.x), float(m['body'].angle))
                    self.space.remove(m['body'], m['shape'])
                    self._ev.append(('lava', m['name'], (float(x), float(z))))
                    break
        # a marble that has stopped where it shouldn't (balanced on a peg, wedged) gets the smallest nudge
        if self.go:
            for m in self.alive():
                b = m['body']
                k = m['stage']
                if k > 0 and (k > len(self.pens) or self.pens[k - 1]['release'] is None):
                    continue                        # waiting in a pen (or on the podium) is fine
                if b.velocity.length < 0.15:
                    m['still'] = m.get('still', 0.0) + dt
                    if m['still'] > 0.4:
                        m['still'] = 0.0
                        b.apply_impulse_at_local_point((float(self.rng.choice([-1.0, 1.0]) * 1.4), 0.6))
                        self._ev.append(('nudge', m['name']))
                else:
                    m['still'] = 0.0
        # a round is over when its losers are all in the lava: then the pen's floor opens
        for k, (tr, pn) in enumerate(zip(self.traps, self.pens)):
            if tr['open_at'] is None or pn['release'] is not None or pn.get('podium'):
                continue
            behind = [m for m in self.marbles if m['stage'] == k]
            if all(m['out'] is not None for m in behind):
                t_last = max([m['out'] for m in behind] + [tr['open_at']])
                if self.t - t_last > 1.1:
                    pn['release'] = self.t
                    self.space.remove(pn['shape'])
                    self._ev.append(('release', k))
                    # the next round drops all but two (all but one in the final)
                    if k + 1 < len(self.traps):
                        n = len(self.alive())
                        nxt = self.traps[k + 1]
                        nxt['keep'] = 1 if k + 2 == len(self.traps) else max(1, n - 2)
        # TNT: a touch lights it, it goes off a moment later
        for i, tn in enumerate(self.tnt):
            if tn['boom'] is not None:
                continue
            if tn['lit'] is None:
                for m in self.alive():
                    x, z = m['body'].position
                    if abs(x - tn['x']) < 0.5 + R + 0.05 and abs(z - tn['z']) < 0.5 + R + 0.05:
                        tn['lit'] = self.t
                        self._ev.append(('lit', i))
                        break
            elif self.t - tn['lit'] > 0.3:
                tn['boom'] = self.t
                self.space.remove(tn['shape'])
                for m in self.alive():
                    d = np.array(m['body'].position) - np.array([tn['x'], tn['z']])
                    L = np.linalg.norm(d)
                    if L < 4.5:
                        imp = d / max(L, 0.3) * 14.0 * (1.0 - L / 4.5) + np.array([0.0, 6.0 * (1.0 - L / 4.5)])
                        m['body'].apply_impulse_at_local_point((float(imp[0]), float(imp[1])))
                self._ev.append(('boom', i, (tn['x'], tn['z'])))

    def step(self, n):
        """Advance n fixed physics steps of 1 / (FPS * SUB) s (a video frame is a whole number of them, so slow
        motion never changes the race); starts the race after GO_STEP steps. Returns the events."""
        self._ev = self.pending
        self.pending = []
        dt = 1.0 / (FPS * SUB)
        for _ in range(n):
            if self.steps == GO_STEP:
                self.start()
                self._ev.extend(self.pending)
                self.pending = []
            self._sub(dt)
            self.steps += 1
        self.frame += 1
        return self._ev

    def step_frame(self, scale=1.0):
        return self.step(max(1, int(round(SUB * scale))))

    # ------------------------------------------------------------------------------------------
    def state(self):
        """Positions, angles, speeds of the marbles; the machinery."""
        ms = []
        for m in self.marbles:
            b = m['body']
            ms.append(dict(name=m['name'], x=float(b.position.x), z=float(b.position.y), a=float(b.angle),
                           vx=float(b.velocity.x), vz=float(b.velocity.y), w=float(b.angular_velocity),
                           out=m['out'], stage=m['stage'], sink=m['sink']))
        traps = [dict(angle=float(tr['body'].angle), open=tr['open_at'], count=list(tr['count']))
                 for tr in self.traps]
        pens = [pn['release'] for pn in self.pens]
        pistons = [float(p['ext']) for p in self.pistons]
        tnt = [dict(lit=t['lit'], boom=t['boom']) for t in self.tnt]
        return dict(t=self.t, go=self.go, go_t=self.go_t, marbles=ms, traps=traps, pens=pens, pistons=pistons,
                    tnt=tnt)

    def standings(self):
        """Marble names from first to last: survivors by how far down the course they are, then the eliminated
        (latest out first)."""
        alive = sorted(self.alive(), key=lambda m: (-m['stage'], m['body'].position.y))
        out = sorted([m for m in self.marbles if m['out'] is not None], key=lambda m: -m['out'])
        return [m['name'] for m in alive] + [m['name'] for m in out]
