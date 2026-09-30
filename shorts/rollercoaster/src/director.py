"""Each frame of the ride: the base scene (ride.base_scene), the events of its world (splashes, the ghast, the
dragon...), the render, the transitions between dimensions, and the interface (the speed on the action bar, the
advancement toasts).
"""
import numpy as np

import fx
import hud as HUD
import props as PR
import ride as RD


def register(r):
    fx.register(r)


def events(sc):
    """World events for the frame: extra rows, point lights and particles."""
    w = sc['world']
    if w == 'over':
        return fx.over_events(sc)
    if w == 'nether':
        return fx.nether_events(sc)
    if w == 'end':
        return fx.end_events(sc)
    return [], [], None


def transition(ride, f):
    """(kind, amount 0..1, phase) of the overlay at a cut between dimensions, or None."""
    tv = f / RD.FPS
    st = ride.starts
    segs = ride.segments
    for i, sg in enumerate(segs):
        end = st[i + 1]
        nxt = segs[(i + 1) % len(segs)]
        kind = {('A', 'B'): 'nether', ('B', 'C'): 'end', ('C', 'D'): 'white'}.get((sg.name, nxt.name))
        if kind is None:
            continue
        lead = {'nether': 0.30, 'end': 0.24, 'white': 0.16}[kind]
        tail = {'nether': 0.45, 'end': 0.50, 'white': 0.70}[kind]
        if end - lead <= tv < end:
            u = (tv - (end - lead)) / lead
            return kind, float(u * u), tv - end
        if end <= tv < end + tail:
            u = 1.0 - (tv - end) / tail
            return kind, float(u * u * (3 - 2 * u)), tv - end
    return None


def render_frame(r, ride, f, hud=True):
    fx.RD_META = ride.metas
    sc = RD.base_scene(ride, f)
    sc['renderer'] = r
    rows, lights, parts = events(sc)
    sc['rows'] += rows
    r.use_world(sc['world'])
    r.time = sc['tau']
    inst = PR.rows_to_instances(sc['rows'])
    lt = np.array(lights[:16], np.float32) if lights else None
    r.render(sc['cam'], sc['env'], instances=inst, lights=lt, particles=parts, streaks=sc.get('streaks'),
             clip_z=(-1e9, 1e9))
    img = r.finish(sc['env'])
    img = fx.overlays(img, sc)
    tr = transition(ride, f)
    if tr is not None:
        img = fx.transition_overlay(img, *tr)
    if hud:
        img = HUD.draw(img, ride, f, sc)
    return img
