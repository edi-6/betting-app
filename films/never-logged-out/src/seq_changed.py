"""ACT 7: the world changes (9:15 - 10:45).

The game unfreezes somewhere else: the village, moved and turned, under a green sky and an enormous moon; dead
spruces; the villagers all standing still, all named NOAH, all watching him. He walks out in a straight line and walks
back into the same plaza. Signs: LEFT. LEFT. LEFT. YOU'RE GOING THE WRONG WAY. Past the last one, where his spawn
was, a crater, and at the bottom, lit, a white house: his real house. His real room. On his monitor, a video
player, at this minute of this video, showing him from behind.
"""
import numpy as np

import anim as A
import common as C
import entities as EN
import film as FM
import looks
import screen as SCR

W_ = 'changed'
HOTBAR = dict(items=['written_book', 'torch', 'bread', 'clock', 'filled_map', 'torn_page'], selected=6,
              counts={1: 16, 2: 5})
NOAHS = [((-14.5, 4.5), -30), ((-4.5, 6.5), 60), ((-10.5, 16.5), 170), ((-3.5, 14.5), 120), ((-16.5, 12.5), -110),
         ((-7.5, 1.5), 10), ((0.5, 10.5), 90)]


def noahs(look_at=None):
    """Every villager is NOAH now, and they stand still and stare (their heads follow him)."""
    skins = ['villager_farmer', 'villager_librarian', 'villager_cartographer', 'villager_cleric', 'villager_smith',
             'villager_shepherd', 'villager_noah']
    out = []
    for k, ((x, y), yaw) in enumerate(NOAHS):
        a = EN.Actor('villager', skins[k % len(skins)], (x, y, 0.0), yaw=np.radians(yaw))
        a.name = 'NOAH'
        if look_at is not None:
            want = np.radians(A.yaw_of(np.asarray(look_at)[:2] - np.array([x, y])))
            d = (want - a.yaw + np.pi) % (2 * np.pi) - np.pi
            a.yaw += d * 0.55                          # the body turns part of the way...
            a.head_yaw = float(np.clip(d * 0.45, -1.1, 1.1))     # ...the head the rest
        out.append(a)
    return out


def shots(ctx):
    S = FM.Shot
    out = []
    w = ctx.worlds[W_]
    P = w.points

    def gz(x, y):
        return C.ground_z(ctx, x, y, W_)

    def pov(pos_keys, **kw):
        pk = [(t, (x, y, gz(x, y) if z is None else z)) for (t, (x, y, z)) in pos_keys]
        return A.POV(pos_keys=pk, **kw)

    CLIP = (-32.0, 80.0)               # draw the crater (the default culls everything below z = -8 from above)

    def scene(pv, t, env='wrong', actors=(), props=None, **kw):
        return C.pov_scene(pv, t, env, world=W_, actors=list(actors), clip_z=CLIP,
                           props=C.world_props(ctx, W_) if props is None else props, **kw)

    def tags(pv, look=True):
        def ov(img, t, T, ctx_, film):
            C.name_tags(img, pv.cam(t), noahs(pv.feet(t) if look else None))
        return ov

    moon_yaw = A.yaw_of(looks.PRESETS['wrong']['moon_dir'][:2])

    # --- w1: the game unfreezes. Somewhere else ---------------------------------------------------------------------------
    pov1 = pov([(0, (-3.0, 0.2, None)), (6.0, (-3.2, 0.6, None))], yaw_keys=[(0, 36), (2.2, 58), (4.4, 14), (6.0, 24)],
               pitch_keys=[(0, 2), (3.0, 6), (6.0, 3)], seed=90, jitter=0.1)

    def w1(t, T):
        return scene(pov1, t, actors=noahs(pov1.feet(t)))

    def w1_fx(t):
        if t < 0.35:
            return {'tear': 0.9, 'vhs': 0.8, 'freeze': 0.6 * (1 - t / 0.35)}
        return {}

    out.append(S('w1_unfreeze', 6.0, '3d', scene=w1, overlay=tags(pov1), hud=HOTBAR, fx=w1_fx,
                 subs=[(2.4, 5.0, "Wait. This isn't where I was.")],
                 cues=[(0.0, 'unfreeze'), (0.3, 'amb', {'kind': 'wrong'})]))

    # --- w2: the sky is the wrong colour. The moon ------------------------------------------------------------------------
    pov2 = pov([(0, (-3.2, 0.6, None)), (7.0, (-3.3, 0.4, None))],
               yaw_keys=[(0, 24), (3.2, moon_yaw - 8), (7.0, moon_yaw - 4)],
               pitch_keys=[(0, 3), (3.2, 8), (7.0, 12)], seed=91, jitter=0.08)

    def w2(t, T):
        return scene(pov2, t, actors=noahs(pov2.feet(t)))

    out.append(S('w2_moon', 7.0, '3d', scene=w2, overlay=tags(pov2), hud=HOTBAR,
                 subs=[(4.2, 6.6, 'What happened to the sky?')],
                 cues=[(2.0, 'drone', {'level': 0.4, 'dur': 5.0})]))

    # --- w3: every one of them is NOAH ------------------------------------------------------------------------------------
    path3 = A.Path([(-3.3, 0.4, 0.0), (-5.0, 3.0, 0.0), (-6.4, 5.4, 0.0)])
    pov3 = A.POV(path=path3, s_keys=[(0, 0.0), (4.0, path3.length), (8.0, path3.length)],
                 yaw_keys=[(0, moon_yaw - 4), (1.6, 40), (4.0, 70), (6.0, 150), (8.0, 100)],
                 pitch_keys=[(0, 10), (1.6, 0), (8.0, -2)], seed=92, jitter=0.1)

    def w3(t, T):
        return scene(pov3, t, actors=noahs(pov3.feet(t)))

    out.append(S('w3_noahs', 8.0, '3d', scene=w3, overlay=tags(pov3), hud=HOTBAR,
                 subs=[(4.6, 7.6, "They're all... Noah.")],
                 cues=[(ts, 'step', {'surface': 'grass'}) for ts in pov3.steps(0, 4.0)] + [(2.0, 'villager_no',
                                                                                              {'pan': 0.3})]))

    # --- w4: he walks out in a straight line... -------------------------------------------------------------------------------
    path4 = A.Path([(-6.4, 5.4, 0.0), (-3.0, 1.0, 0.0), (4.0, -1.0, 0.0), (16.0, -2.5, 0.0), (26.0, -6.0, 0.0)])
    pov4 = A.POV(path=path4, s_keys=[(0, 0.0), (7.0, path4.length)],
                 yaw_keys=[(0, 100), (1.2, -60), (2.4, -80), (7.0, -82)], pitch_keys=[(0, -2), (7.0, -3)], seed=93,
                 jitter=0.1)

    def w4(t, T):
        return scene(pov4, t, actors=noahs(pov4.feet(t)))

    out.append(S('w4_leave', 7.0, '3d', scene=w4, hud=HOTBAR,
                 fx=lambda t: {'fade': float(np.clip((7.0 - t) / 0.8, 0, 1))},
                 subs=[(0.6, 2.8, "Okay. I'm getting out of here.")],
                 cues=[(ts, 'step', {'surface': 'path'}) for ts in pov4.steps(0, 7.0)]))

    # --- w4b: ...and walks back into the same plaza, from the other side -----------------------------------------------------------
    path4b = A.Path([(-28.0, 9.0, 0.0), (-22.0, 9.4, 0.0), (-17.2, 9.6, 0.0)])
    pov4b = A.POV(path=path4b, s_keys=[(0, 0.0), (4.4, path4b.length), (7.0, path4b.length)],
                  yaw_keys=[(0, -90), (4.4, -92), (5.2, -60), (6.2, -110), (7.0, -96)],
                  pitch_keys=[(0, -3), (7.0, -2)], seed=94, jitter=0.1)

    def w4b(t, T):
        return scene(pov4b, t, actors=noahs(pov4b.feet(t)))

    out.append(S('w4b_back', 7.0, '3d', scene=w4b, overlay=tags(pov4b), hud=HOTBAR,
                 fx=lambda t: {'fade': float(np.clip(t / 0.6, 0, 1))},
                 subs=[(4.6, 7.0, 'No. I walked in a straight line.')],
                 cues=[(ts, 'step', {'surface': 'path'}) for ts in pov4b.steps(0, 4.4)] + [(4.4, 'sting_low')]))

    # --- w5: the signs: LEFT, LEFT, LEFT... YOU'RE GOING THE WRONG WAY -----------------------------------------------------------
    facing_off = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
    route = [(0, 3.0, None), (1, 2.4, None), (2, 2.4, None), (3, 5.4, '...Wrong way from where?')]
    for k, dur, line in route:
        sp = P[f'route_{k}']
        sign = next(s for s in w.signs if s.get('key') == f'route_{k}')
        dx, dy = facing_off[sign['facing']]
        board = np.array([sp[0] + dx * 0.06, sp[1] + dy * 0.06, sp[2] + 1.82])
        d0, d1 = (1.9, 1.35) if k < 3 else (2.6, 1.6)
        p0 = np.array([sp[0] + dx * d0 + 0.2 * dy, sp[1] + dy * d0 + 0.2 * dx])
        p1 = np.array([sp[0] + dx * d1, sp[1] + dy * d1])
        e0, e1 = np.array([*p0, sp[2] + 1.62]), np.array([*p1, sp[2] + 1.62])
        aim = board + (0, 0, 0.12) if k < 3 else board + (0, 0, 0.45)
        pk = A.POV(pos_keys=[(0, (*p0, sp[2])), (dur, (*p1, sp[2]))],
                   yaw_keys=[(0, A.look_angles(e0, aim)[0]), (dur, A.look_angles(e1, aim)[0])],
                   pitch_keys=[(0, A.look_angles(e0, aim)[1]), (dur, A.look_angles(e1, aim)[1])], seed=95 + k,
                   jitter=0.06)

        def w5(t, T, pk=pk):
            return scene(pk, t)

        cues = [(0.0, 'step', {'surface': 'grass'})]
        if k == 3:
            cues += [(0.2, 'silence_all', {'dur': 2.0}), (2.2, 'drone', {'level': 0.45, 'dur': 3.2})]
        out.append(S(f'w5_sign{k + 1}', dur, '3d', scene=w5, hud=HOTBAR,
                     subs=[(2.6, 5.2, line)] if line else [], cues=cues))

    # --- w6: where his spawn was: a crater. At the bottom, lit, a white house -----------------------------------------------------
    crater_env = dict(fog=(0.0025, 30.0, -24.0, 110.0), fog_sun=(0.3, 0.36, 0.26))
    you6 = EN.Actor('player', 'you', (6.5, -31.2, gz(6.5, -31.2)), yaw=np.radians(180.0))
    eye6 = A.Keys([(0, (8.8, -23.4, 11.5)), (9.0, (8.2, -25.2, 9.8))])
    tgt6 = A.Keys([(0, (6.4, -62.0, -22.0)), (9.0, (6.5, -64.0, -23.0))])
    rh = P['real_door']
    house_glow = [[rh[0] + dx, rh[1] + dy, rh[2] + 2.8, 9.0, 1.4, 0.9, 0.5] for (dx, dy) in ((-5.5, -1.5), (6.0, -1.5),
                                                                                         (-5.5, 11.5), (6.0, 11.5))]
    house_glow.append([rh[0], rh[1] - 1.2, rh[2] + 1.5, 7.0, 1.2, 0.8, 0.45])            # the light by the door

    def w6(t, T):
        you6.idle_t = t
        return dict(world=W_, env=looks.get('wrong', **crater_env), cam=dict(eye=eye6(t), target=tgt6(t), fov=56),
                    actors=[you6], props=C.world_props(ctx, W_), clip_z=CLIP, lights=house_glow)

    out.append(S('w6_crater', 9.0, '3d', scene=w6,
                 subs=[(3.4, 5.8, "That's... that's my house."), (6.2, 8.8, 'My real house.')],
                 cues=[(0.0, 'wind_crater'), (1.0, 'drone', {'level': 0.35, 'dur': 8.0})]))

    # --- w7: at the bottom, walking up to the door -------------------------------------------------------------------------------
    door = P['real_door']
    path7 = A.Path([(6.3, -83.0, gz(6.3, -83.0)), (6.6, -78.0, gz(6.6, -78.0)), (6.5, -73.0, gz(6.5, -73.0)),
                    (6.5, door[1] - 0.4, door[2])])
    pov7 = A.POV(path=path7, s_keys=[(0, 0.0), (6.0, path7.length), (8.0, path7.length)],
                 yaw_keys=[(0, 6), (2.4, -8), (4.4, 4), (8.0, 0)],
                 pitch_keys=[(0, 22), (2.4, 30), (4.4, 0), (8.0, -4)], seed=99, jitter=0.1)

    def w7(t, T):
        return scene(pov7, t, env=looks.get('wrong', **crater_env), lights=house_glow)

    out.append(S('w7_bottom', 8.0, '3d', scene=w7, hud=HOTBAR,
                 cues=[(ts, 'step', {'surface': 'grass'}) for ts in pov7.steps(0, 6.0)]))

    # --- w8: his room ----------------------------------------------------------------------------------------------------------
    mon = SCR.Monitor((P['room_monitor'][0], P['room_monitor'][1] + 0.38, P['room_monitor'][2]), facing=0)
    feet_in = np.array([door[0], door[1] + 2.4, door[2]])
    behind_chair = np.array([P['room_chair'][0] - 0.1, P['room_chair'][1] - 1.55, door[2]])
    t_door = 1.0
    door8 = A.Keys([(0, 0.0), (t_door, 0.0), (t_door + 0.3, 90.0, 'out')])
    a_mon = A.look_angles(behind_chair + (0, 0, 1.62), mon.centre)
    pov8 = A.POV(pos_keys=[(0, (door[0], door[1] - 0.4, door[2])), (t_door + 0.4, (door[0], door[1] - 0.4, door[2])),
                           (3.2, tuple(feet_in)), (7.0, tuple(behind_chair)), (10.0, tuple(behind_chair))],
                 yaw_keys=[(0, 0), (3.2, 30), (4.6, -40), (5.8, 10), (7.0, a_mon[0]), (10.0, a_mon[0])],
                 pitch_keys=[(0, -2), (3.2, 4), (5.8, -6), (7.0, a_mon[1]), (10.0, a_mon[1])], seed=100, jitter=0.07)
    room_env = dict(exposure=0.85)
    shared = {'img': None}

    def body_at(pv, t):
        a = EN.Actor('player', 'you', pv.feet(t), yaw=np.radians(pv.angles(t)[0]))
        a.idle_t = t
        a.head_pitch = -np.radians(pv.angles(t)[1]) * 0.7
        return a

    def room_scene(pv, t, T, env_kw, door_deg=90.0, film_him=False, swing=0.0):
        props = C.world_props(ctx, W_, real_door=door_deg) + mon.props(glow=0.5)
        lights = mon.light()
        if film_him:
            # the second camera: up in the corner behind him, looking at his back and the screen
            head = pv.feet(t) + np.array([0.0, 0.0, 1.55])
            cam2 = dict(eye=head + np.array([-0.9, -1.8, 0.75]), target=head + np.array([0.35, 1.5, -0.35]), fov=52)
            sc2 = dict(world=W_, env=looks.get('wrong', **env_kw), cam=cam2, actors=[body_at(pv, t)], props=props,
                       lights=lights, clip_z=CLIP, prep=lambda r: r.update_kind_layer('screen', 0, SCR.Monitor.content(
                           shared['img'], T)))
            shared['img'] = FM.render_scene(ctx, sc2, main=False)
        img = shared['img']

        def prep(r):
            r.update_kind_layer('screen', 0, SCR.Monitor.content(img, T))
        sc = scene(pv, t, env=looks.get('wrong', **env_kw), props=props, lights=lights, swing=swing)
        sc['prep'] = prep
        return sc

    def w8(t, T):
        swing = float(np.clip((t - t_door + 0.2) / 0.25, 0, 1)) if t < t_door + 0.05 else 0.0
        return room_scene(pov8, t, T, room_env, door_deg=float(door8(t)), film_him=t > 6.0, swing=swing)

    out.append(S('w8_room', 10.0, '3d', scene=w8, hud=HOTBAR,
                 subs=[(3.4, 5.4, 'This is my room.'), (5.8, 9.4, 'My actual room. My desk. My...')],
                 cues=[(t_door, 'door_open'), (1.5, 'amb_fade', {'to': 'room', 'dur': 3.0}), (7.0, 'pc_hum_start')] +
                 [(ts, 'step', {'surface': 'carpet'}) for ts in pov8.steps(1.4, 7.0)]))

    # --- w9: the screen: a video player, at this minute of this video, and on it: him, from behind ------------------------------------
    lean = behind_chair + np.array([0.02, 0.5, 0.0])
    a_lean = A.look_angles(lean + (0, 0, 1.62), mon.centre)
    pov9 = A.POV(pos_keys=[(0, tuple(behind_chair)), (4.0, tuple(lean)), (10.0, tuple(lean))],
                 yaw_keys=[(0, a_mon[0]), (4.0, a_lean[0]), (10.0, a_lean[0])],
                 pitch_keys=[(0, a_mon[1]), (4.0, a_lean[1]), (10.0, a_lean[1])], seed=101, jitter=0.05)
    pov9.fov_keys = A.Keys([(0, 70.0), (4.0, 60.0), (10.0, 56.0)])

    def w9(t, T):
        sc = room_scene(pov9, t, T, room_env, film_him=True)
        return sc

    out.append(S('w9_screen', 10.0, '3d', scene=w9, hud=HOTBAR,
                 subs=[(3.0, 5.0, "That's me."), (5.6, 9.6, "It's filming me. From behind me. Right now.")],
                 cues=[(1.0, 'drone', {'level': 0.4, 'dur': 9.0}), (5.4, 'sting_note')]))
    return out
