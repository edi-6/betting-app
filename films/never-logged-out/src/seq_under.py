"""ACT 4: the underground room (4:59 - 6:38).

Past midnight (rule one). Air is coming up through the floor: the carpet lifts at one edge. Under it a trapdoor, a
ladder down a shaft, and at the bottom a stair lit with torches that someone lit, going down and down. It opens into a
cavern holding the same house over and over, street after street, each with a date on a sign: 2018... 2020...
2023... 2025... NOAH_404... and today. Inside today's copy: the broken ceiling block and the fallen sign, exactly
as it happened tonight, and on the wall, YOU WILL SLEEP HERE TONIGHT. A second bed: "This bed is occupied". Next
door, a house dated tomorrow.
"""
import numpy as np

import anim as A
import common as C
import entities as EN
import film as FM
import ui

HOTBAR = dict(items=['written_book', 'torch', 'bread', 'clock', 'filled_map'], selected=5, counts={1: 16, 2: 5})
HOTBAR_T = dict(HOTBAR, selected=1)                  # the torch in his hand from the ladder on
COPY_O = (18, 63)            # the copy dated today (u 0..8 -> x 18..26, v 0..10 -> y 63..73), floor surface z = -40
TOMORROW_SIGN = (37, 61)
ZF = -40.0


def carpet_lift(pos, ang_deg, tex='red_wool'):
    """The carpet over the trapdoor, its west edge lifted by the draught (hinged on its east edge)."""
    at = C.atlas()
    a = np.radians(ang_deg)
    hinge = np.array([pos[0] + 1.0, pos[1] + 0.5, pos[2] + 1 / 32])
    c = hinge + np.array([-0.5 * np.cos(a), 0.0, 0.5 * np.sin(a)])
    return [C.row('prop_cube', c, EN.qaxis((0, 1, 0), a), (1.0, 1.0, 1 / 16), at[tex])]


def torch_pov(pov, t, env_name, **kw):
    """POV holding a torch that lights the way (a dynamic-lights mod)."""
    cam = pov.cam(t)
    return C.pov_scene(pov, t, env_name, item='torch', lights=C.held_torch_light(cam, t, strength=0.3), **kw)


def stair_z(y):
    """Feet height walking down the stair from the shaft (y = -31.5, z = -6) to the landing (y = 3, z = -40)."""
    return float(np.clip(-36.0 - y, -40.0, -6.0))


def shots(ctx):
    S = FM.Shot
    out = []
    w = ctx.worlds['village']
    P = w.points
    ceil = w.dyn['house_ceiling']['pos']
    trap = w.dyn['house_trapdoor']['pos']
    tc = np.array([trap[0] + 0.5, trap[1] + 0.5, 0.0])               # the trapdoor's centre, on the floor

    def house_props(t_=0.0, carpet=True, trap_deg=0.0):
        props = C.world_props(ctx, 'village', ceiling='gone', carpet=carpet, trap=trap_deg)
        props += C.sign_prop(ctx, 'village', 'sign_good', (ceil[0], ceil[1], 0), 0)
        return props

    night = dict(light_col=(0.42, 0.52, 0.78))

    # --- u1: the clock on the wall: past midnight ----------------------------------------------------------------------
    clock = np.array([1.5, -24.0, 2.5])
    e1 = np.array([1.5, -26.3, 1.62])
    pov1 = A.POV(pos_keys=[(0, (0.5, -29.9, 0.0)), (1.0, (0.6, -29.4, 0.0)), (3.4, (1.5, -26.3, 0.0)),
                           (7.0, (1.5, -26.1, 0.0))],
                 yaw_keys=[(0, 1), (2.0, 6), (3.4, A.look_angles(e1, clock)[0]), (7.0, 0)],
                 pitch_keys=[(0, -52), (1.6, -12), (3.4, A.look_angles(e1, clock)[1]), (7.0, 20)],
                 seed=41, jitter=0.08)

    def u1(t, T):
        return C.pov_scene(pov1, t, 'interior_night', props=house_props(), envkw=night)

    out.append(S('u1_clock', 7.0, '3d', scene=u1, hud=HOTBAR,
                 subs=[(3.6, 5.0, "It's past midnight."), (5.2, 7.0, 'Rule one...')],
                 cues=[(ts, 'step', {'surface': 'wood'}) for ts in pov1.steps(0, 3.4)] +
                 [(0.0, 'amb', {'kind': 'night_in'}), (3.6, 'clock_tick', {'dur': 3.4})]))

    # --- u2: the carpet by the door lifts at one edge: air from below. He punches it away: a trapdoor -------------------
    stand2 = np.array([tc[0] + 1.2, tc[1] + 0.2, 0.0])
    look2 = A.look_angles(stand2 + (0, 0, 1.62), tc)
    look2_far = A.look_angles(np.array([1.5, -26.1, 1.62]), tc)
    t_break = 6.8
    pov2 = A.POV(pos_keys=[(0, (1.5, -26.1, 0.0)), (2.6, (1.5, -26.1, 0.0)), (5.0, tuple(stand2)),
                           (8.0, tuple(stand2))],
                 yaw_keys=[(0, 0), (1.8, look2_far[0]), (5.0, look2[0]), (8.0, look2[0] + 2)],
                 pitch_keys=[(0, 20), (1.8, look2_far[1]), (2.6, look2_far[1]), (5.0, look2[1]), (8.0, look2[1] - 3)],
                 seed=42, jitter=0.08)

    def lift(t):
        if t >= t_break:
            return None
        u = np.clip((t - 1.0) / 1.5, 0, 1)
        return float(u * max(0.0, 2.5 + 3.5 * np.sin(2 * np.pi * 0.42 * t) + 1.2 * np.sin(2 * np.pi * 1.3 * t)))

    def u2(t, T):
        props = house_props(carpet=False)
        a = lift(t)
        if a is not None:
            props += carpet_lift(trap[:2] + (0,), a)
        props += C.breaking_particles((trap[0], trap[1], -0.45), t - t_break, 'red_wool', n=10, dur=0.8)
        swing = float(np.clip((t - t_break + 0.2) / 0.25, 0, 1)) if t < t_break + 0.05 else 0.0
        return C.pov_scene(pov2, t, 'interior_night', props=props, swing=swing, envkw=night)

    out.append(S('u2_carpet', 8.0, '3d', scene=u2, hud=HOTBAR,
                 subs=[(2.8, 5.2, "...There's air coming up through the floor.")],
                 cues=[(ts, 'step', {'surface': 'wood'}) for ts in pov2.steps(2.6, 5.0)] +
                 [(1.0, 'draft', {'dur': 5.8}), (t_break, 'punch'), (t_break, 'wool_break')]))

    # --- u3: he opens the trapdoor: a shaft, a ladder, and far down, the glow of torches someone lit -------------------------
    t_open = 1.0
    trap3 = A.Keys([(0, 0.0), (t_open, 0.0), (t_open + 0.22, 90.0, 'out')])
    edge = np.array([tc[0] + 0.3, tc[1] + 0.1, 0.0])
    pov3 = A.POV(pos_keys=[(0, tuple(stand2)), (2.0, tuple(stand2)), (3.6, tuple(edge)), (7.0, tuple(edge))],
                 yaw_keys=[(0, look2[0] + 2), (3.6, 90.0), (7.0, 91.0)],
                 pitch_keys=[(0, look2[1] - 3), (2.0, look2[1]), (3.6, -80.0), (7.0, -83.0)], seed=43, jitter=0.05)
    pov3.eye_keys = A.Keys([(0, 1.62), (2.4, 1.62), (3.4, 1.27)])      # he crouches at the edge

    def u3(t, T):
        swing = float(np.clip((t - t_open + 0.2) / 0.25, 0, 1)) if t < t_open + 0.05 else 0.0
        return C.pov_scene(pov3, t, 'interior_night', props=house_props(carpet=False, trap_deg=float(trap3(t))),
                           swing=swing, envkw=night)

    out.append(S('u3_trapdoor', 7.0, '3d', scene=u3, hud=HOTBAR,
                 subs=[(4.0, 6.6, 'Someone lit torches down there.')],
                 cues=[(t_open, 'trapdoor_open'), (t_open + 0.3, 'draft', {'dur': 5.5, 'level': 0.6}),
                       (3.0, 'amb_duck', {'to': 0.3})]))

    # --- u4: down the ladder, facing it; at the bottom he turns round: the stair, going down and down ----------------------
    lad = np.array([tc[0], tc[1] + 0.05, 0.0])
    pov4 = A.POV(pos_keys=[(0, (lad[0], lad[1], 0.2)), (0.6, (lad[0], lad[1], -0.4)), (3.6, (lad[0], lad[1], -6.0)),
                           (6.0, (lad[0], lad[1] + 0.2, -6.0))],
                 yaw_keys=[(0, 180.0), (3.7, 180.0), (5.0, 0.0), (6.0, 2.0)],
                 pitch_keys=[(0, -45.0), (1.0, -38.0), (3.4, -30.0), (4.6, -10.0), (6.0, -24.0)], seed=44,
                 jitter=0.07, bob=0.0)

    def u4(t, T):
        return torch_pov(pov4, t, 'underground', props=house_props(carpet=False, trap_deg=90.0))

    out.append(S('u4_ladder', 6.0, '3d', scene=u4, hud=HOTBAR_T,
                 cues=[(0.35 + 0.42 * k, 'ladder') for k in range(8)] + [(0.0, 'amb', {'kind': 'tunnel'})]))

    # --- u5: the stair: 34 steps down, a torch every five --------------------------------------------------------------------
    y0, y1 = lad[1] + 0.25, 4.6
    ys = np.linspace(y0, y1, 40)
    path5 = A.Path([(lad[0], y, stair_z(y)) for y in ys])
    pov5 = A.POV(path=path5, s_keys=[(0, 0.0), (1.2, 1.2), (9.6, path5.length - 2.2), (11.0, path5.length)],
                 yaw_keys=[(0, 2), (3.0, -4), (6.0, 5), (9.0, -2), (11.0, 0)],
                 pitch_keys=[(0, -24), (2.0, -30), (8.5, -28), (10.2, -8), (11.0, -4)], seed=45, jitter=0.1)

    def u5(t, T):
        return torch_pov(pov5, t, 'underground')

    out.append(S('u5_stairs', 11.0, '3d', scene=u5, hud=HOTBAR_T,
                 subs=[(1.5, 4.2, 'This goes way deeper than a basement.')],
                 cues=[(ts, 'step', {'surface': 'stone', 'echo': 0.6}) for ts in pov5.steps(0, 11.0, stride=0.9)] +
                 [(8.0, 'drone', {'level': 0.2, 'dur': 3.0})]))

    # --- u6: into the cavern: the reveal ---------------------------------------------------------------------------------------
    pov6 = A.POV(pos_keys=[(0, (lad[0], 4.6, ZF)), (3.2, (lad[0], 8.6, ZF)), (9.0, (lad[0], 8.8, ZF))],
                 yaw_keys=[(0, 0), (3.2, 0), (5.2, 24), (7.2, -22), (9.0, -4)],
                 pitch_keys=[(0, -4), (3.2, 6), (5.2, 9), (9.0, 5)], seed=46, jitter=0.1)

    def u6(t, T):
        return torch_pov(pov6, t, 'underground', props=C.world_props(ctx, 'village', ceiling='gone'))

    out.append(S('u6_reveal', 9.0, '3d', scene=u6, hud=HOTBAR_T,
                 subs=[(4.8, 7.2, 'What... is this?')],
                 cues=[(ts, 'step', {'surface': 'stone', 'echo': 0.8}) for ts in pov6.steps(0, 3.2, stride=0.9)] +
                 [(0.0, 'amb', {'kind': 'cavern'}), (1.8, 'cave_reveal')]))

    # --- u6b: wide, from up under the rock: the streets of houses, and him, tiny, at the way in ------------------------------
    eye6b = A.Keys([(0, (-0.55, 4.6, ZF + 2.55)), (6.0, (-0.75, 5.4, ZF + 2.7))])
    tgt6b = A.Keys([(0, (-1.8, 44.0, ZF + 0.5)), (6.0, (-1.7, 44.0, ZF + 1.0))])

    def u6b(t, T):
        you = EN.Actor('player', 'you', (lad[0], 8.8, ZF), yaw=np.radians(-4))
        you.idle_t = t
        hand = np.array([lad[0] + 0.35, 9.1, ZF + 0.95])
        flick = 0.9 + 0.1 * np.sin(t * 13.0) * np.sin(t * 7.3 + 1.0)
        return dict(world='village', env=C.env('underground', exposure=1.35), actors=[you],
                    cam=dict(eye=eye6b(t), target=tgt6b(t), fov=58),
                    lights=[[*hand, 11.0, 1.0 * flick, 0.63 * flick, 0.32 * flick]],
                    props=C.world_props(ctx, 'village', ceiling='gone'))

    out.append(S('u6b_streets', 6.0, '3d', scene=u6b,
                 subs=[(1.0, 4.4, "It's the same house. Over and over.")]))

    # --- u7: the date signs (close, one after another) ---------------------------------------------------------------------
    # (date, sign, duration, line, (distance, sideways, eye height) at the start and at the end)
    signs = [('2018-06-14', (-38, 10), 2.6, "They're dated.", (1.25, -0.35, 0.95), (1.0, -0.3, 0.9)),
             ('2020-11-03', (-38, 27), 1.8, None, (1.3, 0.1, 1.3), (1.15, 0.05, 1.3)),
             ('2023-01-01', (-38, 44), 1.7, None, (1.2, 0.75, 1.2), (1.05, 0.6, 1.2)),
             ('2025-07-01', (-23, 61), 1.8, None, (1.25, -0.1, 1.35), (1.1, -0.05, 1.35)),
             ('NOAH_404', (-8, 61), 3.0, '...Noah?', (1.7, 0.0, 1.35), (1.05, 0.0, 1.3))]
    for k, (date, (sx, sy), dur, line, a0, a1) in enumerate(signs):
        board = np.array([sx + 0.5, sy + 0.5 - 0.06, ZF + 0.82])

        def spot(a, board=board):
            d, side, h = a
            return board[:2] + np.array([side, -d]), h

        (p0, h0), (p1, h1) = spot(a0), spot(a1)
        e0, e1 = np.array([*p0, ZF + h0]), np.array([*p1, ZF + h1])
        aim = board + (0, 0, 0.16)                  # the text sits just under the crosshair
        pk = A.POV(pos_keys=[(0, (*p0, ZF)), (dur, (*p1, ZF))],
                   yaw_keys=[(0, A.look_angles(e0, aim)[0]), (dur, A.look_angles(e1, aim)[0])],
                   pitch_keys=[(0, A.look_angles(e0, aim)[1]), (dur, A.look_angles(e1, aim)[1])],
                   seed=50 + k, jitter=0.05)
        pk.eye_keys = A.Keys([(0, h0), (dur, h1)])

        def u7(t, T, pk=pk, board=board):
            sc = torch_pov(pk, t, 'underground', props=C.world_props(ctx, 'village', ceiling='gone'))
            sc['dof'] = dict(focus=float(np.linalg.norm(board - sc['cam']['eye'])), k=5.0, maxr=6)
            return sc

        cues = [(0.0, 'step', {'surface': 'stone', 'echo': 0.6})]
        if date == 'NOAH_404':
            cues += [(0.3, 'sting_note'), (0.3, 'drone', {'level': 0.35, 'dur': 3.0})]
        out.append(S(f'u7_sign{k + 1}', dur, '3d', scene=u7, hud=HOTBAR_T,
                     subs=[(0.3, dur - 0.1, line)] if line else [], cues=cues))

    # --- u8: the sign dated today, then the house behind it ---------------------------------------------------------------
    ox, oy = COPY_O
    s_today = np.array([ox + 4.5, oy - 2 + 0.5 - 0.06, ZF + 0.82])
    door_out = np.array([ox + 4.5, oy - 0.55, ZF])
    inside = np.array([ox + 4.5, oy + 2.6, ZF])
    hole = np.array([ox + 4.5, oy + 5.5, ZF + 3.0])
    fallen = np.array([ox + 4.5, oy + 5.5 - 0.06, ZF + 0.82])
    by_sign = np.array([ox + 4.5, oy + 4.3, ZF])                    # a step from the fallen sign
    copy_props = C.world_props(ctx, 'village', ceiling='gone')
    p8 = np.array([ox + 4.35, oy - 2.95, ZF])
    e8 = p8 + (0, 0, 1.3)
    pov8a = A.POV(pos_keys=[(0, tuple(p8)), (3.0, tuple(p8 + (0.1, 0.12, 0))), (6.0, (ox + 4.5, oy - 3.3, ZF))],
                  yaw_keys=[(0, A.look_angles(e8, s_today)[0] - 3), (2.8, A.look_angles(e8, s_today)[0]),
                            (6.0, 0.0)],
                  pitch_keys=[(0, A.look_angles(e8, s_today)[1] - 2), (2.8, A.look_angles(e8, s_today)[1]),
                              (4.6, 4), (6.0, 6)], seed=58, jitter=0.07)
    pov8a.eye_keys = A.Keys([(0, 1.3), (3.0, 1.3), (4.4, 1.62)])

    def u8a(t, T):
        sc = torch_pov(pov8a, t, 'underground', props=copy_props)
        if t < 3.0:
            sc['dof'] = dict(focus=float(np.linalg.norm(s_today - sc['cam']['eye'])), k=4.0, maxr=5)
        return sc

    out.append(S('u8_today', 6.0, '3d', scene=u8a, hud=HOTBAR_T,
                 subs=[(0.8, 2.9, "That's today.")],
                 cues=[(0.0, 'drone', {'level': 0.3, 'dur': 6.0}), (3.4, 'sting_low')]))

    # --- u8b: round the sign, in through the door: the hole in the ceiling, the sign on the floor ---------------------------
    t_door = 3.1
    door8 = A.Keys([(0, 0.0), (t_door, 0.0), (t_door + 0.3, 88.0, 'out')])
    ang_hole = A.look_angles(inside + (0, 0, 1.62), hole)
    ang_fallen = A.look_angles(by_sign + (0, 0, 1.62), fallen)
    to_door = [(ox + 4.5, oy - 3.3, ZF), (ox + 3.4, oy - 2.5, ZF), (ox + 3.6, oy - 1.1, ZF), tuple(door_out)]
    path8 = A.Path(to_door + [(ox + 4.5, oy + 0.8, ZF), tuple(inside), tuple(by_sign)])
    s_door = A.Path(to_door).length
    s_in = A.Path(to_door + [(ox + 4.5, oy + 0.8, ZF), tuple(inside)]).length
    pov8b = A.POV(path=path8, s_keys=[(0, 0.0), (2.6, s_door), (3.6, s_door), (5.4, s_in), (7.6, s_in),
                                      (8.6, path8.length), (10.0, path8.length)],
                  yaw_keys=[(0, 0), (0.9, 22), (2.0, -18), (2.6, 0), (10.0, 0)],
                  pitch_keys=[(0, 6), (2.6, -4), (3.6, -2), (5.4, 0), (6.4, ang_hole[1]), (7.4, ang_hole[1]),
                              (8.6, ang_fallen[1]), (10.0, ang_fallen[1] - 2)], seed=59, jitter=0.07)

    def u8b(t, T):
        swing = float(np.clip((t - t_door + 0.2) / 0.25, 0, 1)) if t < t_door + 0.05 else 0.0
        return torch_pov(pov8b, t, 'underground', swing=swing,
                         props=C.world_props(ctx, 'village', ceiling='gone', copy_door=float(door8(t))))

    out.append(S('u8_inside', 10.0, '3d', scene=u8b, hud=HOTBAR_T,
                 subs=[(5.6, 7.6, 'This is my house.'), (8.0, 9.9, 'This is... tonight.')],
                 cues=[(ts, 'step', {'surface': 'stone'}) for ts in pov8b.steps(0, 3.6)] +
                 [(ts, 'step', {'surface': 'wood'}) for ts in pov8b.steps(3.6, 10.0)] + [(t_door, 'door_open')]))

    # --- u9: from the fallen sign up to the back wall: YOU WILL SLEEP HERE TONIGHT ------------------------------------------
    pov9 = A.POV(pos_keys=[(0, tuple(by_sign)), (1.4, tuple(by_sign)), (7.0, tuple(by_sign + (0, 0.35, 0)))],
                 yaw_keys=[(0, 0), (7.0, 0.5)],
                 pitch_keys=[(0, ang_fallen[1] - 2), (1.0, ang_fallen[1] - 2), (2.6, 3.0), (7.0, 2.0)], seed=60,
                 jitter=0.06)

    def u9(t, T):
        return torch_pov(pov9, t, 'underground',
                         props=C.world_props(ctx, 'village', ceiling='gone', copy_door=88.0))

    out.append(S('u9_wall', 7.0, '3d', scene=u9, hud=HOTBAR_T,
                 subs=[(3.2, 6.2, "(whispering) No. No, I'm not.")],
                 cues=[(1.8, 'drone', {'level': 0.45, 'dur': 5.2})]))

    # --- u10: the second bed. "This bed is occupied." It's empty --------------------------------------------------------------
    bed2 = np.array([ox + 5.5, oy + 8.5, ZF + 0.4])
    stand10 = np.array([ox + 5.2, oy + 6.6, ZF])
    a10 = A.look_angles(stand10 + (0, 0, 1.62), bed2)
    t_click = 2.2
    path10 = A.Path([tuple(by_sign + (0, 0.35, 0)), (ox + 5.6, oy + 5.0, ZF), tuple(stand10)])
    pov10 = A.POV(path=path10, s_keys=[(0, 0.0), (1.6, path10.length), (6.0, path10.length)],
                  yaw_keys=[(0, 0.5), (1.6, a10[0]), (6.0, a10[0] - 2)],
                  pitch_keys=[(0, 1), (1.6, a10[1]), (6.0, a10[1] - 2)], seed=61, jitter=0.06)

    def u10(t, T):
        swing = float(np.clip((t - t_click + 0.2) / 0.25, 0, 1)) if t < t_click + 0.05 else 0.0
        return torch_pov(pov10, t, 'underground', swing=swing,
                         props=C.world_props(ctx, 'village', ceiling='gone', copy_door=88.0))

    def u10_over(img, t, T, ctx_, film):
        if t >= t_click:
            a = float(np.clip((5.6 - t) / 0.5, 0, 1))
            ui.action_bar(img, 'This bed is occupied', alpha=a)

    out.append(S('u10_bed', 6.0, '3d', scene=u10, hud=HOTBAR_T, late=u10_over, sub_y=ui.H - 340,
                 subs=[(3.4, 5.9, '...Occupied? By who?')],
                 cues=[(ts, 'step', {'surface': 'wood'}) for ts in pov10.steps(0, 1.6)] + [(t_click, 'bed_click')]))

    # --- u11: out again: the next house is dated tomorrow -------------------------------------------------------------------
    tx, ty = TOMORROW_SIGN
    s_tom = np.array([tx + 0.5, ty + 0.5 - 0.06, ZF + 0.82])
    end11 = np.array([tx + 0.35, ty - 0.85, ZF])
    path11 = A.Path([tuple(door_out), (ox + 5.5, oy - 0.9, ZF), (ox + 6.2, oy - 2.6, ZF), (ox + 10.0, oy - 3.6, ZF),
                     (ox + 13.0, oy - 3.6, ZF), (tx - 2.0, ty - 1.3, ZF), tuple(end11)])
    e11 = end11 + (0, 0, 1.3)
    pov11 = A.POV(path=path11, s_keys=[(0, 0.0), (5.4, path11.length - 0.3), (7.0, path11.length)],
                  yaw_keys=[(0, 0), (0.8, -60), (2.2, -84), (5.4, A.look_angles(e11, s_tom)[0]),
                            (7.0, A.look_angles(e11, s_tom)[0])],
                  pitch_keys=[(0, -4), (2.2, -2), (5.4, A.look_angles(e11, s_tom)[1]),
                              (7.0, A.look_angles(e11, s_tom)[1] - 1)], seed=62, jitter=0.08)
    pov11.eye_keys = A.Keys([(0, 1.62), (5.0, 1.62), (6.0, 1.3)])

    def u11(t, T):
        sc = torch_pov(pov11, t, 'underground',
                       props=C.world_props(ctx, 'village', ceiling='gone', copy_door=88.0))
        if t > 5.0:
            sc['dof'] = dict(focus=float(np.linalg.norm(s_tom - sc['cam']['eye'])), k=4.0, maxr=5)
        return sc

    out.append(S('u11_tomorrow', 7.0, '3d', scene=u11, hud=HOTBAR_T,
                 subs=[(4.8, 6.9, "...And that one's tomorrow.")],
                 cues=[(ts, 'step', {'surface': 'stone'}) for ts in pov11.steps(0, 7.0)] +
                 [(5.2, 'drone', {'level': 0.4, 'dur': 3.0})]))
    return out
