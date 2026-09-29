"""ACT 6: the second player (7:58 - 9:15).

<NOAH_404> don't move. <NOAH_404> it can see your screen. A multiplayer bug? The tab list shows one player: YOU.
<NOAH_404> I'm not in your world. ... <NOAH_404> I'm underneath it. The pit behind the tenth lectern, a ladder down
into the dark, a pillar hanging in a vast chamber, and far below, lit blue, someone standing at the centre. Noah's
skin, his back to us. He walks up to it. It doesn't move. <NOAH_404> don't look at him. He looks. It turns, slowly:
his own face. The game freezes.
"""
import numpy as np

import anim as A
import common as C
import entities as EN
import film as FM
import ui
from seq_records import HOTBAR_P
from seq_under import ZF, torch_pov

HOTBAR_PT = dict(HOTBAR_P, selected=1)          # the torch again (the page in the sixth slot)
WHITE = (255, 255, 255)


def chamber_lights():
    """The glow of the four soul lanterns round the figure (the light volume alone leaves it in the dark)."""
    from worldgen import CHAMBER
    cx, cy = CHAMBER['c']
    out = []
    for k in range(4):
        a = k / 4 * 2 * np.pi + 0.4
        x, y = int(round(cx + 3.5 * np.cos(a))), int(round(cy + 3.5 * np.sin(a)))
        out.append([x + 0.5, y + 0.5, CHAMBER['zf'] + 0.7, 10.0, 0.45, 1.3, 1.7])
    # and a cold glow over the platform, from nowhere: the figure shows from the far end of the chamber
    out.append([cx + 0.5, cy + 0.5, CHAMBER['zf'] + 7.0, 17.0, 0.5, 1.4, 2.0])
    return out


def cham_pov(pov, t, actors=(), torch=0.55, **kw):
    cam = pov.cam(t)
    return C.pov_scene(pov, t, 'chamber', item='torch', actors=list(actors),
                       lights=chamber_lights() + C.held_torch_light(cam, t, strength=torch), **kw)


def msg(text):
    return ('chat', {'text': f'<NOAH_404> {text}', 'color': WHITE})


def shots(ctx):
    S = FM.Shot
    out = []
    w = ctx.worlds['village']
    P = w.points
    props = C.world_props(ctx, 'village', ceiling='gone', copy_door=88.0)
    c14 = P['lchest_14']
    by14 = np.array([c14[0] - 1.8, c14[1] - 1.3, ZF])
    pit = P['pit']                                     # centre of the 3 x 3 hole; the ladder runs down its north side
    lad_x, lad_y = pit[0], pit[1] - 0.05
    foot = P['chamber_ladder_foot']
    fig = P['figure']
    zc = -96.0

    a_pit = A.look_angles(by14 + (0, 0, 1.62), pit + (0, 0, -0.5))

    # --- s1: don't move ------------------------------------------------------------------------------------------------
    pov1 = A.POV(pos_keys=[(0, tuple(by14))], yaw_keys=[(0, a_pit[0]), (7.0, a_pit[0])],
                 pitch_keys=[(0, a_pit[1]), (7.0, a_pit[1])], seed=80, jitter=0.0)

    def s1(t, T):
        return C.pov_scene(pov1, t, 'underground', props=props, item='torn_page')

    out.append(S('s1_dont_move', 7.0, '3d', scene=s1, hud=HOTBAR_P, chat=True,
                 cues=[(0.6, 'chat_msg'), (0.6, *msg("don't move")), (0.6, 'silence_all', {'dur': 3.6}),
                       (4.4, 'chat_msg'), (4.4, *msg('it can see your screen'))]))

    # --- s2: a server? The tab list: one player -------------------------------------------------------------------------------
    pov2 = A.POV(pos_keys=[(0, tuple(by14))], yaw_keys=[(0, a_pit[0]), (1.5, a_pit[0] - 25), (3.2, a_pit[0] + 20),
                                                       (6.0, a_pit[0] + 14)],
                 pitch_keys=[(0, a_pit[1]), (1.5, 2), (6.0, 0)], seed=81, jitter=0.12)
    t_tab0, t_tab1 = 2.6, 5.4

    def s2(t, T):
        return C.pov_scene(pov2, t, 'underground', props=props, item='torn_page')

    def s2_tab(img, t, T, ctx_, film):
        if t_tab0 <= t < t_tab1:
            ui.player_list(img, ['YOU'])

    out.append(S('s2_tab', 6.0, '3d', scene=s2, hud=HOTBAR_P, chat=True, late=s2_tab,
                 subs=[(0.6, 2.4, 'Is this a server? Some multiplayer bug?'),
                       (3.4, 5.8, '(whispering) How is he messaging me?')],
                 cues=[(t_tab0, 'click_soft')]))

    # --- s3: I'm not in your world. ... I'm underneath it ---------------------------------------------------------------------
    pov3 = A.POV(pos_keys=[(0, tuple(by14))],
                 yaw_keys=[(0, a_pit[0] + 14), (1.0, a_pit[0] + 10), (5.6, a_pit[0]), (8.0, a_pit[0])],
                 pitch_keys=[(0, 0), (4.8, -8), (6.0, -52), (8.0, -56)], seed=82, jitter=0.07)

    def s3(t, T):
        return C.pov_scene(pov3, t, 'underground', props=props, item='torn_page')

    out.append(S('s3_underneath', 8.0, '3d', scene=s3, hud=HOTBAR_P, chat=True,
                 cues=[(0.5, 'chat_msg'), (0.5, *msg("I'm not in your world.")), (1.0, 'silence_all', {'dur': 3.6}),
                       (4.6, 'chat_msg'), (4.6, *msg("I'm underneath it.")), (5.0, 'drone', {'level': 0.35,
                                                                                            'dur': 3.0})]))

    # --- s4: to the pit behind the tenth lectern; the torch back in his hand -----------------------------------------------------
    edge = np.array([pit[0] - 2.2, pit[1] - 0.1, ZF])
    path4 = A.Path([tuple(by14), (by14[0] + 2.0, by14[1] - 2.0, ZF), (edge[0] - 3.5, 86.1, ZF),
                    (edge[0] - 0.6, 86.6, ZF), tuple(edge)])
    down = A.look_angles(edge + (0, 0, 1.62), np.array([lad_x, lad_y, ZF - 3.0]))
    pov4 = A.POV(path=path4, s_keys=[(0, 0.0), (5.0, path4.length), (8.0, path4.length)],
                 yaw_keys=[(0, a_pit[0]), (1.0, -70), (4.2, -80), (5.0, down[0]), (8.0, down[0] + 2)],
                 pitch_keys=[(0, -56), (1.0, -8), (4.2, -12), (5.4, down[1]), (8.0, down[1] - 4)], seed=83,
                 jitter=0.08)
    t_sw = 5.6

    def s4(t, T):
        if t < t_sw:
            return C.pov_scene(pov4, t, 'underground', props=props, item='torn_page')
        return torch_pov(pov4, t, 'underground', props=props)

    out.append(S('s4_pit', 8.0, '3d', scene=s4, hud=lambda t: HOTBAR_PT if t >= t_sw else HOTBAR_P,
                 subs=[(6.0, 7.8, '(whispering) Underneath.')],
                 cues=[(ts, 'step', {'surface': 'stone'}) for ts in pov4.steps(0, 5.0)] + [(t_sw, 'click_soft')]))

    # --- s5: down the ladder into the dark; below, the rock opens into nothing. Far down: someone, standing ------------------------
    lady = pit[1] - 0.1                               # the middle of the 1 x 1 shaft, facing the ladder
    pov5 = A.POV(pos_keys=[(0, (lad_x, lady, ZF - 0.6)), (0.8, (lad_x, lady, ZF - 2.2)), (9.0, (lad_x, lady, ZF - 30.0))],
                 yaw_keys=[(0, 0.0), (5.6, 0.0), (7.2, 158.0), (9.0, 162.0)],
                 pitch_keys=[(0, -58), (5.6, -64), (7.2, -38), (9.0, -40)], seed=84, jitter=0.06, bob=0.0)

    def s5(t, T):
        still = EN.Actor('player', 'noah', fig, yaw=np.radians(180.0))
        still.idle_t = 0.0
        return cham_pov(pov5, t, actors=[still])

    out.append(S('s5_ladder', 9.0, '3d', scene=s5, hud=HOTBAR_PT,
                 fx=lambda t: {'fade': float(np.clip((9.0 - t) / 0.6, 0, 1))},
                 cues=[(0.3 + 0.36 * k, 'ladder') for k in range(24)] + [(0.0, 'amb', {'kind': 'chamber'}),
                                                                       (6.8, 'sting_low')]))

    # --- s6: the bottom. He turns: the chamber, the blue lights, the figure with its back to him ----------------------------------
    stand6 = np.array([foot[0], foot[1] - 0.8, zc])
    a_fig = A.look_angles(stand6 + (0, 0, 1.62), fig + (0, 0, 1.3))
    pov6 = A.POV(pos_keys=[(0, (lad_x, lady, zc + 1.2)), (0.8, (lad_x, lady, zc)), (1.6, tuple(stand6)),
                           (10.0, tuple(stand6))],
                 yaw_keys=[(0, 0.0), (1.2, 0.0), (3.6, a_fig[0]), (10.0, a_fig[0] + 1)],
                 pitch_keys=[(0, -30), (1.2, -10), (3.6, a_fig[1]), (10.0, a_fig[1])], seed=85, jitter=0.07)

    def figure(skin='noah', yaw_deg=180.0, head=0.0):
        a = EN.Actor('player', skin, fig, yaw=np.radians(yaw_deg))
        a.idle_t = 0.0                      # no breathing, no swaying: completely still
        a.head_yaw = head
        a.head_pitch = 0.1 if skin == 'you' else 0.0      # it looks at him from under its brow
        return a

    def s6(t, T):
        return cham_pov(pov6, t, actors=[figure()])

    out.append(S('s6_chamber', 10.0, '3d', scene=s6, hud=HOTBAR_PT,
                 fx=lambda t: {'fade': float(np.clip(t / 0.8, 0, 1))},
                 subs=[(5.0, 7.4, "There's someone else here.")],
                 cues=[(0.8, 'step', {'surface': 'deepslate'}), (0.0, 'silence_all', {'dur': 2.0}),
                       (8.4, 'hurt_far', {'pan': 0.7, 'dist': 60})]))

    # --- s7: he walks up to it. It doesn't move. <NOAH_404> don't look at him ------------------------------------------------------
    stop = np.array([fig[0], fig[1] + 2.5, zc])
    path7 = A.Path([tuple(stand6), (fig[0] + 1.5, 81.2, zc), (fig[0] + 0.8, 70.0, zc), tuple(stop)])   # round the lantern
    t_dont = 9.0
    a_head = A.look_angles(stop + (0, 0, 1.62), fig + (0, 0, 1.55))
    pov7 = A.POV(path=path7, s_keys=[(0, 0.0), (8.6, path7.length), (12.0, path7.length)],
                 yaw_keys=[(0, a_fig[0] + 1), (8.6, a_head[0]), (12.0, a_head[0])],
                 pitch_keys=[(0, a_fig[1]), (8.6, a_head[1] - 6), (t_dont + 0.3, a_head[1] - 8), (t_dont + 1.1, -62),
                             (12.0, -64)], seed=86, jitter=0.06)

    def s7(t, T):
        return cham_pov(pov7, t, actors=[figure()])

    out.append(S('s7_approach', 12.0, '3d', scene=s7, hud=HOTBAR_PT, chat=True,
                 cues=[(ts, 'step', {'surface': 'deepslate', 'gain': 0.6}) for ts in pov7.steps(0, 8.6)] +
                 [(t_dont, 'chat_msg'), (t_dont, *msg("don't look at him")), (t_dont, 'silence_all', {'dur': 3.0})]))

    # --- s8: he looks. It turns, slowly. His face -----------------------------------------------------------------------------------
    t_turn0, t_turn1, t_swap = 3.0, 8.4, 5.6
    pov8 = A.POV(pos_keys=[(0, tuple(stop)), (11.0, tuple(stop + (0, 0.12, 0)))],
                 yaw_keys=[(0, a_head[0]), (11.0, a_head[0])],
                 pitch_keys=[(0, -64), (1.2, -40), (3.0, a_head[1]), (11.0, a_head[1] + 0.5)], seed=87, jitter=0.03)
    pov8.fov_keys = A.Keys([(0, 70.0), (3.0, 68.0), (8.4, 50.0), (11.0, 47.0)])      # the world closes in

    def turned(t):
        u = float(A.smoother((t - t_turn0) / (t_turn1 - t_turn0)))
        body = 180.0 - 180.0 * u
        head = np.radians(-40.0) * np.sin(np.pi * u)          # the head leads the turn
        return body, head

    def s8(t, T):
        body, head = turned(t)
        skin = 'noah' if t < t_swap else 'you'
        return cham_pov(pov8, t, actors=[figure(skin, body, head)])

    def s8_fx(t):
        fr = int(round(t * FM.FPS))
        sw = int(round(t_swap * FM.FPS))
        if fr in (sw - 1, sw):
            return {'tear': 0.8, 'vhs': 0.5}                   # two frames: was that a glitch?
        return {}

    out.append(S('s8_turn', 11.0, '3d', scene=s8, hud=HOTBAR_PT, chat=True, fx=s8_fx,
                 cues=[(t_turn0, 'turn_creak', {'dur': t_turn1 - t_turn0}), (t_swap, 'glitch_tick'),
                       (t_turn1, 'silence_all', {'dur': 2.6})]))

    # --- s9: the game freezes ---------------------------------------------------------------------------------------------------
    out.append(S('s9_freeze', 4.5, 'still', scene=s8, still=10.95, hud=HOTBAR_PT,
                 fx=lambda t: {'freeze': float(np.clip(t / 1.2, 0, 1)) * 0.8, 'fade': 0.0 if t > 4.1 else 1.0},
                 cues=[(0.0, 'freeze_buzz', {'dur': 4.1})]))
    return out
