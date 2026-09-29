"""ACT 5: the recordings (6:38 - 7:58).

A page turns somewhere. At the north wall, a row of lecterns: DAY 1 to DAY 19. Noah's days: I found the village. The
villagers don't remember me. Something copies the houses. The copies are always one day ahead. Then days 5 to 17,
one after another, too fast to read (pause the video). DAY 18 is about him, and it is signed YOU. DAY 19: I found
out what it wants. The next page is torn out. He searches the chests and finds it: It doesn't want to kill us.
It wants another player. And in the chat: NOAH_404 joined the game.
"""
import numpy as np

import anim as A
import common as C
import film as FM
import ui
from seq_under import HOTBAR_T, ZF, torch_pov

INV = {27: ('written_book', 1), 28: ('torch', 16), 29: ('bread', 5), 30: ('clock', 1), 31: ('filled_map', 1)}
GUI_SUB_Y = 912                       # subtitles between a GUI and the hotbar
HOTBAR_P = dict(items=['written_book', 'torch', 'bread', 'clock', 'filled_map', 'torn_page'], selected=5,
                counts={1: 16, 2: 5})

DAYS = {
    1: "I found the village.",
    2: "The villagers don't remember me.",
    3: "Something copies the houses.",
    4: "The copies are always one day ahead.",
    5: "I tried to log out. The world loaded back in around me.",
    6: "There's a mark on the map where I'm going to stand. Before I get there.",
    7: "The maps change at night. BURNED is new.",
    8: "I don't sleep here anymore. The bed is always warm.",
    9: "The footsteps only come when I stop moving.",
    10: "I counted the houses. 19. This morning there were 20.",
    11: "Somebody wrote in my book. In my handwriting.",
    12: "don't turn around don't turn around don't turn around don't",
    13: "The tab list says 2 players. I checked every house.",
    14: "It walks like me now. It stops when I stop.",
    15: "There's a crater where I spawned. My house is at the bottom. My REAL house.",
    16: "It isn't copying the houses. It's copying us.",
    17: "Tomorrow someone new comes. I have to leave the rules for him.",
    18: "Someone new found the village today.\n\nHe thinks it's a corrupted save.",
    19: "I found out what it wants.",
}


def shots(ctx):
    S = FM.Shot
    out = []
    w = ctx.worlds['village']
    P = w.points
    props = C.world_props(ctx, 'village', ceiling='gone', copy_door=88.0)

    def lectern_view(d, dx=0.0):
        """His view of the book on lectern d (a still behind the book GUI)."""
        lp = P[f'lectern_{d}']
        feet = np.array([lp[0] + dx, lp[1] - 1.35, ZF])
        pov = A.POV(pos_keys=[(0, tuple(feet))], yaw_keys=[(0, A.look_angles(feet + (0, 0, 1.62), lp + (0, 0, 0.9))[0])],
                    pitch_keys=[(0, A.look_angles(feet + (0, 0, 1.62), lp + (0, 0, 0.9))[1])], seed=70 + d,
                    jitter=0.0)

        def sc(t, T):
            return torch_pov(pov, t, 'underground', props=props)
        return sc

    # --- r1: a page turns, far off. He looks up the street to the north wall -------------------------------------------
    start = np.array([37.35, 60.15, ZF])                     # where act 4 left him, crouched at the sign
    s_tom = np.array([37.5, 61.44, ZF + 0.82])
    a0 = A.look_angles(start + (0, 0, 1.3), s_tom)
    pov1 = A.POV(pos_keys=[(0, tuple(start)), (0.6, tuple(start)), (2.6, (31.3, 61.1, ZF)), (6.0, (31.2, 64.8, ZF))],
                 yaw_keys=[(0, a0[0]), (0.9, a0[0] + 10), (1.6, 70), (2.6, 60), (3.4, 0), (6.0, -2)],
                 pitch_keys=[(0, a0[1]), (0.9, -6), (2.6, 0), (3.4, 4), (6.0, 2)], seed=71, jitter=0.08)
    pov1.eye_keys = A.Keys([(0, 1.3), (0.9, 1.62)])

    def r1(t, T):
        return torch_pov(pov1, t, 'underground', props=props)

    out.append(S('r1_page', 6.0, '3d', scene=r1, hud=HOTBAR_T,
                 subs=[(0.9, 2.1, 'Wait.')],
                 cues=[(0.7, 'page_far')] + [(ts, 'step', {'surface': 'stone'}) for ts in pov1.steps(0.6, 6.0)]))

    # --- r1b: up the street to the lecterns: DAY 1, DAY 2, DAY 3... -----------------------------------------------------
    l1 = P['lectern_1']
    at1 = np.array([l1[0], l1[1] - 1.35, ZF])
    path1b = A.Path([(31.2, 75.0, ZF), (31.3, 80.5, ZF), (33.6, 84.2, ZF), tuple(at1)])
    row_look = A.look_angles(at1 + (-1.5, 0, 1.62), P['lectern_6'] + (0, 1.5, 1.8))
    a1 = A.look_angles(at1 + (0, 0, 1.62), l1 + (0, 0, 0.9))
    pov1b = A.POV(path=path1b, s_keys=[(0, 0.0), (3.8, path1b.length), (6.0, path1b.length)],
                  yaw_keys=[(0, 0), (2.0, -10), (3.4, row_look[0]), (4.8, row_look[0] + 4), (6.0, a1[0])],
                  pitch_keys=[(0, 3), (3.4, row_look[1]), (4.8, row_look[1]), (6.0, a1[1])], seed=72, jitter=0.08)

    def r1b(t, T):
        return torch_pov(pov1b, t, 'underground', props=props)

    out.append(S('r1b_lecterns', 6.0, '3d', scene=r1b, hud=HOTBAR_T,
                 subs=[(3.2, 5.6, 'Day one... day two...')],
                 cues=[(ts, 'step', {'surface': 'stone'}) for ts in pov1b.steps(0, 3.8)]))

    # --- r2: DAY 1 to DAY 4, read ----------------------------------------------------------------------------------------
    def book_late(d, n_pages=1, torn_at=None, signature=None):
        def late(img, t, T, ctx_, film):
            if torn_at is not None and t >= torn_at:
                turn = float(np.clip(1.0 - (t - torn_at) / 0.18, 0, 1))
                ui.book(img, '', 1, n_pages, torn=True, turn=turn, lectern=True)
            else:
                ui.book(img, DAYS[d], 0, n_pages, title=f'DAY {d}', lectern=True, signature=signature)
        return late

    for d, dur, sub in ((1, 3.8, None), (2, 3.6, None), (3, 3.6, None),
                        (4, 4.6, (2.6, 4.5, 'One day ahead...'))):
        out.append(S(f'r2_day{d}', dur, 'still', scene=lectern_view(d), still=0.0, late=book_late(d), hud=HOTBAR_T,
                     sub_y=GUI_SUB_Y,
                     subs=[sub] if sub else [], cues=[(0.0, 'book_open' if d == 1 else 'page')]))

    # --- r3: DAY 5 to DAY 17, flipped through (a frame-by-frame treat) ---------------------------------------------------------
    for d in range(5, 18):
        out.append(S(f'r3_day{d}', 0.6, 'still', scene=lectern_view(d), still=0.0, late=book_late(d), hud=HOTBAR_T,
                     cues=[(0.0, 'page')]))

    # --- r4: DAY 18: today. Signed YOU -------------------------------------------------------------------------------------
    out.append(S('r4_day18', 7.0, 'still', scene=lectern_view(18), still=0.0, late=book_late(18, signature='- YOU'),
                 hud=HOTBAR_T, sub_y=GUI_SUB_Y, subs=[(4.0, 6.8, "...I didn't write that.")],
                 cues=[(0.0, 'page'), (0.2, 'silence_all', {'dur': 3.8}), (4.0, 'drone', {'level': 0.35, 'dur': 3.0})]))

    # --- r5: DAY 19: I found out what it wants. The next page is torn out ----------------------------------------------------
    t_turn = 4.6
    out.append(S('r5_day19', 9.0, 'still', scene=lectern_view(19), still=0.0,
                 late=book_late(19, n_pages=2, torn_at=t_turn), hud=HOTBAR_T, sub_y=GUI_SUB_Y,
                 subs=[(5.4, 8.4, 'The next page is torn out.')],
                 cues=[(0.0, 'page'), (t_turn, 'page'), (t_turn, 'sting_low')]))

    # --- r6: he searches the chests: nothing... then the page -----------------------------------------------------------------
    l19 = P['lectern_19']
    c17, c14 = P['lchest_17'], P['lchest_14']
    by17 = np.array([c17[0] - 1.8, c17[1] - 1.3, ZF])
    by14 = np.array([c14[0] - 1.8, c14[1] - 1.3, ZF])
    a17 = A.look_angles(by17 + (0, 0, 1.62), c17 + (0, 0, 0.5))
    a14 = A.look_angles(by14 + (0, 0, 1.62), c14 + (0, 0, 0.5))
    path6a = A.Path([(l19[0], l19[1] - 1.35, ZF), (l19[0] + 3.0, l19[1] - 1.5, ZF), (by17[0] - 0.4, by17[1] - 1.6, ZF),
                     tuple(by17)])
    pov6a = A.POV(path=path6a, s_keys=[(0, 0.0), (2.6, path6a.length), (3.4, path6a.length)],
                  yaw_keys=[(0, 0), (0.6, -80), (2.0, -60), (2.6, a17[0]), (3.4, a17[0])],
                  pitch_keys=[(0, -20), (0.6, -8), (2.6, a17[1]), (3.4, a17[1])], seed=76, jitter=0.1)

    def r6a(t, T):
        swing = float(np.clip((t - 3.0) / 0.25, 0, 1)) if t < 3.3 else 0.0
        return torch_pov(pov6a, t, 'underground', props=props, swing=swing)

    out.append(S('r6_search', 3.4, '3d', scene=r6a, hud=HOTBAR_T,
                 cues=[(ts, 'step', {'surface': 'stone', 'fast': 1}) for ts in pov6a.steps(0, 2.6)]))
    junk = {0: ('coal', 7), 4: ('bread', 1), 11: ('apple', 2), 22: ('book', 1)}
    out.append(S('r6_chest1', 2.6, 'still', scene=r6a, still=3.35, hud=HOTBAR_T,
                 late=lambda img, t, T, c, f: ui.chest(img, junk, inventory=INV), sub_y=GUI_SUB_Y,
                 subs=[(1.1, 2.5, 'No...')], cues=[(0.0, 'chest_open')]))
    path6c = A.Path([tuple(by17), (by17[0] - 0.3, by17[1] - 2.2, ZF), (by14[0] - 0.6, by14[1] - 2.1, ZF), tuple(by14)])
    pov6c = A.POV(path=path6c, s_keys=[(0, 0.0), (2.4, path6c.length), (2.8, path6c.length)],
                  yaw_keys=[(0, a17[0]), (0.4, -85), (1.9, -80), (2.4, a14[0]), (2.8, a14[0])],
                  pitch_keys=[(0, a17[1]), (0.4, -6), (2.4, a14[1]), (2.8, a14[1])], seed=77, jitter=0.1)

    def r6c(t, T):
        swing = float(np.clip((t - 2.45) / 0.25, 0, 1)) if t < 2.75 else 0.0
        return torch_pov(pov6c, t, 'underground', props=props, swing=swing)

    out.append(S('r6_run', 2.8, '3d', scene=r6c, hud=HOTBAR_T,
                 cues=[(0.0, 'chest_close')] + [(ts, 'step', {'surface': 'stone', 'fast': 1})
                                                 for ts in pov6c.steps(0, 2.4)]))
    found = {3: ('coal', 2), 13: ('torn_page', 1), 20: ('clock', 1)}

    def chest2(img, t, T, c, f):
        took = t >= 2.9
        cont = {k: v for k, v in found.items() if not (took and k == 13)}
        hover = (13, [('Torn Page', (255, 255, 255))]) if 1.2 <= t < 2.9 else None
        ui.chest(img, cont, hover=hover, inventory={**INV, **({32: ('torn_page', 1)} if took else {})})

    out.append(S('r6_chest2', 3.6, 'still', scene=r6c, still=2.75, hud=HOTBAR_T, late=chest2,
                 cues=[(0.0, 'chest_open'), (2.9, 'item_pickup')]))

    # --- r7: the torn page. Both sides ---------------------------------------------------------------------------------------
    t_flip = 5.2

    def page_late(img, t, T, c, f):
        if t < t_flip:
            ui.torn_page(img, "It doesn't want to kill us.", side=0)
        else:
            ui.torn_page(img, 'It wants another player.', side=1,
                         turn=float(np.clip(1.0 - (t - t_flip) / 0.2, 0, 1)))

    out.append(S('r7_page', 11.0, 'still', scene=r6c, still=2.75, hud=HOTBAR_P, late=page_late,
                 cues=[(0.0, 'paper'), (0.6, 'drone', {'level': 0.2, 'dur': 4.4}), (4.2, 'silence_all', {'dur': 1.0}),
                       (t_flip, 'paper'), (t_flip + 0.4, 'sting_low'),
                       (t_flip + 0.4, 'drone', {'level': 0.4, 'dur': 5.4})]))

    # --- r8: NOAH_404 joined the game ------------------------------------------------------------------------------------------
    pit = P['pit']
    a_pit = A.look_angles(by14 + (0, 0, 1.62), pit + (0, 0, -0.5))
    pov8 = A.POV(pos_keys=[(0, tuple(by14)), (8.0, tuple(by14))],
                 yaw_keys=[(0, a14[0]), (2.2, a14[0] - 2), (5.2, a14[0] - 6), (8.0, a_pit[0])],
                 pitch_keys=[(0, a14[1]), (2.2, a14[1] + 6), (5.2, 0), (8.0, a_pit[1])], seed=78, jitter=0.05)

    def r8(t, T):
        return C.pov_scene(pov8, t, 'underground', props=props, item='torn_page')     # the page in his hand, no torch

    out.append(S('r8_joined', 8.0, '3d', scene=r8, hud=HOTBAR_P, chat=True,
                 subs=[(4.2, 6.4, "That's not possible.")],
                 cues=[(0.0, 'silence_all', {'dur': 2.0}), (2.0, 'join'),
                       (2.0, 'chat', {'text': 'NOAH_404 joined the game', 'color': ui.YELLOW})]))
    return out
